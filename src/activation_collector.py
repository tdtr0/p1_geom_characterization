"""
Activation Collection Module

Extracts and stores transformer activations using TransformerLens.
Supports multiple aggregation strategies and efficient HDF5 storage.
"""

import torch
import h5py
import numpy as np
import gc
from typing import List, Dict, Optional, Literal
from pathlib import Path
from tqdm import tqdm
import warnings


class ActivationCollector:
    """
    Collects and stores activations from transformer forward passes.

    Design decisions:
    - Store residual stream (not attention/MLP separately) because:
      1. It's the "main" representation at each layer
      2. Reduces storage by ~3x vs storing all components
      3. Sufficient for geometric analysis
    - Store both pre-MLP and post-MLP for each layer to capture full trajectory
    - Use float16 to halve storage (precision sufficient for SVD)
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        use_transformer_lens: bool = True
    ):
        """
        Initialize activation collector with a model.

        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on
            dtype: Data type for model weights
            use_transformer_lens: Whether to use TransformerLens (recommended)
        """
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.use_transformer_lens = use_transformer_lens

        print(f"Loading model: {model_name}")

        if use_transformer_lens:
            try:
                from transformer_lens import HookedTransformer
                self.model = HookedTransformer.from_pretrained(
                    model_name,
                    device=device,
                    dtype=dtype,
                    fold_ln=False,  # Keep LayerNorm separate for clearer analysis
                    center_writing_weights=False,
                    center_unembed=False,
                )
                self.model.eval()
                self.n_layers = self.model.cfg.n_layers
                self.d_model = self.model.cfg.d_model
                print(f"✓ Loaded via TransformerLens: {self.n_layers} layers, d_model={self.d_model}")
            except Exception as e:
                print(f"Warning: TransformerLens failed ({e}), falling back to transformers")
                self.use_transformer_lens = False

        if not self.use_transformer_lens:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            # Use auto device mapping when device is "cuda" to handle memory efficiently
            # This will split model across available GPU memory if needed
            device_map_arg = "auto" if device == "cuda" else device

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=device_map_arg,
                low_cpu_mem_usage=True,  # Reduce CPU memory usage during loading
            )
            self.model.eval()
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.n_layers = self.model.config.num_hidden_layers
            self.d_model = self.model.config.hidden_size
            print(f"✓ Loaded via transformers: {self.n_layers} layers, d_model={self.d_model}")
            if device_map_arg == "auto":
                print(f"  Device map: {self.model.hf_device_map}")

    def get_hook_names(self) -> List[str]:
        """Return hook names for residual stream at each layer."""
        if not self.use_transformer_lens:
            warnings.warn("Hook names only available for TransformerLens models")
            return []

        hooks = []
        for layer in range(self.n_layers):
            # Post-attention, pre-MLP
            hooks.append(f"blocks.{layer}.hook_resid_mid")
            # Post-MLP (full layer output)
            hooks.append(f"blocks.{layer}.hook_resid_post")
        return hooks

    def collect_activations(
        self,
        texts: List[str],
        aggregation: Literal["last_token", "mean", "all_tokens"] = "last_token",
        batch_size: int = 1,
        max_length: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Collect activations for a batch of texts.

        Args:
            texts: List of input strings
            aggregation: How to aggregate across sequence positions
                - "last_token": Use only final token (for causal models)
                - "mean": Mean-pool across all tokens
                - "all_tokens": Store full sequence (expensive)
            batch_size: Number of texts to process at once
            max_length: Maximum sequence length (None = model default)

        Returns:
            Dict mapping hook names to activation arrays
            Shape depends on aggregation:
                - "last_token"/"mean": (n_texts, d_model)
                - "all_tokens": (n_texts, max_seq_len, d_model)
        """
        if not self.use_transformer_lens:
            return self._collect_with_hooks(texts, aggregation, batch_size, max_length)

        hook_names = self.get_hook_names()
        all_activations = {name: [] for name in hook_names}

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Collecting activations"):
                batch = texts[i:i+batch_size]

                for text in batch:
                    # Tokenize
                    tokens = self.model.to_tokens(text, prepend_bos=True)

                    # Truncate if needed
                    if max_length is not None and tokens.shape[1] > max_length:
                        tokens = tokens[:, :max_length]

                    # Forward with cache
                    try:
                        _, cache = self.model.run_with_cache(
                            tokens,
                            names_filter=hook_names,
                            remove_batch_dim=False
                        )
                    except Exception as e:
                        print(f"Warning: Failed to process text (length {tokens.shape[1]}): {e}")
                        # Store NaN for failed samples
                        for name in hook_names:
                            if aggregation == "all_tokens":
                                all_activations[name].append(
                                    np.full((max_length or 512, self.d_model), np.nan, dtype=np.float16)
                                )
                            else:
                                all_activations[name].append(
                                    np.full(self.d_model, np.nan, dtype=np.float16)
                                )
                        continue

                    # Extract and aggregate
                    for name in hook_names:
                        act = cache[name]  # Shape: (1, seq_len, d_model)

                        if aggregation == "last_token":
                            act = act[0, -1, :]  # (d_model,)
                        elif aggregation == "mean":
                            act = act[0].mean(dim=0)  # (d_model,)
                        elif aggregation == "all_tokens":
                            act = act[0]  # (seq_len, d_model)

                        all_activations[name].append(
                            act.cpu().numpy().astype(np.float16)
                        )

        # Stack into arrays
        for name in hook_names:
            all_activations[name] = np.stack(all_activations[name])

        return all_activations

    def _collect_with_hooks(
        self,
        texts: List[str],
        aggregation: Literal["last_token", "mean", "all_tokens"],
        batch_size: int,
        max_length: Optional[int]
    ) -> Dict[str, np.ndarray]:
        """Collect activations using PyTorch hooks for transformers models."""
        # Storage for all activations
        all_layer_acts = {f"layer_{i}": [] for i in range(self.n_layers)}

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Collecting activations"):
                batch_texts = texts[i:i+batch_size]

                for text_idx_in_batch, text in enumerate(batch_texts):
                    text_global_idx = i + text_idx_in_batch
                    # Tokenize
                    tokens = self.tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True if max_length else False,
                        max_length=max_length
                    )

                    # Move to appropriate device (first device in device map)
                    device = next(self.model.parameters()).device
                    tokens = {k: v.to(device) for k, v in tokens.items()}

                    # Hook storage for this sample
                    layer_outputs = {}

                    def make_hook(layer_idx):
                        def hook(_module, _input, output):
                            # Extract hidden states from output
                            # For most models, output is either a tensor or tuple where first element is hidden states
                            if isinstance(output, tuple):
                                hidden_states = output[0]
                            else:
                                hidden_states = output

                            # Move to CPU and store
                            layer_outputs[f"layer_{layer_idx}"] = hidden_states.detach().cpu()
                        return hook

                    # Register hooks
                    handles = []
                    for layer_idx in range(self.n_layers):
                        layer = self.model.model.layers[layer_idx]
                        handle = layer.register_forward_hook(make_hook(layer_idx))
                        handles.append(handle)

                    try:
                        # Forward pass
                        _ = self.model(**tokens)

                        # Aggregate and store activations
                        for layer_idx in range(self.n_layers):
                            key = f"layer_{layer_idx}"
                            hidden = layer_outputs[key]

                            if aggregation == "last_token":
                                # Use last token
                                act = hidden[0, -1, :].numpy().astype(np.float16)
                            elif aggregation == "mean":
                                # Mean across sequence
                                act = hidden[0].mean(dim=0).numpy().astype(np.float16)
                            else:  # all_tokens
                                act = hidden[0].numpy().astype(np.float16)

                            all_layer_acts[key].append(act)

                    except Exception as e:
                        import traceback
                        print(f"\n⚠ FAILED sample {text_global_idx}:")
                        print(f"  Error: {type(e).__name__}: {e}")
                        print(f"  Text length: {len(text)} chars")
                        print(f"  Text preview: {text[:100]}...")
                        if len(text) > 1000:
                            print(f"  (Long text warning)")
                        print(f"  Traceback: {traceback.format_exc()}")
                        # Store NaN for failed samples
                        for layer_idx in range(self.n_layers):
                            if aggregation == "all_tokens":
                                all_layer_acts[f"layer_{layer_idx}"].append(
                                    np.full((max_length or 512, self.d_model), np.nan, dtype=np.float16)
                                )
                            else:
                                all_layer_acts[f"layer_{layer_idx}"].append(
                                    np.full(self.d_model, np.nan, dtype=np.float16)
                                )

                    finally:
                        # Remove hooks
                        for handle in handles:
                            handle.remove()

        # Convert to arrays
        return {
            name: np.stack(acts, axis=0)
            for name, acts in all_layer_acts.items()
        }

    def save_to_hdf5(
        self,
        activations: Dict[str, np.ndarray],
        filepath: str,
        metadata: Optional[Dict] = None,
        compression: str = "gzip"
    ):
        """
        Save activations with metadata for reproducibility.

        Args:
            activations: Dict of activation arrays
            filepath: Path to save HDF5 file
            metadata: Optional metadata dict
            compression: Compression algorithm (gzip recommended)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(filepath, 'w') as f:
            # Store activations
            for name, arr in activations.items():
                f.create_dataset(
                    name,
                    data=arr,
                    compression=compression,
                    compression_opts=4  # Good balance of speed/size
                )

            # Store metadata
            if metadata:
                meta_group = f.create_group('metadata')
                for key, value in metadata.items():
                    if isinstance(value, str):
                        meta_group.attrs[key] = value
                    elif isinstance(value, (list, np.ndarray)):
                        meta_group.create_dataset(key, data=value)
                    elif isinstance(value, (int, float, bool)):
                        meta_group.attrs[key] = value
                    else:
                        # Convert to string for unsupported types
                        meta_group.attrs[key] = str(value)

            # Always store model info
            f.attrs['model_name'] = self.model_name
            f.attrs['n_layers'] = self.n_layers
            f.attrs['d_model'] = self.d_model

        print(f"✓ Saved activations to {filepath} ({filepath.stat().st_size / 1e6:.1f} MB)")

    def load_from_hdf5(self, filepath: str) -> tuple[Dict[str, np.ndarray], Dict]:
        """
        Load activations and metadata from HDF5 file.

        Returns:
            Tuple of (activations_dict, metadata_dict)
        """
        activations = {}
        metadata = {}

        with h5py.File(filepath, 'r') as f:
            # Load activations
            for key in f.keys():
                if key != 'metadata':
                    activations[key] = f[key][:]

            # Load metadata
            if 'metadata' in f:
                meta_group = f['metadata']
                for key in meta_group.attrs:
                    metadata[key] = meta_group.attrs[key]
                for key in meta_group.keys():
                    metadata[key] = meta_group[key][:]

            # Load model info from root attrs
            for key in f.attrs:
                metadata[key] = f.attrs[key]

        return activations, metadata

    def __del__(self):
        """Clean up GPU memory on deletion."""
        if hasattr(self, 'model'):
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
