#!/usr/bin/env python3
"""Quick verification of generation trajectory HDF5 files."""
import sys
import h5py
import numpy as np

path = sys.argv[1]
print(f"Checking: {path}")

with h5py.File(path, "r") as f:
    keys = sorted([k for k in f.keys() if k.startswith("sample_")])
    print(f"Samples: {len(keys)}")

    for k in keys:
        g = f[k]
        hs = g["hidden_states"]
        ent = g["entropy"]
        att = g["attention"]

        hs_nz = np.count_nonzero(hs[:])
        att_nz = np.count_nonzero(att[:])
        ent_nz = np.count_nonzero(ent[:])
        gen_len = g.attrs["gen_len"]

        print(f"\n  {k}: gen_len={gen_len}")
        print(f"    hidden_states: shape={hs.shape}, nonzero={hs_nz}")
        print(f"    attention:     shape={att.shape}, nonzero={att_nz}")
        print(f"    entropy:       nonzero={ent_nz}/{len(ent[:])}")

        # Check norms at a few steps
        for step in [0, min(5, hs.shape[0] - 1), hs.shape[0] - 1]:
            norm = np.linalg.norm(hs[step, -1, :].astype(np.float32))
            print(f"    step {step:3d} last-layer norm: {norm:.3f}")

        if hs_nz == 0:
            print("    *** WARNING: ALL ZEROS — hooks not capturing! ***")
        else:
            print("    OK: hidden states captured")
