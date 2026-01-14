#!/usr/bin/env python3
"""
vast.ai Instance Launcher for ManiVer

This script:
1. Searches for best GPU instances (by performance or cost-effectiveness)
2. Filters by bandwidth (>200 Mbps) and VRAM (>24GB)
3. Shows best option and asks for confirmation
4. Creates instance with persistent storage mount
5. Runs collection script automatically

Usage:
    python scripts/vast_launcher.py search [--sort-by perf|cost] # Find best offers
    python scripts/vast_launcher.py launch [--sort-by perf|cost] # Launch instance
    python scripts/vast_launcher.py status                      # Check running instances
    python scripts/vast_launcher.py destroy <id>                # Destroy instance
"""

import subprocess
import json
import sys
import os
from pathlib import Path

# Configuration
MIN_BANDWIDTH_MBPS = 200
MIN_VRAM_GB = 24
MIN_DISK_GB = 300
PREFERRED_REGION = "US"
GIT_REPO = "https://github.com/YOUR_USERNAME/ManiVer.git"  # TODO: Update
WORKSPACE_DIR = "/workspace/maniver"

# On-start script for vast.ai instance
ONSTART_SCRIPT = '''#!/bin/bash
set -e

echo "=== ManiVer Phase 2 Pipeline Setup ==="

# Install dependencies
pip install torch transformers datasets h5py scipy scikit-learn tqdm pyyaml b2

# Clone or pull repo
if [ -d "{workspace}" ]; then
    cd {workspace}
    git pull
else
    git clone {repo} {workspace}
    cd {workspace}
fi

# Create data directories
mkdir -p data/trajectories data/checkpoints logs

# Setup B2 CLI automatically
if [ -f "configs/b2-configs.txt" ]; then
    echo "Setting up Backblaze B2 CLI..."
    bash scripts/setup_b2_on_vastai.sh
fi

# Show GPU info
nvidia-smi

echo ""
echo "=== Ready to run Phase 2 collection ==="
echo ""
echo "Commands:"
echo "  Full pipeline (collect + upload): bash scripts/run_phase2_pipeline.sh"
echo "  Collection only:                  python scripts/collect_trajectories_with_labels.py"
echo "  Upload existing data:             python scripts/b2_upload.py"
echo "  Download from B2:                 python scripts/b2_download.py --list-only"
echo ""
'''.format(workspace=WORKSPACE_DIR, repo=GIT_REPO)

# Collection script that runs in parallel on multi-GPU
PARALLEL_COLLECTION_SCRIPT = '''#!/bin/bash
set -e
cd {workspace}

# Get number of GPUs
N_GPUS=$(nvidia-smi -L | wc -l)
echo "Found $N_GPUS GPUs"

# Models to collect (split across GPUs)
MODELS=("olmo3_base" "olmo3_sft" "olmo3_rl_zero" "olmo3_think" "deepseek_r1")

# Launch parallel collection
for i in $(seq 0 $((N_GPUS-1))); do
    # Assign models to GPUs round-robin
    GPU_MODELS=""
    for j in $(seq $i $N_GPUS ${{#MODELS[@]}}); do
        if [ $j -lt ${{#MODELS[@]}} ]; then
            GPU_MODELS="$GPU_MODELS ${{MODELS[$j]}}"
        fi
    done

    if [ -n "$GPU_MODELS" ]; then
        echo "GPU $i: Collecting$GPU_MODELS"
        CUDA_VISIBLE_DEVICES=$i python scripts/collect_single_model.py $GPU_MODELS &
    fi
done

wait
echo "=== All collection complete ==="
'''.format(workspace=WORKSPACE_DIR)


def run_vastai_cmd(args: list) -> dict:
    """Run vastai CLI command and return JSON result."""
    cmd = ["vastai"] + args + ["--raw"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout) if result.stdout.strip() else {}
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        return {}
    except json.JSONDecodeError:
        # Some commands return non-JSON output
        return {"output": result.stdout}


def search_offers(sort_by: str = "cost"):
    """Search for GPU offers matching our criteria."""
    sort_map = {
        "cost": ("dlperf_usd", "lower is better"),
        "perf": ("dlperf", "higher is better"),
    }
    order, explanation = sort_map.get(sort_by, sort_map["cost"])

    print(f"Searching for GPU offers, sorted by {sort_by} ({order})...")
    print(f"  Min VRAM: {MIN_VRAM_GB} GB")
    print(f"  Min Bandwidth: {MIN_BANDWIDTH_MBPS} Mbps")
    print(f"  Min Disk: {MIN_DISK_GB} GB")
    print()

    # Search command with filters
    cmd = [
        "search", "offers",
        "--order", order,
        f"gpu_ram >= {MIN_VRAM_GB}",
        f"inet_down >= {MIN_BANDWIDTH_MBPS}",
        f"inet_up >= {MIN_BANDWIDTH_MBPS}",
        f"disk_space >= {MIN_DISK_GB}",
        "reliability >= 0.95",
        "verified = true",
    ]

    result = subprocess.run(
        ["vastai"] + cmd,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"Error searching: {result.stderr}")
        return []

    # Parse the tabular output
    lines = result.stdout.strip().split('\n')
    if len(lines) < 2:
        print("No offers found matching criteria")
        return []

    # Return raw output for display
    print(f"Top offers (sorted by {order} - {explanation}):")
    print("-" * 120)
    for line in lines[:11]:  # Header + top 10
        print(line)
    print("-" * 120)

    return lines[1:11]  # Return top 10 (skip header)


def parse_offer_line(line: str) -> dict:
    """Parse a vastai offer line into a dict."""
    parts = line.split()
    if len(parts) < 10:
        return {}

    return {
        "id": parts[0],
        "cuda_vers": parts[1],
        "num_gpus": parts[2].replace("x", ""),
        "gpu_name": parts[3],
        "gpu_ram": parts[4],
        "dlperf": parts[5],
        "cpu_ram": parts[6],
        "disk": parts[7],
        "dlperf_usd": parts[9] if len(parts) > 9 else "N/A",
        "inet_down": parts[10] if len(parts) > 10 else "N/A",
        "inet_up": parts[11] if len(parts) > 11 else "N/A",
    }


def confirm_instance(offer: dict) -> bool:
    """Ask user to confirm instance selection."""
    print()
    print("=" * 60)
    print("SELECTED INSTANCE:")
    print("=" * 60)
    print(f"  ID:           {offer.get('id', 'N/A')}")
    print(f"  GPU:          {offer.get('num_gpus', '?')}x {offer.get('gpu_name', 'Unknown')}")
    print(f"  VRAM:         {offer.get('gpu_ram', '?')} GB")
    print(f"  Disk:         {offer.get('disk', '?')} GB")
    print(f"  DLPerf:       {offer.get('dlperf', 'N/A')}")
    print(f"  DLPerf/$:     {offer.get('dlperf_usd', 'N/A')}")
    print(f"  Bandwidth:    {offer.get('inet_down', '?')} / {offer.get('inet_up', '?')} Mbps")
    print("=" * 60)
    print()

    response = input("Proceed with this instance? [y/N]: ").strip().lower()
    return response == 'y'


def create_instance(offer_id: str, disk_gb: int = 300):
    """Create a vast.ai instance."""
    print(f"Creating instance from offer {offer_id}...")

    # Create on-start script file
    onstart_path = Path("/tmp/maniver_onstart.sh")
    onstart_path.write_text(ONSTART_SCRIPT)

    cmd = [
        "vastai", "create", "instance", offer_id,
        "--image", "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel",
        "--disk", str(disk_gb),
        "--onstart-cmd", f"bash -c '{ONSTART_SCRIPT}'",
        "--raw"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error creating instance: {result.stderr}")
        return None

    try:
        data = json.loads(result.stdout)
        instance_id = data.get("new_contract")
        print(f"Instance created: {instance_id}")
        return instance_id
    except:
        print(f"Instance creation output: {result.stdout}")
        return None


def show_status():
    """Show running instances."""
    print("Running instances:")
    result = subprocess.run(
        ["vastai", "show", "instances"],
        capture_output=True,
        text=True
    )
    print(result.stdout)


def destroy_instance(instance_id: str):
    """Destroy an instance."""
    response = input(f"Destroy instance {instance_id}? [y/N]: ").strip().lower()
    if response != 'y':
        print("Cancelled")
        return

    result = subprocess.run(
        ["vastai", "destroy", "instance", instance_id],
        capture_output=True,
        text=True
    )
    print(result.stdout or "Instance destroyed")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1]
    
    sort_by = "cost"
    if "--sort-by" in sys.argv:
        try:
            idx = sys.argv.index("--sort-by")
            val = sys.argv[idx + 1]
            if val in ["perf", "cost"]:
                sort_by = val
            else:
                print(f"Warning: Invalid sort option '{val}'. Defaulting to 'cost'.")
        except (ValueError, IndexError):
            print("Warning: --sort-by flag used without a value. Defaulting to 'cost'.")


    if cmd == "search":
        search_offers(sort_by=sort_by)

    elif cmd == "launch":
        offers = search_offers(sort_by=sort_by)
        if not offers:
            return

        # Parse first (best) offer
        best = parse_offer_line(offers[0])
        if not best:
            print("Could not parse best offer")
            return

        if confirm_instance(best):
            instance_id = create_instance(best["id"])
            if instance_id:
                print()
                print("Next steps:")
                print(f"  1. Wait for instance to start: vastai show instances")
                print(f"  2. SSH into instance: vastai ssh {instance_id}")
                print(f"  3. Run collection: python scripts/collect_trajectories_with_labels.py")
        else:
            print("Cancelled")

    elif cmd == "status":
        show_status()

    elif cmd == "destroy":
        if len(sys.argv) < 3:
            print("Usage: vast_launcher.py destroy <instance_id>")
            return
        destroy_instance(sys.argv[2])

    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()
