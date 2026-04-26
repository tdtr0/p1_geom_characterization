#!/usr/bin/env python3
"""
Download activation trajectories from Backblaze B2 storage.

This script is designed to run on vast.ai GPU instances to download
previously collected activation data.

Usage:
    python scripts/b2_download.py [--remote-prefix PREFIX] [--local-dir DATA_DIR]

Examples:
    # Download all trajectories to default location
    python scripts/b2_download.py

    # Download specific experiment
    python scripts/b2_download.py --remote-prefix experiments/phase2/run1

    # Download to custom directory
    python scripts/b2_download.py --local-dir /workspace/maniver/data/trajectories
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Load B2 config
CONFIG_FILE = Path(__file__).parent.parent / "configs" / "b2-configs.txt"


def load_b2_config():
    """Load B2 configuration from config file."""
    config = {}
    with open(CONFIG_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, value = line.split('=', 1)
                config[key.strip()] = value.strip()
    return config


def authorize_b2(config):
    """Authorize B2 CLI with credentials from config."""
    print("Authorizing B2 CLI...")
    result = subprocess.run(
        ['b2', 'authorize-account', config['B2_KEY_ID'], config['B2_APP_KEY']],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"Error authorizing B2: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    print("✓ B2 authorized")


def list_available_files(bucket_name, remote_prefix):
    """List files available in B2 bucket."""
    print(f"\nListing files in b2://{bucket_name}/{remote_prefix}...")
    result = subprocess.run(
        ['b2', 'ls', f'b2://{bucket_name}/{remote_prefix}'],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"Error listing files: {result.stderr}", file=sys.stderr)
        return []

    files = [line.strip() for line in result.stdout.split('\n') if line.strip()]
    return files


def download_directory(bucket_name, remote_prefix, local_dir):
    """
    Download directory from B2 bucket using sync.

    Args:
        bucket_name: B2 bucket name
        remote_prefix: Remote path prefix in bucket
        local_dir: Local directory to download to
    """
    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading: b2://{bucket_name}/{remote_prefix}")
    print(f"To: {local_dir}")
    print("-" * 60)

    # Use b2 sync for efficient download (only downloads new/changed files)
    cmd = [
        'b2', 'sync',
        '--replace-newer',  # Replace if remote is newer
        '--threads', '4',  # Parallel downloads
        f'b2://{bucket_name}/{remote_prefix}',
        str(local_path)
    ]

    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("\n✓ Download completed successfully")

        # Show downloaded files
        downloaded_files = list(local_path.rglob('*'))
        print(f"\nDownloaded {len(downloaded_files)} files to {local_dir}")

        # Show directory size
        total_size = sum(f.stat().st_size for f in downloaded_files if f.is_file())
        print(f"Total size: {total_size / 1e9:.2f} GB")
    else:
        print(f"\n✗ Download failed with exit code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)


def download_single_file(bucket_name, remote_file, local_file):
    """Download a single file from B2."""
    print(f"\nDownloading: {remote_file}")
    print(f"To: {local_file}")

    local_path = Path(local_file)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        'b2', 'download-file-by-name',
        bucket_name,
        remote_file,
        str(local_path)
    ]

    result = subprocess.run(cmd)

    if result.returncode == 0:
        size = local_path.stat().st_size
        print(f"✓ Downloaded {size / 1e6:.2f} MB")
    else:
        print(f"✗ Download failed", file=sys.stderr)
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description='Download activation trajectories from Backblaze B2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--remote-prefix',
        default='trajectories',
        help='Remote path prefix in B2 bucket (default: trajectories)'
    )
    parser.add_argument(
        '--local-dir',
        default='data/trajectories',
        help='Local directory to download to (default: data/trajectories)'
    )
    parser.add_argument(
        '--list-only',
        action='store_true',
        help='List available files without downloading'
    )
    parser.add_argument(
        '--file',
        help='Download a specific file instead of syncing directory'
    )

    args = parser.parse_args()

    # Load config and authorize
    config = load_b2_config()
    authorize_b2(config)

    # List files if requested
    if args.list_only:
        files = list_available_files(config['B2_BUCKET_NAME'], args.remote_prefix)
        print(f"\n{len(files)} files available:")
        for f in files:
            print(f"  {f}")
        return

    # Download single file or sync directory
    if args.file:
        download_single_file(
            config['B2_BUCKET_NAME'],
            args.file,
            Path(args.local_dir) / Path(args.file).name
        )
    else:
        download_directory(
            config['B2_BUCKET_NAME'],
            args.remote_prefix,
            args.local_dir
        )


if __name__ == '__main__':
    main()
