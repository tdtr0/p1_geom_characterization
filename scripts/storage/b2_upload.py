#!/usr/bin/env python3
"""
Upload activation trajectories to Backblaze B2 storage.

Usage:
    python scripts/b2_upload.py [--local-dir DATA_DIR] [--remote-prefix PREFIX]

Examples:
    # Upload all trajectories from default location
    python scripts/b2_upload.py

    # Upload from custom directory
    python scripts/b2_upload.py --local-dir /workspace/maniver/data/trajectories

    # Upload to specific B2 prefix
    python scripts/b2_upload.py --remote-prefix experiments/phase2/run1
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


def upload_directory(local_dir, bucket_name, remote_prefix):
    """
    Upload directory to B2 bucket using sync.

    Args:
        local_dir: Local directory to upload
        bucket_name: B2 bucket name
        remote_prefix: Remote path prefix in bucket
    """
    local_path = Path(local_dir)
    if not local_path.exists():
        print(f"Error: Local directory not found: {local_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"\nUploading: {local_dir}")
    print(f"To: b2://{bucket_name}/{remote_prefix}")
    print("-" * 60)

    # Use b2 sync for efficient upload (only uploads new/changed files)
    cmd = [
        'b2', 'sync',
        '--replace-newer',  # Replace if local is newer
        '--keep-days', '30',  # Keep deleted files for 30 days
        '--threads', '4',  # Parallel uploads
        str(local_path),
        f'b2://{bucket_name}/{remote_prefix}'
    ]

    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("\n✓ Upload completed successfully")
        print(f"\nFiles available at:")
        print(f"  Direct B2: https://f005.backblazeb2.com/file/{bucket_name}/{remote_prefix}")
        print(f"  Cloudflare CDN (once DNS configured): https://activations.maniact.space/{remote_prefix}")
    else:
        print(f"\n✗ Upload failed with exit code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description='Upload activation trajectories to Backblaze B2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--local-dir',
        default='data/trajectories',
        help='Local directory to upload (default: data/trajectories)'
    )
    parser.add_argument(
        '--remote-prefix',
        default='trajectories',
        help='Remote path prefix in B2 bucket (default: trajectories)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be uploaded without actually uploading'
    )

    args = parser.parse_args()

    # Load config and authorize
    config = load_b2_config()
    authorize_b2(config)

    # Upload
    if args.dry_run:
        print(f"\n[DRY RUN] Would upload:")
        print(f"  From: {args.local_dir}")
        print(f"  To: b2://{config['B2_BUCKET_NAME']}/{args.remote_prefix}")
        return

    upload_directory(
        args.local_dir,
        config['B2_BUCKET_NAME'],
        args.remote_prefix
    )


if __name__ == '__main__':
    main()
