#!/bin/bash
#
# Setup B2 CLI on vast.ai instance
#
# This script:
# 1. Installs B2 CLI
# 2. Authorizes with credentials from config file
# 3. Verifies connection
#
# Usage:
#   bash scripts/setup_b2_on_vastai.sh

set -e  # Exit on error

echo "================================================"
echo "Setting up Backblaze B2 CLI on vast.ai instance"
echo "================================================"

# Install B2 CLI
echo ""
echo "[1/3] Installing B2 CLI..."
pip install -q b2

# Read credentials from config
CONFIG_FILE="configs/b2-configs.txt"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo ""
echo "[2/3] Reading credentials from $CONFIG_FILE..."
B2_KEY_ID=$(grep "^B2_KEY_ID=" "$CONFIG_FILE" | cut -d'=' -f2)
B2_APP_KEY=$(grep "^B2_APP_KEY=" "$CONFIG_FILE" | cut -d'=' -f2)
B2_BUCKET_NAME=$(grep "^B2_BUCKET_NAME=" "$CONFIG_FILE" | cut -d'=' -f2)

if [ -z "$B2_KEY_ID" ] || [ -z "$B2_APP_KEY" ]; then
    echo "Error: Could not read B2 credentials from config file"
    exit 1
fi

# Authorize
echo ""
echo "[3/3] Authorizing B2 CLI..."
b2 authorize-account "$B2_KEY_ID" "$B2_APP_KEY" > /dev/null

# Verify
echo ""
echo "Verifying connection..."
b2 list-buckets | grep -q "$B2_BUCKET_NAME" && echo "✓ Successfully connected to bucket: $B2_BUCKET_NAME"

echo ""
echo "================================================"
echo "B2 CLI setup complete!"
echo "================================================"
echo ""
echo "You can now:"
echo "  • Download trajectories: python scripts/b2_download.py"
echo "  • Upload results: python scripts/b2_upload.py"
echo "  • List files: b2 ls b2://$B2_BUCKET_NAME/"
echo ""
