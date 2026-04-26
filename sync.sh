#!/bin/bash
# ManiVer Sync Script
# Syncs code between local, eyecog, and GitHub

set -e

EYECOG_DIR="~/p1_geom_characterization"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

usage() {
    echo "Usage: ./sync.sh [command]"
    echo ""
    echo "Commands:"
    echo "  eyecog      Sync local -> eyecog (default)"
    echo "  pull        Pull from eyecog -> local"
    echo "  push        Push to GitHub (from eyecog)"
    echo "  status      Show git status on eyecog"
    echo "  all         Sync to eyecog, then push to GitHub"
    echo ""
}

sync_to_eyecog() {
    echo -e "${GREEN}Syncing to eyecog...${NC}"
    rsync -av --progress \
        --exclude='.git' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='*.h5' \
        --exclude='.venv' \
        --exclude='venv' \
        --exclude='node_modules' \
        --exclude='data/activations/*.h5' \
        --exclude='data/trajectories/*.h5' \
        --exclude='wandb/' \
        --exclude='.claude/' \
        --exclude='offload/' \
        --exclude='logs/*.log' \
        --exclude='*.egg-info' \
        --exclude='.DS_Store' \
        . eyecog:${EYECOG_DIR}/
    echo -e "${GREEN}✓ Sync to eyecog complete${NC}"
}

pull_from_eyecog() {
    echo -e "${GREEN}Pulling from eyecog...${NC}"
    rsync -av --progress \
        --exclude='.git' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='*.h5' \
        --exclude='data/' \
        --exclude='wandb/' \
        --exclude='.claude/' \
        --exclude='offload/' \
        eyecog:${EYECOG_DIR}/ .
    echo -e "${GREEN}✓ Pull from eyecog complete${NC}"
}

push_to_github() {
    echo -e "${GREEN}Pushing to GitHub from eyecog...${NC}"
    ssh eyecog "cd ${EYECOG_DIR} && git add -A && git status"
    echo ""
    read -p "Commit message: " msg
    if [ -z "$msg" ]; then
        msg="Update from sync script"
    fi
    ssh eyecog "cd ${EYECOG_DIR} && git commit -m '${msg}' && git push"
    echo -e "${GREEN}✓ Pushed to GitHub${NC}"
}

show_status() {
    echo -e "${YELLOW}Git status on eyecog:${NC}"
    ssh eyecog "cd ${EYECOG_DIR} && git status"
}

# Main
case "${1:-eyecog}" in
    eyecog)
        sync_to_eyecog
        ;;
    pull)
        pull_from_eyecog
        ;;
    push)
        push_to_github
        ;;
    status)
        show_status
        ;;
    all)
        sync_to_eyecog
        push_to_github
        ;;
    -h|--help)
        usage
        ;;
    *)
        echo "Unknown command: $1"
        usage
        exit 1
        ;;
esac
