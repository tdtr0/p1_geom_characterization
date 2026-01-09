#!/bin/bash
# Sync local changes to eyecog server

echo "Syncing to eyecog..."
rsync -av --exclude='.git' --exclude='__pycache__' --exclude='*.h5' --exclude='data/activations/*.h5' --exclude='data/trajectories/*.h5' --exclude='wandb/' --exclude='.claude/' . eyecog:~/p1_geom_characterization/

echo "✓ Sync complete"
