#!/bin/bash
# Run trajectory collection with correctness labels in tmux

SESSION_NAME="labeled_collection"

# Kill existing session if any
tmux kill-session -t $SESSION_NAME 2>/dev/null || true

# Create new session
tmux new-session -d -s $SESSION_NAME

# Send commands
tmux send-keys -t $SESSION_NAME 'cd ~/p1_geom_characterization' Enter
tmux send-keys -t $SESSION_NAME 'source ~/miniconda3/etc/profile.d/conda.sh' Enter
tmux send-keys -t $SESSION_NAME 'conda activate geometric_transfer' Enter
tmux send-keys -t $SESSION_NAME 'echo Starting labeled trajectory collection...' Enter
tmux send-keys -t $SESSION_NAME 'python scripts/collect_trajectories_with_labels.py 2>&1 | tee logs/labeled_collection.log' Enter

echo "Started tmux session: $SESSION_NAME"
echo "Attach with: tmux attach -t $SESSION_NAME"
echo "View logs: tail -f ~/p1_geom_characterization/logs/labeled_collection.log"
