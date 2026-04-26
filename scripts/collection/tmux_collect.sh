#!/bin/bash
# Launch activation collection in tmux session
# Usage: bash scripts/tmux_collect.sh

SESSION_NAME="olmo_collection"
GPU_ID=1

# Kill existing session if it exists
tmux kill-session -t $SESSION_NAME 2>/dev/null

# Create new tmux session
tmux new-session -d -s $SESSION_NAME

# Set up environment and run collection
tmux send-keys -t $SESSION_NAME "cd ~/p1_geom_characterization" C-m
tmux send-keys -t $SESSION_NAME "source ~/miniconda3/etc/profile.d/conda.sh" C-m
tmux send-keys -t $SESSION_NAME "conda activate geometric_transfer" C-m
tmux send-keys -t $SESSION_NAME "clear" C-m
tmux send-keys -t $SESSION_NAME "echo '========================================'" C-m
tmux send-keys -t $SESSION_NAME "echo 'OLMo 3 Activation Collection'" C-m
tmux send-keys -t $SESSION_NAME "echo 'Models: Base, SFT, RL-Zero'" C-m
tmux send-keys -t $SESSION_NAME "echo 'GPU: $GPU_ID'" C-m
tmux send-keys -t $SESSION_NAME "echo '========================================'" C-m
tmux send-keys -t $SESSION_NAME "echo ''" C-m
tmux send-keys -t $SESSION_NAME "nvidia-smi" C-m
tmux send-keys -t $SESSION_NAME "echo ''" C-m
tmux send-keys -t $SESSION_NAME "echo 'Starting collection with auto-restart...'" C-m
tmux send-keys -t $SESSION_NAME "bash scripts/run_with_restart.sh scripts/collect_activations.py --gpu $GPU_ID --tasks gsm8k humaneval --checkpoint-freq 50" C-m

echo "=========================================="
echo "Tmux session '$SESSION_NAME' created!"
echo "=========================================="
echo ""
echo "Commands:"
echo "  Attach:    tmux attach -t $SESSION_NAME"
echo "  Detach:    Ctrl+B, then D"
echo "  Kill:      tmux kill-session -t $SESSION_NAME"
echo ""
echo "Monitoring:"
echo "  tail -f ~/p1_geom_characterization/collection.log"
echo "  watch -n 5 nvidia-smi"
echo ""
