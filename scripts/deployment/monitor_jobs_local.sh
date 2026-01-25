#!/bin/bash
# Monitor SLURM jobs from local machine
# Usage: bash scripts/deployment/monitor_jobs_local.sh [refresh_interval]

INTERVAL=${1:-60}  # Default 60 seconds

# Function to safely run SSH command with retry
ssh_cmd() {
    local host=$1
    shift
    local result
    result=$(ssh -o ConnectTimeout=10 -o BatchMode=yes "$host" "$@" 2>/dev/null)
    if [ $? -eq 0 ]; then
        echo "$result"
    else
        echo "  [SSH failed - VPN down?]"
    fi
}

while true; do
    clear
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║           SLURM Job Monitor - $(date '+%Y-%m-%d %H:%M:%S')           ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""

    # Get job status
    echo "┌─ Active Jobs ─────────────────────────────────────────────────┐"
    ssh_cmd ai_inst "squeue -u ttdo"
    echo "└────────────────────────────────────────────────────────────────┘"

    echo ""
    echo "┌─ Think Collection Progress ────────────────────────────────────┐"
    THINK=$(ssh_cmd ai_inst "grep 'olmo3_think' /home/ttdo/logiqa_think_2gpu_err.txt | tail -1")
    if [ -n "$THINK" ]; then
        PCT=$(echo "$THINK" | grep -oE '[0-9]+%' | head -1)
        BATCHES=$(echo "$THINK" | grep -oE '[0-9]+/[0-9]+' | head -1)
        CORRECT=$(echo "$THINK" | grep -oE 'correct=[0-9]+' | head -1 | sed 's/correct=//')
        INCORRECT=$(echo "$THINK" | grep -oE 'incorrect=[0-9]+' | head -1 | sed 's/incorrect=//')
        OUTLEN=$(echo "$THINK" | grep -oE 'out_len=[0-9]+' | head -1 | sed 's/out_len=//')
        echo "  Progress:  $PCT ($BATCHES batches)"
        echo "  Correct:   $CORRECT"
        echo "  Incorrect: $INCORRECT"
        echo "  Out len:   $OUTLEN tokens"
    else
        echo "  No progress data or job completed"
    fi
    echo "└────────────────────────────────────────────────────────────────┘"

    echo ""
    echo "┌─ B2 Upload Status ─────────────────────────────────────────────┐"
    UPLOAD_STATUS=$(ssh_cmd ai_inst "grep -E '(Uploading|Done|fileName)' /home/ttdo/upload_b2_out.txt | tail -5")
    if echo "$UPLOAD_STATUS" | grep -q "Done"; then
        echo "  ✓ Upload COMPLETE"
        # Show uploaded files
        echo "  Files uploaded:"
        echo "$UPLOAD_STATUS" | grep "fileName" | sed 's/.*"fileName": "/    - /' | sed 's/".*//'
    elif [ -n "$UPLOAD_STATUS" ]; then
        echo "  In progress..."
        echo "$UPLOAD_STATUS" | head -3
    else
        echo "  No upload job or not started"
    fi
    echo "└────────────────────────────────────────────────────────────────┘"

    echo ""
    echo "┌─ B2 Cloud Files (LogiQA) ──────────────────────────────────────┐"
    ssh_cmd eyecog "source ~/miniconda3/etc/profile.d/conda.sh && conda activate base && b2 ls -l b2://ml-activations-store/trajectories/ 2>/dev/null | grep logiqa | awk '{print \"  \" \$5/1e9 \"GB  \" \$6 \" \" \$7 \"  \" \$9}'"
    echo "└────────────────────────────────────────────────────────────────┘"

    echo ""
    echo "┌─ SLURM File Sizes ─────────────────────────────────────────────┐"
    ssh_cmd ai_inst "ls -lh /home/ttdo/maniver/ManiVer/data/trajectories/*/logiqa*.h5 | awk '{print \"  \" \$5 \"  \" \$9}'"
    echo "└────────────────────────────────────────────────────────────────┘"

    echo ""
    echo "Refreshing in ${INTERVAL}s... (Ctrl+C to exit)"
    sleep $INTERVAL
done
