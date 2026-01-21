#!/bin/bash
#
# Local monitoring script for SLURM LogiQA collection
# Runs on LOCAL machine, monitors remote jobs via SSH
#
# Usage: ./monitor_slurm_jobs.sh
#
# Features:
#   - SSH keepalive to prevent VPN drops
#   - Auto-kill quadro1 when olmo3_sft completes
#   - Verify HDF5 integrity before upload
#   - Auto-upload to B2 when models complete
#

# Exit on undefined variable (but not on command failure - we handle those)
set -u

# Configuration
SSH_HOST="ai_inst"
POLL_INTERVAL=300  # 5 minutes
SSH_KEEPALIVE=60   # 1 minute
REPO_DIR="/home/ttdo/maniver/ManiVer"
LOG_FILE="$HOME/slurm_monitor.log"

# Job IDs (update these when resubmitting)
JOB_QUADRO1=7864  # olmo3_sft (+ rl_zero, think - but we'll kill after sft)
JOB_QUADRO2=7865  # olmo3_rl_zero, olmo3_think

# Log file paths on remote
LOG_Q1_OUT="/home/ttdo/logiqa_recollect_out.txt"
LOG_Q1_ERR="/home/ttdo/logiqa_recollect_err.txt"
LOG_Q2_OUT="/home/ttdo/logiqa_q2_out.txt"
LOG_Q2_ERR="/home/ttdo/logiqa_q2_err.txt"

# State tracking
SFT_DONE=false
RLZERO_DONE=false
THINK_DONE=false
QUADRO1_KILLED=false

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# SSH with keepalive options - returns exit code
# Uses existing SSH config (ControlMaster, ProxyCommand, etc.)
ssh_cmd() {
    ssh -o ServerAliveInterval=$SSH_KEEPALIVE \
        -o ServerAliveCountMax=3 \
        -o ConnectTimeout=30 \
        "$SSH_HOST" "$@" 2>/dev/null
    return $?
}

# Check SSH connectivity
check_ssh() {
    if ssh_cmd "echo ok" | grep -q "ok"; then
        return 0
    else
        return 1
    fi
}

# Check if job is still running
job_running() {
    local job_id=$1
    if ssh_cmd "squeue -j $job_id 2>/dev/null" | grep -q "$job_id"; then
        return 0
    else
        return 1
    fi
}

# Check if model collection completed (by looking for completion message)
check_model_done() {
    local log_file=$1
    local model=$2
    if ssh_cmd "grep -q 'Collection complete for $model' '$log_file'" 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

# Check if model is starting (to detect transition)
check_model_starting() {
    local log_file=$1
    local model=$2
    if ssh_cmd "grep -q 'Collecting: $model' '$log_file'" 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

# Get current progress from stderr (batch X/125)
get_progress() {
    local log_file=$1
    local model=$2
    ssh_cmd "grep -oE '${model}/logiqa:[ ]+[0-9]+%.*' '$log_file' 2>/dev/null | tail -1" 2>/dev/null || echo "unknown"
}

# Verify HDF5 file integrity
verify_h5_file() {
    local model=$1
    local h5_path="$REPO_DIR/data/trajectories/$model/logiqa_trajectories_vllm_optimized.h5"

    local result
    result=$(ssh_cmd "source ~/miniconda3/etc/profile.d/conda.sh && conda activate maniver_env && python3 -c \"
import h5py
import sys
try:
    with h5py.File('$h5_path', 'r') as f:
        n = len(f['is_correct'])
        shape = f['trajectories'].shape
        correct = f['is_correct'][:].sum()
        if n != 500 or shape[0] != 500:
            print(f'CORRUPT: wrong size n={n} shape={shape}')
            sys.exit(1)
        # Read a sample to verify data integrity
        _ = f['trajectories'][0, 0, 0, 0]
        _ = f['trajectories'][-1, -1, -1, -1]
        print(f'OK: {n} samples, {correct} correct, shape {shape}')
        sys.exit(0)
except Exception as e:
    print(f'CORRUPT: {e}')
    sys.exit(1)
\"" 2>/dev/null)

    echo "$result"
    if echo "$result" | grep -q "^OK:"; then
        return 0
    else
        return 1
    fi
}

# Upload model to B2
upload_to_b2() {
    local model=$1
    local h5_file="$REPO_DIR/data/trajectories/$model/logiqa_trajectories_vllm_optimized.h5"
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local remote_path="logiqa_0shot_recollect_${timestamp}/trajectories/$model/logiqa_trajectories_vllm_optimized.h5"

    log "Uploading $model to B2..."
    log "  Local: $h5_file"
    log "  Remote: b2://ml-activations-store/$remote_path"

    local result
    result=$(ssh_cmd "cd $REPO_DIR && \
        source ~/miniconda3/etc/profile.d/conda.sh && \
        conda activate maniver_env && \
        source configs/b2-configs.txt && \
        b2 authorize-account \$B2_KEY_ID \$B2_APP_KEY && \
        b2 upload-file ml-activations-store '$h5_file' '$remote_path'" 2>&1)

    if [ $? -eq 0 ]; then
        log "  Upload successful!"
        log "$result" | tail -5
        return 0
    else
        log "  Upload FAILED!"
        log "$result"
        return 1
    fi
}

# Main monitoring loop
main() {
    log "========================================="
    log "SLURM Job Monitor Started"
    log "========================================="
    log "Monitoring:"
    log "  quadro1 (job $JOB_QUADRO1): olmo3_sft"
    log "  quadro2 (job $JOB_QUADRO2): olmo3_rl_zero, olmo3_think"
    log "Poll interval: ${POLL_INTERVAL}s"
    log "Log file: $LOG_FILE"
    log ""

    local retry_count=0
    local max_retries=5

    while true; do
        log "--- Checking status ---"

        # Check SSH connectivity first
        if ! check_ssh; then
            retry_count=$((retry_count + 1))
            log "SSH connection failed (attempt $retry_count/$max_retries)"
            if [ $retry_count -ge $max_retries ]; then
                log "ERROR: Too many SSH failures. Check VPN and try again."
                log "Waiting 60s before retry..."
                sleep 60
                retry_count=0
            fi
            sleep 30
            continue
        fi
        retry_count=0

        # Check quadro1 (olmo3_sft)
        if ! $QUADRO1_KILLED; then
            if job_running $JOB_QUADRO1; then
                # Check if sft is done and rl_zero is starting
                if check_model_starting "$LOG_Q1_OUT" "olmo3_rl_zero"; then
                    log "DETECTED: olmo3_rl_zero starting on quadro1 - KILLING JOB $JOB_QUADRO1"
                    ssh_cmd "scancel $JOB_QUADRO1" || true
                    QUADRO1_KILLED=true
                    sleep 5

                    # Verify and upload sft
                    log "Verifying olmo3_sft..."
                    if verify_h5_file "olmo3_sft"; then
                        SFT_DONE=true
                        upload_to_b2 "olmo3_sft"
                    else
                        log "ERROR: olmo3_sft file corrupted or missing!"
                    fi
                else
                    local progress
                    progress=$(get_progress "$LOG_Q1_ERR" "olmo3_sft")
                    log "quadro1: olmo3_sft running - $progress"
                fi
            else
                log "quadro1: Job $JOB_QUADRO1 no longer running"
                QUADRO1_KILLED=true

                # Check if sft completed before job ended
                if ! $SFT_DONE; then
                    log "Checking if olmo3_sft completed..."
                    if verify_h5_file "olmo3_sft"; then
                        log "olmo3_sft found and valid!"
                        SFT_DONE=true
                        upload_to_b2 "olmo3_sft"
                    fi
                fi
            fi
        fi

        # Check quadro2 (olmo3_rl_zero, olmo3_think)
        if job_running $JOB_QUADRO2; then
            # Check rl_zero completion
            if ! $RLZERO_DONE; then
                if check_model_starting "$LOG_Q2_OUT" "olmo3_think"; then
                    # If think is starting, rl_zero must be done
                    log "DETECTED: olmo3_think starting - olmo3_rl_zero must be complete"
                    if verify_h5_file "olmo3_rl_zero"; then
                        log "olmo3_rl_zero verified OK"
                        RLZERO_DONE=true
                        upload_to_b2 "olmo3_rl_zero"
                    else
                        log "ERROR: olmo3_rl_zero file corrupted!"
                    fi
                else
                    local progress
                    progress=$(get_progress "$LOG_Q2_ERR" "olmo3_rl_zero")
                    log "quadro2: olmo3_rl_zero running - $progress"
                fi
            fi

            # Check think progress
            if $RLZERO_DONE && ! $THINK_DONE; then
                local progress
                progress=$(get_progress "$LOG_Q2_ERR" "olmo3_think")
                log "quadro2: olmo3_think running - $progress"
            fi
        else
            log "quadro2: Job $JOB_QUADRO2 no longer running"

            # Check if models completed
            if ! $RLZERO_DONE; then
                log "Checking if olmo3_rl_zero completed..."
                if verify_h5_file "olmo3_rl_zero"; then
                    log "olmo3_rl_zero found and valid!"
                    RLZERO_DONE=true
                    upload_to_b2 "olmo3_rl_zero"
                fi
            fi

            if ! $THINK_DONE; then
                log "Checking if olmo3_think completed..."
                if verify_h5_file "olmo3_think"; then
                    log "olmo3_think found and valid!"
                    THINK_DONE=true
                    upload_to_b2 "olmo3_think"
                fi
            fi
        fi

        # Status summary
        log "Status: sft=$SFT_DONE, rl_zero=$RLZERO_DONE, think=$THINK_DONE"

        # Check if all done
        if $SFT_DONE && $RLZERO_DONE && $THINK_DONE; then
            log "========================================="
            log "ALL MODELS COMPLETE AND UPLOADED!"
            log "========================================="
            exit 0
        fi

        # Check if both jobs ended without completion
        if $QUADRO1_KILLED && ! job_running $JOB_QUADRO2; then
            if ! $SFT_DONE || ! $RLZERO_DONE || ! $THINK_DONE; then
                log "========================================="
                log "WARNING: Jobs ended but not all models complete!"
                log "Missing: sft=$SFT_DONE, rl_zero=$RLZERO_DONE, think=$THINK_DONE"
                log "========================================="
            fi
            exit 1
        fi

        log "Sleeping ${POLL_INTERVAL}s..."
        echo ""
        sleep $POLL_INTERVAL
    done
}

# Handle Ctrl+C gracefully
trap 'log "Interrupted. Exiting..."; exit 130' INT

main "$@"
