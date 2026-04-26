#!/bin/bash
# Auto-complete script: monitors collection, uploads to B2, terminates instance
# Run this on the vast.ai instance with: nohup bash scripts/collection/auto_complete_and_upload.sh &

set -e

cd /workspace/maniver
export PYTHONPATH=/workspace/maniver/src:$PYTHONPATH

LOG_FILE="data/logs/auto_complete_$(date +%Y%m%d_%H%M%S).log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "=========================================="
log "AUTO-COMPLETE MONITOR STARTED"
log "=========================================="

# Expected: 12 files (4 models × 3 tasks), each should be >1GB when complete
EXPECTED_FILES=12
MIN_FILE_SIZE=$((500 * 1024 * 1024))  # 500MB minimum per file (conservative)

check_collection_done() {
    # Check if all collection processes are done
    local running=$(pgrep -f "collect_trajectories" | wc -l)
    if [ "$running" -gt 0 ]; then
        return 1
    fi

    # Check GPU memory - all should be near zero
    local gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{sum+=$1} END {print sum}')
    if [ "$gpu_mem" -gt 1000 ]; then  # More than 1GB total = still running
        return 1
    fi

    return 0
}

count_valid_files() {
    local count=0
    for model in olmo3_base olmo3_sft olmo3_rl_zero olmo3_think; do
        for task in gsm8k logiqa humaneval; do
            local file="data/trajectories/${model}/${task}_trajectories.h5"
            if [ -f "$file" ]; then
                local size=$(stat -c%s "$file" 2>/dev/null || echo 0)
                if [ "$size" -gt "$MIN_FILE_SIZE" ]; then
                    count=$((count + 1))
                fi
            fi
        done
    done
    echo $count
}

show_status() {
    log "--- STATUS ---"
    log "Running processes: $(pgrep -f 'collect_trajectories' | wc -l)"
    log "GPU memory usage:"
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | while read line; do
        log "  GPU $line"
    done
    log "Valid files: $(count_valid_files) / $EXPECTED_FILES"
    log "Total size: $(du -sh data/trajectories 2>/dev/null | cut -f1)"
    log "--------------"
}

# Main monitoring loop
log "Monitoring every 5 minutes..."
log "Will upload to B2 and terminate when collection completes"

while true; do
    show_status

    if check_collection_done; then
        valid_files=$(count_valid_files)
        log "Collection processes finished!"
        log "Valid files: $valid_files / $EXPECTED_FILES"

        if [ "$valid_files" -ge 10 ]; then  # At least 10 of 12 files
            log "=========================================="
            log "COLLECTION COMPLETE - Starting B2 upload"
            log "=========================================="

            # Final file listing
            log "Final files:"
            find data/trajectories -name "*.h5" -exec ls -lh {} \; | tee -a "$LOG_FILE"

            # Upload to B2
            log "Uploading to Backblaze B2..."
            if python3 scripts/storage/b2_upload.py 2>&1 | tee -a "$LOG_FILE"; then
                log "=========================================="
                log "UPLOAD COMPLETE!"
                log "=========================================="

                # Verify upload
                log "Verifying upload..."
                python3 scripts/storage/b2_download.py --list-only 2>&1 | tee -a "$LOG_FILE"

                log "=========================================="
                log "ALL DONE! Safe to terminate instance."
                log "=========================================="

                # Create completion marker
                echo "COMPLETE: $(date)" > data/COLLECTION_COMPLETE.txt

                # Optional: Auto-terminate (commented out for safety)
                # log "Terminating instance in 60 seconds..."
                # sleep 60
                # shutdown -h now

                exit 0
            else
                log "ERROR: B2 upload failed!"
                exit 1
            fi
        else
            log "WARNING: Only $valid_files files found, expected $EXPECTED_FILES"
            log "Some models may have failed. Check logs."
            log "Waiting 5 more minutes before retry..."
        fi
    fi

    sleep 300  # Check every 5 minutes
done
