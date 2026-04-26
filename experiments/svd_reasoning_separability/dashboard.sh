#!/bin/bash
# Dashboard for SVD Reasoning Separability Experiment
# Run this locally: ./dashboard.sh
# It will SSH to eyecog every 20s and show status

REMOTE="eyecog"
REMOTE_DIR="~/p1_geom_characterization/experiments/svd_reasoning_separability"

clear
echo "=============================================="
echo "  SVD REASONING SEPARABILITY - DASHBOARD"
echo "=============================================="
echo ""
echo "Monitoring $REMOTE:$REMOTE_DIR"
echo "Press Ctrl+C to exit"
echo ""

while true; do
    # Clear and redraw
    tput cup 6 0  # Move cursor to line 6

    echo "----------------------------------------------"
    echo "Last check: $(date '+%H:%M:%S')"
    echo "----------------------------------------------"
    echo ""

    # Get status.json
    STATUS=$(ssh -o ConnectTimeout=5 $REMOTE "cat $REMOTE_DIR/status.json 2>/dev/null" 2>/dev/null)

    if [ $? -ne 0 ]; then
        echo "  [ERROR] Cannot connect to $REMOTE"
        echo ""
    elif [ -z "$STATUS" ]; then
        echo "  [WAITING] No status file yet - experiment may not have started"
        echo ""
    else
        # Parse JSON with python (available on most systems)
        echo "$STATUS" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    state = d.get('state', 'unknown')
    progress = d.get('progress_pct', 0)
    last_update = d.get('last_update', 'N/A')

    # Progress bar
    bar_width = 40
    filled = int(bar_width * progress / 100)
    bar = '=' * filled + '-' * (bar_width - filled)

    print(f'  State:    {state.upper()}')
    print(f'  Progress: [{bar}] {progress}%')
    print(f'  Updated:  {last_update}')
    print()

    if state == 'running':
        task = d.get('task', '?')
        task_idx = d.get('task_idx', '?')
        total_tasks = d.get('total_tasks', '?')
        layer = d.get('layer', '?')
        total_layers = d.get('total_layers', '?')
        print(f'  Task:     {task_idx}/{total_tasks} - {task.upper()}')
        print(f'  Layer:    {layer}/{total_layers-1 if isinstance(total_layers, int) else \"?\"}')
    elif state == 'completed':
        total_time = d.get('total_time_seconds', 0)
        results_file = d.get('results_file', 'N/A')
        print(f'  Time:     {total_time:.1f}s ({total_time/60:.1f}m)')
        print(f'  Results:  {results_file}')
except Exception as e:
    print(f'  [ERROR] Parsing status: {e}')
"
    fi

    echo ""
    echo "----------------------------------------------"
    echo "Recent log output:"
    echo "----------------------------------------------"

    # Get last 10 lines of log
    ssh -o ConnectTimeout=5 $REMOTE "tail -10 $REMOTE_DIR/output.log 2>/dev/null" 2>/dev/null || echo "  (no log output yet)"

    echo ""
    echo "----------------------------------------------"
    echo "Next update in 20s... (Ctrl+C to exit)"

    sleep 20
done
