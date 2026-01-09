#!/bin/bash
# Auto-restart wrapper for collection script
# Restarts on failure, uses checkpointing to resume

MAX_RETRIES=5
RETRY_COUNT=0
WAIT_TIME=10  # seconds to wait between retries

# Parse command line arguments
SCRIPT="$1"
shift
ARGS="$@"

if [ -z "$SCRIPT" ]; then
    echo "Usage: $0 <script.py> [args...]"
    exit 1
fi

echo "=========================================="
echo "AUTO-RESTART WRAPPER"
echo "=========================================="
echo "Script: $SCRIPT"
echo "Args: $ARGS"
echo "Max retries: $MAX_RETRIES"
echo ""

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    echo "----------------------------------------"
    echo "Attempt $(($RETRY_COUNT + 1))/$MAX_RETRIES"
    echo "----------------------------------------"

    # Run the script
    python "$SCRIPT" $ARGS --resume

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Script completed successfully"
        exit 0
    else
        RETRY_COUNT=$(($RETRY_COUNT + 1))
        echo ""
        echo "✗ Script failed with exit code $EXIT_CODE"

        if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
            echo "⟳ Restarting in $WAIT_TIME seconds..."
            sleep $WAIT_TIME

            # Clear GPU memory
            echo "Clearing GPU cache..."
            python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
        fi
    fi
done

echo ""
echo "✗ Failed after $MAX_RETRIES attempts"
exit 1
