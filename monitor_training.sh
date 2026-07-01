#!/bin/bash
# Training Monitor Script - Run on MortySmith
# Shows progress, loss, and estimated completion time

LOG_FILE="/home/nodeadmin/trading-llm/training_run_v3.log"
OUTPUT_DIR="/home/nodeadmin/trading-llm/output/trader-v2"

echo "========================================"
echo "  QLoRA Training Monitor"
echo "========================================"
echo ""

# Check if training is running
PID=$(ps aux | grep train_qlora_v3 | grep -v grep | awk '{print $2}' | head -1)
if [ -n "$PID" ]; then
    ELAPSED=$(ps -p $PID -o etime --no-headers | tr -d ' ')
    echo "✅ Training running (PID: $PID, elapsed: $ELAPSED)"
else
    echo "❌ Training NOT running"
fi

echo ""

# Show last few lines of log
if [ -f "$LOG_FILE" ]; then
    echo "Last 5 lines of log:"
    tail -5 "$LOG_FILE"
    echo ""
    
    # Try to extract progress
    PROGRESS=$(grep -oP '\d+/222' "$LOG_FILE" | tail -1)
    if [ -n "$PROGRESS" ]; then
        echo "Progress: $PROGRESS steps"
    fi
    
    # Try to extract loss
    LOSS=$(grep -oP 'loss.*?\d+\.\d+' "$LOG_FILE" | tail -1)
    if [ -n "$LOSS" ]; then
        echo "Latest: $LOSS"
    fi
fi

echo ""

# Show GPU usage
echo "GPU Status:"
rocm-smi --showmeminfo vram 2>/dev/null | grep 'VRAM Total Used' | while read line; do
    echo "  $line"
done

echo ""

# Show checkpoints
if [ -d "$OUTPUT_DIR" ]; then
    echo "Checkpoints:"
    ls -la "$OUTPUT_DIR" | grep -E "checkpoint|final" 2>/dev/null || echo "  None yet"
fi

echo ""
echo "========================================"