#!/bin/bash
# Monitor script per controllare lo stato del pretraining in detached mode

CORPUS="${1:-bohemia}"
PID_FILE="checkpoints/palmo-${CORPUS}.pid"
LOG_FILE="checkpoints/palmo-${CORPUS}-output.log"
TIMING_LOG="checkpoints/palmo-${CORPUS}-timing.log"

echo "======================================"
echo "  Palmo Pretraining Monitor"
echo "======================================"
echo "Corpus: $CORPUS"
echo ""

# Check if PID file exists
if [ ! -f "$PID_FILE" ]; then
    echo "❌ PID file not found: $PID_FILE"
    echo "   Process may not be running in detached mode"
    exit 1
fi

PID=$(cat "$PID_FILE")
echo "Process ID: $PID"

# Check if process is running
if ps -p "$PID" > /dev/null 2>&1; then
    echo "Status: ✅ RUNNING"
    
    # Show CPU and memory usage
    echo ""
    echo "Resource Usage:"
    ps -p "$PID" -o pid,ppid,%cpu,%mem,etime,cmd --no-headers | \
        awk '{printf "  PID: %s\n  CPU: %s%%\n  Memory: %s%%\n  Elapsed: %s\n", $1, $3, $4, $5}'
else
    echo "Status: ❌ NOT RUNNING"
    echo ""
    echo "Process may have completed or crashed."
    echo "Check logs for details."
fi

echo ""
echo "======================================"
echo "  Log Files"
echo "======================================"

# Show output log info
if [ -f "$LOG_FILE" ]; then
    LOG_SIZE=$(du -h "$LOG_FILE" | cut -f1)
    LOG_LINES=$(wc -l < "$LOG_FILE")
    echo "Output Log: $LOG_FILE"
    echo "  Size: $LOG_SIZE"
    echo "  Lines: $LOG_LINES"
    echo ""
    echo "Last 10 lines:"
    echo "----------------------------------------"
    tail -n 10 "$LOG_FILE" | sed 's/^/  /'
else
    echo "Output log not found: $LOG_FILE"
fi

echo ""

# Show timing log info
if [ -f "$TIMING_LOG" ]; then
    TIMING_SIZE=$(du -h "$TIMING_LOG" | cut -f1)
    TIMING_LINES=$(wc -l < "$TIMING_LOG")
    echo "Timing Log: $TIMING_LOG"
    echo "  Size: $TIMING_SIZE"
    echo "  Lines: $TIMING_LINES"
    echo ""
    
    # Extract timing information
    if grep -q "PRETRAINING COMPLETATO" "$TIMING_LOG"; then
        echo "✅ Training completed!"
        echo ""
        echo "Summary:"
        echo "----------------------------------------"
        grep -E "(Tempo|Loss|Miglioramento)" "$TIMING_LOG" | tail -n 5 | sed 's/^/  /'
    else
        echo "Training in progress..."
        # Show last timing entry
        echo ""
        echo "Latest progress:"
        echo "----------------------------------------"
        tail -n 5 "$TIMING_LOG" | sed 's/^/  /'
    fi
else
    echo "Timing log not found: $TIMING_LOG"
fi

echo ""
echo "======================================"
echo "  Commands"
echo "======================================"
echo "Watch output log:"
echo "  tail -f $LOG_FILE"
echo ""
echo "Watch timing log:"
echo "  tail -f $TIMING_LOG"
echo ""
echo "Stop process:"
echo "  kill $PID"
echo ""
