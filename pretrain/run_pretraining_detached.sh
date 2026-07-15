#!/bin/bash
# Script per eseguire il pretraining in modalità detached (background)
# Il processo continuerà anche dopo la chiusura del terminale

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configurazione
CORPUS="${1:-bohemia}"    # Default: bohemia
EPOCHS="${2:-10}"         # Default: 10
BATCH_SIZE="${3:-}"       # Default: use run_pretraining.py defaults
LOG_FILE="checkpoints/palmo-${CORPUS}-output.log"

echo "Starting Palmo Pretraining in detached mode..."
echo "Corpus: $CORPUS"
echo "Epochs: $EPOCHS"
if [ -n "$BATCH_SIZE" ]; then
    echo "Batch Size: $BATCH_SIZE"
fi
echo "Output log: $LOG_FILE"
echo ""

# Crea la directory checkpoints se non esiste
mkdir -p checkpoints

# Esegui in background con nohup (no hang-up)
# stdout e stderr vengono reindirizzati al log file
CMD="python3 $SCRIPT_DIR/run_pretraining.py --corpus $CORPUS --epochs $EPOCHS"
if [ -n "$BATCH_SIZE" ]; then
    CMD="$CMD --batch-size $BATCH_SIZE"
fi

nohup $CMD > "$LOG_FILE" 2>&1 &

PID=$!
echo "Process started with PID: $PID"
echo "PID saved to: checkpoints/palmo-${CORPUS}.pid"
echo "$PID" > "checkpoints/palmo-${CORPUS}.pid"

echo ""
echo "Monitor progress with:"
echo "  tail -f $LOG_FILE"
echo ""
echo "Check if process is running:"
echo "  ps -p $PID"
echo ""
echo "Stop the process:"
echo "  kill $PID"
echo ""
