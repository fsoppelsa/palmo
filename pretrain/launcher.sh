#!/bin/bash
# Simple interactive launcher for Palmo pretraining

echo "╔════════════════════════════════════════╗"
echo "║   Palmo Pretraining Pipeline Launcher ║"
echo "╚════════════════════════════════════════╝"
echo ""

# Function to show menu
show_menu() {
    echo "Select an action:"
    echo ""
    echo "  1) Train on Bohemia (10 epochs) - detached"
    echo "  2) Train on Sherlock (20 epochs) - detached"
    echo "  3) Train on Malpelo (10 epochs) - detached"
    echo "  4) Train on Verdict (10 epochs) - detached"
    echo ""
    echo "  5) Monitor Bohemia training"
    echo "  6) Monitor Sherlock training"
    echo "  7) Monitor Malpelo training"
    echo "  8) Monitor Verdict training"
    echo ""
    echo "  9) Custom training (interactive)"
    echo ""
    echo "  0) Exit"
    echo ""
    echo -n "Choice: "
}

# Function to train
train_detached() {
    local corpus=$1
    local epochs=$2
    echo ""
    echo "Starting training on $corpus ($epochs epochs)..."
    ./run_pretraining_detached.sh "$corpus" "$epochs"
    echo ""
    echo "Training started! Monitor with:"
    echo "  ./monitor_pretraining.sh $corpus"
    echo ""
    echo "Press Enter to continue..."
    read
}

# Function to monitor
monitor_training() {
    local corpus=$1
    echo ""
    ./monitor_pretraining.sh "$corpus"
    echo ""
    echo "Press Enter to continue..."
    read
}

# Function for custom training
custom_training() {
    echo ""
    echo "Custom Training"
    echo "───────────────"
    echo -n "Corpus name (bohemia/sherlock/malpelo/verdict): "
    read corpus
    echo -n "Number of epochs (default: 10): "
    read epochs
    epochs=${epochs:-10}
    echo -n "Learning rate (default: 4e-4): "
    read lr
    lr=${lr:-4e-4}
    echo -n "Batch size (default: 32): "
    read batch
    batch=${batch:-32}
    echo -n "Vocab size (default: 8000): "
    read vocab
    vocab=${vocab:-8000}
    
    echo ""
    echo "Configuration:"
    echo "  Corpus: $corpus"
    echo "  Epochs: $epochs"
    echo "  Learning Rate: $lr"
    echo "  Batch Size: $batch"
    echo "  Vocab Size: $vocab"
    echo ""
    echo -n "Start training? (y/n): "
    read confirm
    
    if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
        echo ""
        echo "Starting custom training..."
        python3 run_pretraining.py \
            --corpus "$corpus" \
            --epochs "$epochs" \
            --lr "$lr" \
            --batch-size "$batch" \
            --vocab-size "$vocab"
    else
        echo "Training cancelled."
    fi
    
    echo ""
    echo "Press Enter to continue..."
    read
}

# Main loop
while true; do
    clear
    echo "╔════════════════════════════════════════╗"
    echo "║   Palmo Pretraining Pipeline Launcher ║"
    echo "╚════════════════════════════════════════╝"
    echo ""
    show_menu
    read choice
    
    case $choice in
        1) train_detached "bohemia" 10 ;;
        2) train_detached "sherlock" 20 ;;
        3) train_detached "malpelo" 10 ;;
        4) train_detached "verdict" 10 ;;
        5) monitor_training "bohemia" ;;
        6) monitor_training "sherlock" ;;
        7) monitor_training "malpelo" ;;
        8) monitor_training "verdict" ;;
        9) custom_training ;;
        0) echo "Goodbye!"; exit 0 ;;
        *) echo "Invalid choice. Press Enter to continue..."; read ;;
    esac
done
