# Palmo Pretraining - Standalone Script

This directory contains standalone scripts to run the Palmo pretraining pipeline outside of Jupyter notebooks.

## Files

- **`run_pretraining.py`**: Main pretraining pipeline script
- **`run_pretraining_detached.sh`**: Helper script to run pretraining in detached/background mode

## Quick Start

### Interactive Mode

Run pretraining directly (blocks terminal):

```bash
cd palmo
python3 run_pretraining.py --corpus bohemia --epochs 10
```

### Detached Mode (Recommended)

Run pretraining in background (terminal can be closed):

```bash
cd palmo
./run_pretraining_detached.sh bohemia 10
```

This will:
- Start the training process in background
- Save output to `checkpoints/palmo-bohemia-output.log`
- Save the process ID to `checkpoints/palmo-bohemia.pid`

## Command Line Options

```bash
python3 run_pretraining.py [OPTIONS]

Options:
  --corpus TEXT          Corpus name: bohemia or sherlock (default: bohemia)
  --epochs INT           Number of training epochs (default: 10)
  --lr FLOAT            Learning rate (default: 4e-4)
  --batch-size INT      Batch size (default: 32)
  --vocab-size INT      Maximum vocabulary size (default: 8000)
  --force-retrain       Force retraining even if checkpoint exists
```

## Examples

### Train on Bohemia corpus (default settings)
```bash
python3 run_pretraining.py
```

### Train on Sherlock corpus with 20 epochs
```bash
python3 run_pretraining.py --corpus sherlock --epochs 20
```

### Force retraining even if checkpoint exists
```bash
python3 run_pretraining.py --corpus bohemia --force-retrain
```

### Custom learning rate and batch size
```bash
python3 run_pretraining.py --corpus sherlock --lr 5e-4 --batch-size 64
```

## Detached Mode Usage

### Start training in background
```bash
./run_pretraining_detached.sh bohemia 10
```

### Monitor progress in real-time
```bash
tail -f checkpoints/palmo-bohemia-output.log
```

### Check detailed timing logs
```bash
tail -f checkpoints/palmo-bohemia-timing.log
```

### Check if process is still running
```bash
ps -p $(cat checkpoints/palmo-bohemia.pid)
```

### Stop the background process
```bash
kill $(cat checkpoints/palmo-bohemia.pid)
```

## Time Tracking

The pipeline tracks and logs timing information for each phase:

1. **Corpus Loading**: Time to read the text file
2. **Normalization**: Time to clean and normalize the corpus
3. **Tokenization**: Time for BPE training and encoding
4. **Model Initialization**: Time to create and move model to device
5. **Pretraining**: Detailed timing for each epoch

### Timing Logs

Two types of logs are generated:

- **`checkpoints/palmo-{corpus}-timing.log`**: Detailed timing information for each phase
- **`checkpoints/palmo-{corpus}-output.log`**: Full training output (only in detached mode)

Example timing log output:
```
============================================================
PHASE 1: Corpus Loading
============================================================
Corpus loaded: 49,214 characters, 8,956 words
Time: 0.03s

============================================================
PHASE 2: Corpus Normalization
============================================================
Characters: 49,214 → 48,891 (reduction: 323)
Whitespace: 8,675 → 8,352 (reduction: 3.7%)
Time: 0.01s

============================================================
PHASE 5: Pre-training
============================================================
Training started: 10 epochs

PRE-TRAINING COMPLETE
============================================================
Total pre-training time: 245.32s (4.1 min)
Average time per epoch: 24.53s
Initial loss: 6.8234
Final loss: 2.3456
Improvement: 65.6%
```

## Output Files

After successful training, the following files are created:

- `checkpoints/palmo-{corpus}-best.pt`: Best model checkpoint (lowest validation loss)
- `checkpoints/palmo-{corpus}-epoch{N}.pt`: Checkpoint for each epoch
- `checkpoints/palmo-{corpus}-pretrained.pt`: Final trained model
- `checkpoints/palmo-{corpus}-timing.log`: Detailed timing information
- `checkpoints/palmo-{corpus}-output.log`: Full console output (detached mode only)
- `checkpoints/palmo-{corpus}.pid`: Process ID (detached mode only)

## Pipeline Phases

The script executes the following phases in order:

1. **Corpus Loading**: Loads the text file from `corpus/{corpus}.txt`
2. **Normalization**: Cleans whitespace and normalizes the corpus
3. **Tokenization**: Trains or loads BPE tokenizer, encodes text
4. **Model Initialization**: Creates transformer model and moves to GPU/CPU
5. **Pretraining**: Trains the model with validation split and early stopping

## Requirements

Make sure the following files exist in the palmo directory:
- `src/transformer.py`: PalmoModel implementation
- `src/tokenizer.py`: PalmoTokenizer (BPE) implementation
- `src/pretraining.py`: Training utilities and functions
- `corpus/{corpus}.txt`: Corpus text file

## Notes

- The script automatically detects and uses GPU (CUDA/MPS) if available
- Checkpoints are saved after each epoch
- Early stopping is enabled (patience=3 epochs)
- Validation split is 10% of the data
- Previous tokenizer and vocabulary are reused if available
- Time tracking is automatic and logged to `*-timing.log`
