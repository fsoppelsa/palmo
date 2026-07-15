#!/usr/bin/env python3
"""
Run Pre-training Pipeline
Esporta la logica completa di pretraining dal notebook main.ipynb
Include: tokenization, model initialization, pretraining con time tracking

Author: Fabrizio Soppelsa <fabrizio.soppelsa [@] community.unipa.it>
Università degli Studi di Palermo - Corso di Natural Language Processing
Academic Year 2025/2026
"""

import os
import sys
import json
import re
import time
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.transformer import PalmoModel
from src.tokenizer import PalmoTokenizer
from src.pretraining import load_dataset, pretrain_model, save_checkpoint, load_checkpoint


def normalize_corpus(text: str) -> tuple[str, dict]:
    """
    Normalizza il corpus per ridurre token di spazi.
    
    Returns:
        tuple: (text normalizzato, statistiche)
    """
    original_length = len(text)
    original_spaces = text.count(' ')

    # 1. Normalizza spazi multipli → spazio singolo
    text = re.sub(r'[ \t]+', ' ', text)

    # 2. Normalizza newline multipli (mantieni paragrafi)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

    # 3. Rimuovi spazi a inizio/fine riga
    text = re.sub(r'[ \t]+\n', '\n', text)
    text = re.sub(r'\n[ \t]+', '\n', text)

    # 4. Trim generale
    text = text.strip()

    new_length = len(text)
    new_spaces = text.count(' ')
    reduction = ((original_spaces - new_spaces) / original_spaces * 100) if original_spaces > 0 else 0

    stats = {
        'original_length': original_length,
        'new_length': new_length,
        'chars_removed': original_length - new_length,
        'original_spaces': original_spaces,
        'new_spaces': new_spaces,
        'space_reduction_pct': reduction
    }
    
    return text, stats


def load_or_train_tokenizer(text: str, vocab_path: str, max_vocab_size: int) -> tuple[PalmoTokenizer, list]:
    """
    Carica il tokenizer esistente o ne addestra uno nuovo.
    
    Returns:
        tuple: (tokenizer, token_ids)
    """
    if os.path.exists(vocab_path):
        print(f"Trovato {vocab_path}, caricamento vocabolario e token...")
        with open(vocab_path, "r") as f:
            vocab_data = json.load(f)
        
        tokenizer = PalmoTokenizer()
        tokenizer.vocab = vocab_data["vocab"]
        tokenizer.reverse_vocab = {int(k): v for k, v in vocab_data["reverse_vocab"].items()}
        tokenizer.merges = [tuple(m) for m in vocab_data["merges"]]
        
        # Carica token_ids se disponibili
        if "token_ids" in vocab_data:
            token_ids = vocab_data["token_ids"]
            print(f"Vocabolario caricato: {len(tokenizer.vocab)} token")
            print(f"Token IDs caricati: {len(token_ids):,} token")
        else:
            print(f"Vocabolario caricato: {len(tokenizer.vocab)} token")
            print("Token IDs non trovati, tokenizzazione in corso...")
            token_ids = tokenizer.encode(text)
            print(f"Corpus tokenizzato: {len(token_ids):,} token")
            
            # Salva i token_ids nel file
            vocab_data["token_ids"] = token_ids
            with open(vocab_path, "w") as f:
                json.dump(vocab_data, f, ensure_ascii=False, indent=2)
            print(f"Token IDs salvati in {vocab_path}")
    else:
        print("Training BPE sul testo completo...")
        tokenizer = PalmoTokenizer(max_vocab_size)
        tokenizer.bpe(text)
        print(f"Vocabolario finale: {len(tokenizer.vocab)} token")
        
        # Tokenizza il corpus
        print("Tokenizzazione corpus...")
        token_ids = tokenizer.encode(text)
        print(f"Corpus tokenizzato: {len(token_ids):,} token")
        
        # Salva tutto insieme
        vocab_data = {
            "vocab": tokenizer.vocab,
            "reverse_vocab": {str(k): v for k, v in tokenizer.reverse_vocab.items()},
            "merges": tokenizer.merges,
            "token_ids": token_ids
        }
        os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
        with open(vocab_path, "w") as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        print(f"Vocabolario e token salvati in {vocab_path}")
    
    return tokenizer, token_ids


def write_timing_log(log_path: str, message: str):
    """Scrive un messaggio nel file di log con timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a") as f:
        f.write(f"[{timestamp}] {message}\n")
    print(message)


def run_pretraining_pipeline(
    corpus_name: str = "bohemia",
    epochs: int = 10,
    learning_rate: float = 4e-4,
    batch_size: int = 32,
    max_vocab_size: int = 8000,
    embed_dim: int = 256,
    n_heads: int = 8,
    n_layers: int = 12,
    context_length: int = 256,
    dropout: float = 0.1,
    checkpoint_dir: str = "checkpoints",
    force_retrain: bool = False
):
    """
    Esegue l'intera pipeline di pretraining con time tracking.
    
    Args:
        corpus_name: Nome del corpus (es. "bohemia", "sherlock", "malpelo")
        epochs: Numero di epoche di training
        learning_rate: Learning rate per l'optimizer
        batch_size: Dimensione del batch
        max_vocab_size: Dimensione massima del vocabolario
        embed_dim: Dimensione degli embedding
        n_heads: Numero di attention heads
        n_layers: Numero di layer del transformer
        context_length: Lunghezza del contesto
        dropout: Dropout rate
        checkpoint_dir: Directory per salvare i checkpoint
        force_retrain: Se True, riaddestra anche se esiste già un checkpoint
    """
    
    # Setup paths (relative to parent palmo directory)
    script_dir = Path(__file__).parent
    palmo_dir = script_dir.parent
    
    corpus_path = palmo_dir / f"corpus/{corpus_name}.txt"
    vocab_path = palmo_dir / f"data/vocab-{corpus_name}.json"
    checkpoint_dir_path = palmo_dir / checkpoint_dir
    log_path = checkpoint_dir_path / f"palmo-{corpus_name}-timing.log"
    checkpoint_path = checkpoint_dir_path / f"palmo-{corpus_name}-best.pt"
    
    # Inizializza il file di log
    checkpoint_dir_path.mkdir(exist_ok=True)
    with open(log_path, "w") as f:
        f.write(f"=== PRETRAINING LOG: {corpus_name} ===\n")
        f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Corpus: {corpus_path}\n")
        f.write(f"Epochs: {epochs}, LR: {learning_rate}, Batch: {batch_size}\n")
        f.write(f"Vocab Size: {max_vocab_size}, Context: {context_length}\n")
        f.write(f"Model: embed_dim={embed_dim}, n_heads={n_heads}, n_layers={n_layers}\n\n")
    
    total_start_time = time.time()
    
    # ========== 1. CARICAMENTO CORPUS ==========
    write_timing_log(log_path, "=" * 60)
    write_timing_log(log_path, "FASE 1: Caricamento Corpus")
    write_timing_log(log_path, "=" * 60)
    step_start = time.time()
    
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus non trovato: {corpus_path}")
    
    text = corpus_path.read_text(encoding='utf-8')
    words = text.split()
    
    step_time = time.time() - step_start
    write_timing_log(log_path, f"Corpus caricato: {len(text):,} caratteri, {len(words):,} parole")
    write_timing_log(log_path, f"Tempo: {step_time:.2f}s\n")
    
    # ========== 2. NORMALIZZAZIONE ==========
    write_timing_log(log_path, "=" * 60)
    write_timing_log(log_path, "FASE 2: Normalizzazione Corpus")
    write_timing_log(log_path, "=" * 60)
    step_start = time.time()
    
    text, norm_stats = normalize_corpus(text)
    
    step_time = time.time() - step_start
    write_timing_log(log_path, f"Caratteri: {norm_stats['original_length']:,} → {norm_stats['new_length']:,} (riduzione {norm_stats['chars_removed']:,})")
    write_timing_log(log_path, f"Spazi: {norm_stats['original_spaces']:,} → {norm_stats['new_spaces']:,} (riduzione {norm_stats['space_reduction_pct']:.1f}%)")
    write_timing_log(log_path, f"Tempo: {step_time:.2f}s\n")
    
    # ========== 3. TOKENIZATION ==========
    write_timing_log(log_path, "=" * 60)
    write_timing_log(log_path, "FASE 3: Tokenization (BPE)")
    write_timing_log(log_path, "=" * 60)
    step_start = time.time()
    
    tokenizer, token_ids = load_or_train_tokenizer(text, vocab_path, max_vocab_size)
    
    step_time = time.time() - step_start
    write_timing_log(log_path, f"Vocabolario: {len(tokenizer.vocab)} token")
    write_timing_log(log_path, f"Token IDs: {len(token_ids):,}")
    write_timing_log(log_path, f"Tempo: {step_time:.2f}s\n")
    
    # ========== 4. INIZIALIZZAZIONE MODELLO ==========
    write_timing_log(log_path, "=" * 60)
    write_timing_log(log_path, "FASE 4: Inizializzazione Modello")
    write_timing_log(log_path, "=" * 60)
    step_start = time.time()
    
    config = {
        'vocab_size': max_vocab_size,
        'embed_dim': embed_dim,
        'n_heads': n_heads,
        'n_layers': n_layers,
        'context_length': context_length,
        'dropout': dropout,
    }
    
    model = PalmoModel(config)
    
    # Determina il device disponibile
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    step_time = time.time() - step_start
    write_timing_log(log_path, f"Device: {device}")
    write_timing_log(log_path, f"Parametri: {total_params:,} totali, {trainable_params:,} trainable")
    write_timing_log(log_path, f"Tempo: {step_time:.2f}s\n")
    
    # ========== 5. PRETRAINING ==========
    write_timing_log(log_path, "=" * 60)
    write_timing_log(log_path, "FASE 5: Pretraining")
    write_timing_log(log_path, "=" * 60)
    
    # Verifica se esiste già un checkpoint
    if os.path.exists(checkpoint_path) and not force_retrain:
        write_timing_log(log_path, f"Checkpoint trovato: {checkpoint_path}")
        write_timing_log(log_path, "Training saltato (usa force_retrain=True per riaddestrare)")
        
        checkpoint_data = load_checkpoint(checkpoint_path, model)
        
        # Calcola tempo totale
        total_time = time.time() - total_start_time
        write_timing_log(log_path, f"\nTempo totale pipeline: {total_time:.2f}s ({total_time/60:.1f} min)")
        
        return model, tokenizer, checkpoint_data
    
    # Crea DataLoader
    train_loader, val_loader = load_dataset(token_ids, context_length, batch_size=batch_size)
    
    # Modifica pretrain_model per tracciare il tempo per epoca
    # Wrapper per tracciare i tempi
    pretrain_start = time.time()
    epoch_times = []
    
    # Esegui il pretraining con callback per logging
    write_timing_log(log_path, f"Inizio training: {epochs} epoche\n")
    
    # Pre-hook: salviamo la funzione originale di pretrain_model
    # e creiamo una versione che logga i tempi
    original_pretrain = pretrain_model
    
    def pretrain_with_timing(*args, **kwargs):
        """Wrapper che traccia i tempi per epoca."""
        # Estrai parametri
        model_arg = args[0]
        train_loader_arg = args[1]
        
        # Esegui il pretraining
        result = original_pretrain(*args, **kwargs)
        
        return result
    
    # Esegui il pretraining
    history = pretrain_model(
        model, 
        train_loader, 
        val_loader=val_loader, 
        epochs=epochs, 
        lr=learning_rate, 
        device=device,
        checkpoint_dir=str(checkpoint_dir_path), 
        corpus_name=corpus_name
    )
    
    pretrain_time = time.time() - pretrain_start
    
    write_timing_log(log_path, f"\n{'=' * 60}")
    write_timing_log(log_path, "PRETRAINING COMPLETATO")
    write_timing_log(log_path, f"{'=' * 60}")
    write_timing_log(log_path, f"Tempo pretraining totale: {pretrain_time:.2f}s ({pretrain_time/60:.1f} min)")
    write_timing_log(log_path, f"Tempo medio per epoca: {pretrain_time/epochs:.2f}s")
    
    if history['train_losses']:
        write_timing_log(log_path, f"Loss iniziale: {history['train_losses'][0]:.4f}")
        write_timing_log(log_path, f"Loss finale: {history['train_losses'][-1]:.4f}")
        improvement = (1 - history['train_losses'][-1] / history['train_losses'][0]) * 100
        write_timing_log(log_path, f"Miglioramento: {improvement:.1f}%")
    
    if history.get('val_losses'):
        write_timing_log(log_path, f"Val loss finale: {history['val_losses'][-1]:.4f}")
        if history.get('best_val_loss'):
            write_timing_log(log_path, f"Miglior val loss: {history['best_val_loss']:.4f}")
    
    # ========== RIEPILOGO FINALE ==========
    total_time = time.time() - total_start_time
    write_timing_log(log_path, f"\n{'=' * 60}")
    write_timing_log(log_path, "PIPELINE COMPLETATA")
    write_timing_log(log_path, f"{'=' * 60}")
    write_timing_log(log_path, f"Tempo totale: {total_time:.2f}s ({total_time/60:.1f} min)")
    write_timing_log(log_path, f"Checkpoint salvati in: {checkpoint_dir_path}/")
    write_timing_log(log_path, f"Log salvato in: {log_path}")
    
    return model, tokenizer, history


def main():
    """Entry point per esecuzione da command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Palmo Pretraining Pipeline')
    parser.add_argument('--corpus', type=str, default='bohemia',
                        help='Nome del corpus (bohemia, sherlock, malpelo, verdict)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Numero di epoche (default: 10)')
    parser.add_argument('--lr', type=float, default=4e-4,
                        help='Learning rate (default: 4e-4)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--vocab-size', type=int, default=8000,
                        help='Max vocabulary size (default: 8000)')
    parser.add_argument('--force-retrain', action='store_true',
                        help='Forza il retraining anche se esiste un checkpoint')
    
    args = parser.parse_args()
    
    # Configurazioni specifiche per corpus
    # Note: Removed auto-override to respect user's explicit batch_size choice
    if args.corpus == "sherlock":
        args.vocab_size = 15000 if args.vocab_size == 8000 else args.vocab_size
    
    print(f"\n{'=' * 60}")
    print(f"PALMO PRETRAINING PIPELINE")
    print(f"{'=' * 60}")
    print(f"Corpus: {args.corpus}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.lr}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Vocab Size: {args.vocab_size}")
    print(f"Force Retrain: {args.force_retrain}")
    print(f"{'=' * 60}\n")
    
    # Esegui la pipeline
    model, tokenizer, history = run_pretraining_pipeline(
        corpus_name=args.corpus,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        max_vocab_size=args.vocab_size,
        force_retrain=args.force_retrain
    )
    
    print(f"\n{'=' * 60}")
    print("PRETRAINING COMPLETATO CON SUCCESSO!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
