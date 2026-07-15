"""
Pre-training Module

Handles pre-training the model on a text corpus.
Based on Raschka's "Build a LLM from Scratch" - Chapter 5.1

Author: Fabrizio Soppelsa <fabrizio.soppelsa [@] community.unipa.it>
University of Palermo - Natural Language Processing Course
Academic Year 2025/2026
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List
import time
import math
from tqdm import tqdm


# Automatic CPU/GPU detection.
def get_device():
    """Automatically detects whether to use GPU (CUDA/MPS) or CPU."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')  # Apple Silicon.
    else:
        return torch.device('cpu')


def text2token(text: str, tokenizer):
    """Converts text into a list of token IDs."""
    return tokenizer.encode(text)


def token2text(token_ids: List[int], tokenizer):
    """Converts a list of token IDs into text."""
    return tokenizer.decode(token_ids)


def calc_loss_batch(input_batch, target_batch, model, device):
    """Calculates cross-entropy loss for a single batch."""
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    
    logits = model(input_batch)
    
    # logits: (batch_size, seq_len, vocab_size) -> (batch_size * seq_len, vocab_size)
    # targets: (batch_size, seq_len) -> (batch_size * seq_len)
    loss = nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        target_batch.view(-1)
    )
    
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None, verbose=False):
    """Calculates the mean loss over an entire DataLoader."""
    total_loss = 0.0
    
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    if verbose:
        print(f"Valutazione su {num_batches} batch...")
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()
    
    return total_loss / num_batches


# Dataset and DataLoader Creation
class TextDataset(Dataset):
    """Dataset for text sequences."""
    
    def __init__(self, token_ids: List[int], context_length: int):
        self.token_ids = token_ids
        self.context_length = context_length
        print(f"Dataset creato: {len(token_ids):,} token, context_length={context_length}, {len(self):,} sequenze")
        
    def __len__(self):
        return len(self.token_ids) - self.context_length
    
    def __getitem__(self, idx):
        # x = tokens[idx : idx + context_length]
        # y = tokens[idx + 1 : idx + context_length + 1]
        x = torch.tensor(self.token_ids[idx : idx + self.context_length], dtype=torch.long)
        y = torch.tensor(self.token_ids[idx + 1 : idx + self.context_length + 1], dtype=torch.long)
        return x, y


def create_dataloader(token_ids, context_length: int, batch_size: int = 32, 
                      shuffle: bool = True, drop_last: bool = True, num_workers: int = 0):
    """Creates a DataLoader from preprocessed tokens for training."""
    print(f"\nCreazione DataLoader: {len(token_ids):,} token disponibili")
    
    dataset = TextDataset(token_ids, context_length)
    
    num_batches = len(dataset) // batch_size
    if not drop_last and len(dataset) % batch_size != 0:
        num_batches += 1
    
    print(f"Parametri DataLoader: batch_size={batch_size}, shuffle={shuffle}, drop_last={drop_last}, num_workers={num_workers}")
    print(f"{num_batches:,} batch per epoca\n")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    
    return dataloader


def load_dataset(token_ids, context_length: int, batch_size: int = 32, val_split: float = 0.1):
    """Creates training and validation DataLoaders with an automatic split.
    
    Args:
        token_ids: List of token IDs
        context_length: Context length
        batch_size: Batch size
        val_split: Percentage of data for validation (default 10%)
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    val_size = int(len(token_ids) * val_split)
    train_size = len(token_ids) - val_size
    
    print(f"\nSplit dataset: {train_size:,} token per training, {val_size:,} token per validazione ({val_split*100:.0f}%)")
    
    train_tokens = token_ids[:train_size]
    val_tokens = token_ids[train_size:]
    
    train_loader = create_dataloader(train_tokens, context_length, batch_size, shuffle=True, drop_last=True)
    val_loader = create_dataloader(val_tokens, context_length, batch_size, shuffle=False, drop_last=False)
    
    return train_loader, val_loader


# Training and Evaluation Functions
def perplexity(model, dataloader, device='cpu'):
    """Calculates perplexity (exp of the mean loss)."""
    model.eval()
    
    with torch.no_grad():
        avg_loss = calc_loss_loader(dataloader, model, device)
    
    # Perplexity = e^(mean loss).
    perplexity = math.exp(avg_loss)
    
    return perplexity


def train_palmo32(model, train_loader, val_loader, optimizer, device, num_epochs,
                  eval_freq=None, eval_iter=5, start_context="", tokenizer=None):
    """FP32 (float32) training loop from Raschka Chapter 5.1 with optional periodic evaluation."""
    model = model.float()
    
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    
    for epoch in range(num_epochs):
        model.train()
        
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            
            tokens_seen += input_batch.numel()
            global_step += 1
            
            # Periodic evaluation (optional).
            #
            # if eval_freq and global_step % eval_freq == 0:
            #     train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
            #     val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
            #     
            #     train_losses.append(train_loss)
            #     val_losses.append(val_loss)
            #     track_tokens_seen.append(tokens_seen)
            #     
            #     print(f"Ep {epoch+1} (Step {global_step:06d}): "
            #           f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
        
        if tokenizer and start_context:
            model.eval()
            with torch.no_grad():
                token_ids = text2token(start_context, tokenizer)
                token_tensor = torch.tensor([token_ids], dtype=torch.long).to(device)
                generated = model.generate(token_tensor, max_new_tokens=50, temperature=0.7)
                generated_text = token2text(generated[0].tolist(), tokenizer)
                print(f"Esempio generato: {generated_text}\n")
    
    return train_losses, val_losses, track_tokens_seen


def train_palmo16(model, train_loader, val_loader, optimizer, device, num_epochs,
                  eval_freq=None, eval_iter=5, start_context="", tokenizer=None):
    """Training loop with mixed precision (AMP) for improved performance and numerical stability."""
    scaler = torch.amp.GradScaler('cuda')
    
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    
    for epoch in range(num_epochs):
        model.train()
        
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            
            # Forward pass with autocast (mixed precision).
            with torch.amp.autocast('cuda'):
                logits = model(input_batch)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    target_batch.view(-1)
                )
            
            # Backward pass with gradient scaling.
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            tokens_seen += input_batch.numel()
            global_step += 1
        
        if tokenizer and start_context:
            model.eval()
            with torch.no_grad():
                token_ids = text2token(start_context, tokenizer)
                token_tensor = torch.tensor([token_ids], dtype=torch.long).to(device)
                generated = model.generate(token_tensor, max_new_tokens=50, temperature=0.7)
                generated_text = token2text(generated[0].tolist(), tokenizer)
                print(f"Esempio generato: {generated_text}\n")
    
    return train_losses, val_losses, track_tokens_seen


def train_palmo(model, train_loader, val_loader, optimizer, device, num_epochs,
                eval_freq=None, eval_iter=5, start_context="", tokenizer=None, use_fp16=False):
    """Main training loop with FP16/FP32 support.
    
    Args:
        use_fp16: If True, uses FP16; otherwise, uses FP32
    """
    if use_fp16:
        return train_palmo16(model, train_loader, val_loader, optimizer, device, num_epochs,
                           eval_freq, eval_iter, start_context, tokenizer)
    else:
        return train_palmo32(model, train_loader, val_loader, optimizer, device, num_epochs,
                           eval_freq, eval_iter, start_context, tokenizer)


def pretrain_model(model, train_loader, val_loader=None, epochs: int = 10, 
                   lr: float = 3e-4, device=None, eval_freq: int = 100,
                   eval_iter: int = 5, start_context: str = "", patience: int = 3,
                   checkpoint_dir: str = None, corpus_name: str = "model"):
    if device is None:
        device = get_device()
        print(f"Dispositivo rilevato automaticamente: {device}")
    
    # Determine whether to use FP16 or FP32.
    use_fp16 = (device.type == 'cuda')
    precision = "FP16" if use_fp16 else "FP32"
    
    print("INIZIO PRE-TRAINING")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Config: device={device}, precision={precision}, epochs={epochs}, lr={lr}, batches={len(train_loader):,}")
    print(f"Modello: {total_params:,} parametri totali, {trainable_params:,} trainable")
    if val_loader:
        print(f"Batch validazione: {len(val_loader):,}")
    if checkpoint_dir:
        print(f"Checkpoint: {checkpoint_dir}")
    
    print(f"\nSpostamento modello su {device} ({precision})...")
    model = model.to(device)
    
    if use_fp16:
        print("Utilizzo Automatic Mixed Precision (AMP)")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    print("Modello pronto\n")
    
    scaler = torch.amp.GradScaler('cuda') if use_fp16 else None
    
    train_losses = []
    val_losses = []
    
    # Early-stopping variables.
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Timer for total training time.
    total_start_time = time.time()
    
    print("TRAINING")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        start_time = time.time()
        
        print(f"\nEpoca {epoch+1}/{epochs}")
        
        pbar = tqdm(train_loader, desc=f"Epoca {epoch+1}/{epochs}", unit="batch")
        for batch_idx, (input_batch, target_batch) in enumerate(pbar):
            optimizer.zero_grad()
            
            # Forward pass with AMP when enabled.
            if use_fp16:
                with torch.amp.autocast('cuda'):
                    loss = calc_loss_batch(input_batch, target_batch, model, device)
                # Backward pass with gradient scaling.
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard FP32 forward/backward pass.
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            avg_loss_so_far = epoch_loss / num_batches
            pbar.set_postfix({'loss': f'{avg_loss_so_far:.4f}'})
        
        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)
        
        if val_loader is not None:
            print(f"  Calcolo validation loss...")
            model.eval()
            with torch.no_grad():
                val_loss = calc_loss_loader(val_loader, model, device, verbose=False)
                val_losses.append(val_loss)
                perplexity = math.exp(val_loss)
            
            # Early-stopping logic.
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                print(f"  Nuovo miglior val_loss: {val_loss:.4f}")
                
                if checkpoint_dir:
                    import os
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    best_path = os.path.join(checkpoint_dir, f"palmo-{corpus_name}-best.pt")
                    save_checkpoint(
                        model, optimizer, epoch, val_loss, best_path,
                        train_losses=train_losses.copy(),
                        val_losses=val_losses.copy()
                    )
                    print(f"  Miglior modello salvato: {best_path}")
            else:
                patience_counter += 1
                print(f"  Nessun miglioramento. Patience: {patience_counter}/{patience}")
                
                if patience_counter >= patience:
                    total_elapsed = time.time() - total_start_time
                    print(f"\nEarly stopping attivato dopo {epoch+1} epoche (miglior val_loss: {best_val_loss:.4f})")
                    print("TRAINING INTERROTTO (Early Stopping)")
                    print(f"Loss finale: {train_losses[-1]:.4f}, Miglior val_loss: {best_val_loss:.4f}")
                    print(f"Tempo totale di training: {total_elapsed:.2f}s ({total_elapsed/60:.1f} min)")
                    return {
                        'train_losses': train_losses,
                        'val_losses': val_losses,
                        'stopped_early': True,
                        'best_val_loss': best_val_loss
                    }
        else:
            perplexity = math.exp(avg_train_loss)
        
        elapsed = time.time() - start_time
        
        print(f"\nEpoca {epoch+1}/{epochs} completata: train_loss={avg_train_loss:.4f}", end="")
        if val_loader:
            print(f", val_loss={val_loss:.4f}", end="")
        print(f", perplexity={perplexity:.2f}, time={elapsed:.2f}s, tokens/sec={int(num_batches * train_loader.batch_size * train_loader.dataset.context_length / elapsed)}")
        
        if checkpoint_dir:
            import os
            os.makedirs(checkpoint_dir, exist_ok=True)
            epoch_path = os.path.join(checkpoint_dir, f"palmo-{corpus_name}-epoch{epoch+1}.pt")
            current_loss = val_loss if val_loader else avg_train_loss
            save_checkpoint(model, optimizer, epoch, current_loss, epoch_path)
            print(f"  Checkpoint salvato: {epoch_path}")
    
    total_elapsed = time.time() - total_start_time
    print("\nPRE-TRAINING COMPLETATO")
    print(f"Loss finale: {train_losses[-1]:.4f}, Miglioramento: {train_losses[0]:.4f} -> {train_losses[-1]:.4f}")
    print(f"Tempo totale di training: {total_elapsed:.2f}s ({total_elapsed/60:.1f} min)")
    
    if checkpoint_dir:
        import os
        os.makedirs(checkpoint_dir, exist_ok=True)
        pretrained_path = os.path.join(checkpoint_dir, f"palmo-{corpus_name}-pretrained.pt")
        final_loss = val_losses[-1] if val_losses else train_losses[-1]
        save_checkpoint(
            model, optimizer, epochs-1, final_loss, pretrained_path,
            train_losses=train_losses,
            val_losses=val_losses if val_loader else None
        )
        print(f"Modello finale salvato: {pretrained_path}")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses if val_loader else None,
        'stopped_early': False,
        'best_val_loss': best_val_loss if val_loader else None
    }


def save_checkpoint(model, optimizer, epoch, loss, path: str, **kwargs):
    """Saves a model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    checkpoint.update(kwargs)
    
    torch.save(checkpoint, path)
    print(f"Checkpoint salvato in {path}")


def load_checkpoint(path: str, model, optimizer=None):
    """Loads a model checkpoint."""
    checkpoint = torch.load(path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint caricato da {path}")
    print(f"Epoca: {checkpoint.get('epoch', 'N/A')}, Loss: {checkpoint.get('loss', 'N/A')}")
    
    return checkpoint
