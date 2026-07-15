"""
Fine-tuning Module

Handles fine-tuning the Palmo model on instruction datasets.
Supports full fine-tuning (LoRA is currently disabled for simplicity).

Author: Fabrizio Soppelsa <fabrizio.soppelsa [@] community.unipa.it>
University of Palermo - Natural Language Processing Course
Academic Year 2025/2026
"""

import os
import torch
import torch.nn as nn
import math
from pathlib import Path
from tqdm import tqdm
import json
from torch.utils.data import Dataset, DataLoader


class LoRALayer(nn.Module):
    """
    Layer LoRA (Low-Rank Adaptation).
    
    Implements low-rank decomposition: ΔW = B @ A,
    where A is in R^(d×r) and B is in R^(r×k), with r << min(d,k).
    
    Follows Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models"
    (2021), and Raschka, "Build a Large Language Model", Chapter 6.
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        rank: Rank of the low-rank matrices (r)
        alpha: Scaling factor (typically 16 or 32)
    """
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Trainable low-rank matrices.
        # A projects into the rank-r space.
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        # Fill the tensor with values from the He/Kaiming uniform distribution.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        
    def forward(self, x):
        """
        Applies the LoRA adaptation.
        
        Args:
            x: Input tensor (..., in_features)
            
        Returns:
            Low-rank update: x @ A @ B * scaling
        """
        # x @ A: (..., in_features) @ (in_features, rank) -> (..., rank)
        # result @ B: (..., rank) @ (rank, out_features) -> (..., out_features)
        return (x @ self.lora_A @ self.lora_B) * self.scaling


class LinearWithLoRA(nn.Module):
    """
    Linear layer with optional LoRA adaptation.
    
    Wraps nn.Linear and adds low-rank adaptation:
        output = x @ W + bias + x @ (A @ B) * scaling
    
    The original W weights are frozen; only A and B are trainable.
    This drastically reduces the number of trainable parameters (~1% of full fine-tuning).
    
    Args:
        linear: Original nn.Linear layer (will be frozen)
        rank: Rank for LoRA decomposition
        alpha: LoRA scaling factor
    """
    def __init__(self, linear: nn.Linear, rank=8, alpha=16):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features,
            linear.out_features,
            rank=rank,
            alpha=alpha
        )
        
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
    
    def forward(self, x):
        """
        Forward pass with frozen linear layer and LoRA adaptation.
        
        Args:
            x: Input tensor
            
        Returns:
            Output combinato: frozen_linear(x) + lora_adaptation(x)
        """
        # Frozen original transformation.
        original_output = self.linear(x)
        
        lora_output = self.lora(x)
        
        return original_output + lora_output



def format_instruction_alpaca(example):
    """
    Formats an example in Alpaca style for training.
    
    Args:
        example: Dictionary with 'instruction' and 'output'
        
    Returns:
        Formatted string
    """
    instruction = example.get('instruction', '')
    output = example.get('output', '')
    
    # Format: ### Instruction:\n...\n### Response:\n...
    formatted = f"### Istruzione:\n{instruction}\n\n### Risposta:\n{output}"
    return formatted


class InstructionDataset(Dataset):
    """
    Instruction dataset with prompt masking.
    
    Args:
        data: List of dictionaries with 'instruction' and 'output'
        tokenizer: Tokenizer for encoding
        pad_token_id: Padding token ID (used to mask the prompt)
    """
    def __init__(self, data, tokenizer, pad_token_id=0):
        self.data = data
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        instruction = example.get('instruction', '')
        output = example.get('output', '')
        
        prompt_text = f"Question: {instruction}\nAnswer: "
        
        full_text = prompt_text + output
        
        prompt_tokens = self.tokenizer.encode(prompt_text)
        full_tokens = self.tokenizer.encode(full_text)
        
        prompt_len = len(prompt_tokens)
        
        input_ids = torch.tensor(full_tokens, dtype=torch.long)
        labels = input_ids.clone()
        
        labels[:prompt_len] = self.pad_token_id
        
        return {
            'input_ids': input_ids,
            'labels': labels
        }


def custom_collate_fn(batch, pad_token_id=0, device='cpu'):
    """
    Collate function for padding variable-length sequences.
    Handles both input_ids and labels with prompt masking.
    
    Args:
        batch: List of dictionaries with 'input_ids' and 'labels'
        pad_token_id: Padding token ID (default: 0)
        device: Device for the tensors
        
    Returns:
        Tuple of padded and aligned (unshifted) input_ids and labels
    """
    max_len = max(len(item['input_ids']) for item in batch)
    
    padded_input_ids = []
    padded_labels = []
    
    for item in batch:
        input_ids = item['input_ids']
        labels = item['labels']
        
        padding_len = max_len - len(input_ids)
        
        padded_input = torch.cat([
            input_ids,
            torch.full((padding_len,), pad_token_id, dtype=torch.long)
        ])
        padded_input_ids.append(padded_input)
        
        padded_label = torch.cat([
            labels,
            torch.full((padding_len,), pad_token_id, dtype=torch.long)
        ])
        padded_labels.append(padded_label)
    
    # Stack in batch tensors
    input_ids_batch = torch.stack(padded_input_ids).to(device)
    labels_batch = torch.stack(padded_labels).to(device)
    
    # No shift: inputs and labels are aligned.
    return input_ids_batch, labels_batch


def replace_linear_with_lora(model, rank=8, alpha=16):
    """
    Replaces the specified linear layers with LoRA versions.
    
    Implementation based on Raschka, "Build a Large Language Model", Chapter 6.
    This function iterates through all model modules and replaces selected
    nn.Linear layers with LinearWithLoRA.
    
    Specifically for Palmo, it applies LoRA to W_query and W_value in
    attention layers (W_key remains frozen, as is common practice).
    
    Args:
        model: PalmoModel model
        rank: Rank for LoRA matrices (default: 8)
              Lower = fewer parameters but less expressiveness
              Typically: 4–16 for small models, 32–64 for large models
        alpha: LoRA scaling factor (default: 16)
               Controls the influence of the adapters
               Typically: 16 or 32
        
    Returns:
        Model with linear layers replaced by LoRA layers
    """
    from src.transformer import MultiHeadAttention
    
    for param in model.parameters():
        param.requires_grad = False
    
    modified_layers = 0
    
    for name, module in model.named_modules():
        if isinstance(module, MultiHeadAttention):
            # Replace W_query and W_value with LoRA versions.
            # W_key remains frozen, as is common practice.
            if isinstance(module.W_query, nn.Linear):
                module.W_query = LinearWithLoRA(module.W_query, rank=rank, alpha=alpha)
            if isinstance(module.W_value, nn.Linear):
                module.W_value = LinearWithLoRA(module.W_value, rank=rank, alpha=alpha)
            modified_layers += 1
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"LoRA applicato a {modified_layers} layer di attenzione")
    print(f"Parametri totali: {total_params:,}")
    print(f"Parametri trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    print(f"Riduzione parametri: {100*(1-trainable_params/total_params):.1f}%")
    print(f"Configurazione: rank={rank}, alpha={alpha}\n")
    
    return model


# Alias retained for compatibility with existing code.
setup_lora = replace_linear_with_lora


def finetune(
    model,
    tokenizer,
    instruction_data,
    max_epochs=5,
    batch_size=4,
    lr=5e-5,
    device='cpu',
    use_lora=True,
    save_path=None,
    text_name="model",
    patience=1,
    min_delta=0.0,
    val_split=0.2
):
    """
    Fine-tunes instructions with batching, padding, and early stopping.
    
    Args:
        model: PalmoModel model to fine-tune
        tokenizer: Tokenizer for encoding/decoding
        instruction_data: List of dictionaries with 'instruction' and 'output'
        max_epochs: Maximum number of epochs (default: 5)
        batch_size: Batch size (default: 4)
        lr: Learning rate
        device: Device for training ('cpu', 'cuda', 'mps')
        use_lora: If True, uses LoRA for efficiency
        save_path: Optional path at which to save the model
        text_name: Corpus name for checkpoints
        patience: Epochs without improvement before early stopping (default: 1)
        min_delta: Minimum improvement considered significant (default: 0.0)
        val_split: Percentage of data for validation (default: 0.2 = 20%)
        
    Returns:
        Fine-tuned model and loss history
    """
    print(f"=== FINE-TUNING ===")
    print(f"Esempi totali: {len(instruction_data)}")
    print(f"Train/Val split: {int((1-val_split)*100)}/{int(val_split*100)}")
    print(f"Batch size: {batch_size}")
    print(f"Max epoche: {max_epochs}")
    print(f"Learning rate: {lr}")
    print(f"Device: {device}")
    print(f"Early stopping: patience={patience}, min_delta={min_delta}")
    print(f"LoRA: {'Abilitato' if use_lora else 'Disabilitato (full fine-tuning)'}\n")
    
    if use_lora:
        model = setup_lora(model, rank=8, alpha=16)
    
    # Train/validation split.
    import random
    random.seed(42)
    shuffled_data = instruction_data.copy()
    random.shuffle(shuffled_data)
    
    val_size = int(len(shuffled_data) * val_split)
    train_data = shuffled_data[val_size:]
    val_data = shuffled_data[:val_size]
    
    print(f"Train: {len(train_data)} esempi")
    print(f"Val: {len(val_data)} esempi\n")
    
    model = model.to(device)
    model.train()
    
    pad_token_id = getattr(tokenizer, 'pad_token_id', 0)
    
    train_dataset = InstructionDataset(train_data, tokenizer, pad_token_id)
    val_dataset = InstructionDataset(val_data, tokenizer, pad_token_id)
    
    print("\n=== ESEMPIO DATASET ===")
    ex = train_dataset[0]
    print("Input completo:")
    print(tokenizer.decode(ex['input_ids'].tolist()))
    print(f"\nToken totali: {len(ex['input_ids'])}")
    non_ignored = [id.item() for id in ex['labels'] if id != pad_token_id]
    print(f"Token non-ignorati (output): {len(non_ignored)}")
    if non_ignored:
        print("Output effettivo:")
        print(tokenizer.decode(non_ignored))
    print("=" * 40 + "\n")
    
    # DataLoader with a custom collate function.
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: custom_collate_fn(batch, pad_token_id, device)
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: custom_collate_fn(batch, pad_token_id, device)
    )
    
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr
    )
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training loop with early stopping.
    for epoch in range(max_epochs):
        # === TRAINING ===
        model.train()
        epoch_train_losses = []
        
        # Training progress bar.
        pbar = tqdm(train_dataloader, desc=f"Epoca {epoch+1}/{max_epochs} [Train]")
        
        for input_ids, target_ids in pbar:
            # Forward pass.
            logits = model(input_ids)
            
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
                ignore_index=pad_token_id
            )
            
            # Backward pass.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Metric tracking.
            epoch_train_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Mean training loss.
        avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        train_losses.append(avg_train_loss)
        
        # === VALIDATION ===
        model.eval()
        epoch_val_losses = []
        
        with torch.no_grad():
            pbar_val = tqdm(val_dataloader, desc=f"Epoca {epoch+1}/{max_epochs} [Val]")
            for input_ids, target_ids in pbar_val:
                # Forward pass.
                logits = model(input_ids)
                
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    target_ids.view(-1),
                    ignore_index=pad_token_id
                )
                
                epoch_val_losses.append(loss.item())
                pbar_val.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Mean validation loss.
        avg_val_loss = sum(epoch_val_losses) / len(epoch_val_losses)
        val_losses.append(avg_val_loss)
        
        print(f"\nEpoca {epoch+1}/{max_epochs}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")
        
        # === EARLY STOPPING ===
        if avg_val_loss + min_delta < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            print(f"  => Nuovo miglior val_loss: {best_val_loss:.4f}")
            
            if save_path:
                best_path = save_path.replace('.pt', '-best.pt')
                save_finetuned_model(model, tokenizer, best_path, {
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'epoch': epoch + 1
                }, use_lora)
                print(f"  => Miglior modello salvato: {best_path}")
        else:
            patience_counter += 1
            print(f"  => Nessun miglioramento. Patience: {patience_counter}/{patience}")
            
            if patience_counter > patience:
                print(f"\n=> Early stopping attivato dopo {epoch+1} epoche")
                print(f"=> Miglior val_loss: {best_val_loss:.4f}")
                break
    
    if save_path:
        save_finetuned_model(model, tokenizer, save_path, {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss
        }, use_lora)
        print(f"\n=> Modello finale salvato in {save_path}")
    
    return model, {'train_losses': train_losses, 'val_losses': val_losses, 'best_val_loss': best_val_loss}


def save_finetuned_model(model, tokenizer, path, history, use_lora):
    """
    Saves the fine-tuned model.
    
    Args:
        model: Model to save
        tokenizer: Tokenizer
        path: File path
        history: Dictionary with training history (losses, epoch, etc.)
        use_lora: If True, saves only the LoRA weights
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    if use_lora:
        lora_state_dict = {
            name: param for name, param in model.state_dict().items()
            if 'lora_A' in name or 'lora_B' in name
        }
        
        lora_path = path.replace('.pt', '-lora.pt')
        torch.save({
            'lora_state_dict': lora_state_dict,
            'history': history,
            'lora': True,
            'lora_config': {'rank': 8, 'alpha': 16}
        }, lora_path)
        
        size_mb = os.path.getsize(lora_path) / (1024**2)
        print(f"Adapter LoRA salvato: {lora_path} ({size_mb:.2f} MB)")
        print(f"Parametri LoRA: {len(lora_state_dict)} tensori")
    else:
        torch.save({
            'model_state_dict': model.state_dict(),
            'history': history,
            'lora': False
        }, path)


def load_instruction_data(path):
    """
    Loads instruction data from a JSON file.
    
    Args:
        path: Path to the JSON file
        
    Returns:
        List of dictionaries with 'instruction' and 'output'
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"=> Caricati {len(data)} esempi da {path}")
    return data
