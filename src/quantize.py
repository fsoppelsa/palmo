"""
Quantization Module

Reduces the size of the Palmo model through dynamic INT8 quantization.
Enables efficient deployment on resource-constrained devices.

Author: Fabrizio Soppelsa <fabrizio.soppelsa [@] community.unipa.it>
University of Palermo - Natural Language Processing Course
Academic Year 2025/2026
"""

import os
import torch
import torch.quantization as quantization
import platform
from pathlib import Path
from src.transformer import PalmoModel
from src.pretraining import load_checkpoint


def quantize_model(
    text_name: str,
    config: dict,
    tokenizer,
    history: dict,
    test_prompt: str = "The detective said ",
    max_new_tokens: int = 30,
    temperature: float = 0.7
):
    """
    Quantizes a Palmo model using dynamic INT8 quantization.
    
    Args:
        text_name: Name of the text/corpus (e.g., "bohemia", "sherlock")
        config: Model configuration dictionary
        tokenizer: Tokenizer for encoding/decoding
        history: Dictionary with train_losses and val_losses
        test_prompt: Prompt for testing the quantized model
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature
        
    Returns:
        dict: Dictionary with information about the quantized model
    """
    print("QUANTIZZAZIONE MODELLO")
    
    quantized_path = f"checkpoints/palmo-{text_name}-quantized.pt"
    
    if os.path.exists(quantized_path):
        print("=> Modello già quantizzato, skip quantizzazione")
        return {"skipped": True, "path": quantized_path}
    
    model_to_quantize = PalmoModel(config)
    
    best_checkpoint = f"checkpoints/palmo-{text_name}-best.pt"
    pretrained_checkpoint = f"checkpoints/palmo-{text_name}-pretrained.pt"
    
    if os.path.exists(best_checkpoint):
        checkpoint = load_checkpoint(best_checkpoint, model_to_quantize)
        print(f"Caricato miglior modello da {best_checkpoint}")
    elif os.path.exists(pretrained_checkpoint):
        checkpoint = load_checkpoint(pretrained_checkpoint, model_to_quantize)
        print(f"Caricato modello da {pretrained_checkpoint}")
    else:
        raise FileNotFoundError(
            f"Nessun checkpoint trovato. Serve almeno uno tra:\n"
            f"  - {best_checkpoint}\n"
            f"  - {pretrained_checkpoint}"
        )
    
    model_to_quantize = model_to_quantize.cpu().eval()
    
    machine = platform.machine().lower()
    if any(arch in machine for arch in ['x86', 'amd64', 'i386', 'i686']):
        backend = 'fbgemm'  # Intel/AMD x86.
    elif any(arch in machine for arch in ['arm', 'aarch64']):
        backend = 'qnnpack'  # ARM (Orin, Apple Silicon, and so on).
    else:
        backend = 'fbgemm'  # Default fallback.
    
    print(f"CPU Architecture: {machine}")
    print(f"Uso Quantization backend: {backend}")
    
    try:
        torch.backends.quantized.engine = backend
    except RuntimeError as e:
        print(f"Backend {backend} non disponibile, provo fallback...")
        # Try the other backend.
        fallback = 'qnnpack' if backend == 'fbgemm' else 'fbgemm'
        try:
            torch.backends.quantized.engine = fallback
            print(f"[OK] Uso backend alternativo: {fallback}")
        except RuntimeError:
            print(f"[ERROR] Nessun backend di quantizzazione disponibile")
            print("  Salto quantizzazione\n")
            return {"skipped": True, "error": "No quantization backend available", "path": None}
    
    original_size = sum(p.numel() * p.element_size() for p in model_to_quantize.parameters()) / (1024**2)
    print(f"Dimensione modello originale: {original_size:.2f} MB")
    
    try:
        model_quantized = torch.quantization.quantize_dynamic(
            model_to_quantize,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
    except RuntimeError as e:
        if "NoQEngine" in str(e):
            print(f"\n ERRORE: Quantizzazione non supportata su questo sistema")
            print("   Quantizzazione richiede CPU x86 con supporto fbgemm o qnnpack")
            print("   Salto quantizzazione e uso modello originale\n")
            return {"skipped": True, "error": str(e), "path": None}
        else:
            raise
    
    quantized_size = sum(p.numel() * p.element_size() for p in model_quantized.parameters()) / (1024**2)
    savings = original_size - quantized_size
    savings_pct = (1 - quantized_size/original_size) * 100
    
    print(f"Dimensione modello quantizzato: {quantized_size:.2f} MB")
    print(f"Risparmio: {savings:.2f} MB ({savings_pct:.1f}%)")
    
    print("\nTEST GENERAZIONE")
    encoded = tokenizer.encode(test_prompt)
    encoded_tensor = torch.tensor([encoded], dtype=torch.long)
    
    generated = model_quantized.generate(encoded_tensor, max_new_tokens=max_new_tokens, temperature=temperature)
    decoded = tokenizer.decode(generated[0].tolist())
    
    print(f"Prompt: '{test_prompt}'")
    print(f"Output: {decoded}")
    
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        'model': model_quantized,
        'config': config,
        'epoch': checkpoint.get('epoch', 10),
        'loss': checkpoint.get('loss', 0.0),
        'train_losses': history.get('train_losses', []),
        'val_losses': history.get('val_losses', None),
        'quantized': True
    }, quantized_path)
    
    print(f"\n=> Modello quantizzato salvato in {quantized_path}")
    
    return {
        "skipped": False,
        "path": quantized_path,
        "original_size_mb": original_size,
        "quantized_size_mb": quantized_size,
        "savings_mb": savings,
        "savings_pct": savings_pct,
        "test_output": decoded
    }
