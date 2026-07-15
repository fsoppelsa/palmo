#!/usr/bin/env python3
"""
Flask Web App for Palmo Model Inference

Interactive web interface for testing Palmo models with different checkpoints.

Author: Fabrizio Soppelsa <fabrizio.soppelsa [@] community.unipa.it>
University of Palermo - Natural Language Processing Course
Academic Year 2025/2026
"""

import os
import json
import torch
import torch.serialization
import torch.backends.quantized
from flask import Flask, render_template, request, jsonify
from pathlib import Path
from src.transformer import PalmoModel
from src.tokenizer import PalmoTokenizer
from src.pretraining import load_checkpoint

try:
    torch.serialization.add_safe_globals([PalmoModel])
except AttributeError:
    pass

try:
    torch.backends.quantized.engine = 'qnnpack'
except RuntimeError:
    try:
        torch.backends.quantized.engine = 'fbgemm'
    except RuntimeError:
        print("Attenzione: Nessun backend di quantizzazione disponibile, i modelli quantizzati potrebbero non funzionare")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
app = Flask(
    __name__,
    template_folder=str(PROJECT_ROOT / "templates"),
    static_folder=str(PROJECT_ROOT / "static"),
)

# Detect the available device (GPU/MPS/CPU).
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print("Usando GPU CUDA")
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
    print("Usando Apple Silicon MPS")
else:
    DEVICE = torch.device('cpu')
    print("Usando CPU (lento!)")

# Cache models to avoid repeated loading.
model_cache = {}
tokenizer_cache = {}

def get_vocab_path(model_name):
    """Returns the vocabulary path based on the model name."""
    if 'bohemia' in model_name.lower():
        return 'data/vocab-bohemia.json'
    else:  # Sherlock corpus.
        return 'data/vocab-sherlock.json'


def get_model_config(vocab_size):
    """Returns the model configuration."""
    return {
        'vocab_size': vocab_size,
        'embed_dim': 256,
        'n_heads': 8,
        'n_layers': 12,
        'context_length': 256,
        'dropout': 0.1,
    }


def load_tokenizer(vocab_path):
    """Loads the tokenizer from the vocabulary file."""
    if vocab_path in tokenizer_cache:
        return tokenizer_cache[vocab_path]
    
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
    
    with open(vocab_path, "r") as f:
        vocab_data = json.load(f)
    
    tokenizer = PalmoTokenizer()
    tokenizer.vocab = vocab_data["vocab"]
    tokenizer.reverse_vocab = {int(k): v for k, v in vocab_data["reverse_vocab"].items()}
    tokenizer.merges = [tuple(m) for m in vocab_data["merges"]]
    
    tokenizer_cache[vocab_path] = tokenizer
    return tokenizer


def load_model_checkpoint(checkpoint_path, vocab_size):
    """Loads the model from a checkpoint (supports regular, quantized, and LoRA checkpoints)."""
    if checkpoint_path in model_cache:
        return model_cache[checkpoint_path]
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Use CPU for quantized models and GPU/MPS for all others.
    is_quantized_path = 'quantized' in checkpoint_path.lower()
    load_device = 'cpu' if is_quantized_path else DEVICE
    
    checkpoint = torch.load(checkpoint_path, map_location=load_device, weights_only=False)
    
    if 'quantized' in checkpoint and checkpoint['quantized']:
        model = checkpoint['model']
        print(f"✓ Caricato modello quantizzato da {checkpoint_path} [CPU]")
    else:
        config = get_model_config(vocab_size)
        model = PalmoModel(config)
        
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        has_lora_keys = any('lora' in key for key in state_dict.keys())
        
        if has_lora_keys:
            from src.finetune import replace_linear_with_lora
            print(f"Rilevato checkpoint LoRA, applico struttura LoRA...")
            model = replace_linear_with_lora(model, rank=8, alpha=16)
            model.load_state_dict(state_dict, strict=False)
            print(f"✓ Caricato modello LoRA da {checkpoint_path}")
        else:
            model.load_state_dict(state_dict)
            print(f"✓ Caricato modello regolare da {checkpoint_path}")
        
        model = model.to(load_device)
        print(f"   Modello su: {load_device}")
    
    model.eval()
    model_cache[checkpoint_path] = model
    return model


def generate_response(prompt, model, tokenizer, max_tokens=20, temperature=0.7):
    """Generates text completion for a prompt with safety measures."""
    encoded = tokenizer.encode(prompt)
    
    model_device = next(model.parameters()).device
    encoded_tensor = torch.tensor([encoded], dtype=torch.long, device=model_device)
    
    # Cap generation length to prevent runaway loops.
    max_tokens = min(max_tokens, 200)
    
    import time
    start_time = time.time()
    timeout_seconds = 30  # 30-second timeout.
    
    generated_tokens = encoded_tensor.clone()
    
    with torch.no_grad():
        for step in range(max_tokens):
            # Check the timeout.
            if time.time() - start_time > timeout_seconds:
                print(f"Timeout generazione dopo {step} token")
                break
            
            idx_cond = generated_tokens[:, -model.config['context_length']:]
            logits = model.forward(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print("NaN/Inf rilevato nei logits, arresto generazione")
                break
            
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated_tokens = torch.cat((generated_tokens, next_token), dim=1)
            
            if hasattr(tokenizer, 'vocab') and tokenizer.vocab.get('<eos>', None) is not None:
                if next_token.item() == tokenizer.vocab['<eos>']:
                    break
            
            if generated_tokens.size(1) >= 6:
                last_three = generated_tokens[0, -3:].tolist()
                prev_three = generated_tokens[0, -6:-3].tolist()
                if last_three == prev_three:
                    print("Ripetizione rilevata, arresto generazione")
                    break
    
    return tokenizer.decode(generated_tokens[0].tolist())


@app.route('/')
def index():
    """Renders the main page."""
    return render_template('index.html')


@app.route('/inference', methods=['POST'])
def inference():
    """Handles an inference request."""
    try:
        data = request.json
        model_selection = data.get('model', 'Sherlock')
        prompt = data.get('prompt', '')
        temperature = float(data.get('temperature', 0.7))
        max_tokens = int(data.get('max_tokens', 30))
        
        if not prompt.strip():
            return jsonify({'error': 'Please provide a prompt'}), 400
        
        checkpoint_map = {
            'Bohemia': ['checkpoints/palmo-bohemia-quantized.pt', 'checkpoints/palmo-bohemia-best.pt'],
            'Bohemia-tuned': ['checkpoints/palmo-bohemia-tuned-quantized.pt', 'checkpoints/palmo-bohemia-tuned-best.pt'],
            'Sherlock': ['checkpoints/palmo-sherlock-quantized.pt', 'checkpoints/palmo-sherlock-best.pt'],
            'Sherlock-tuned': ['checkpoints/palmo-sherlock-tuned-quantized.pt', 'checkpoints/palmo-sherlock-tuned-best.pt'],
        }
        
        checkpoint_paths = checkpoint_map.get(model_selection)
        if not checkpoint_paths:
            return jsonify({'error': f'Invalid model selection: {model_selection}'}), 400
        
        checkpoint_path = None
        for path in checkpoint_paths:
            if os.path.exists(path):
                checkpoint_path = path
                break
        
        if not checkpoint_path:
            return jsonify({'error': f'No checkpoint found for {model_selection}. Tried: {checkpoint_paths}'}), 404
        
        # Load tokenizer
        vocab_path = get_vocab_path(model_selection)
        tokenizer = load_tokenizer(vocab_path)
        
        # Load model
        vocab_size = len(tokenizer.vocab)
        model = load_model_checkpoint(checkpoint_path, vocab_size)
        
        # Generate response
        response = generate_response(prompt, model, tokenizer, max_tokens=max_tokens, temperature=temperature)
        
        return jsonify({
            'response': response,
            'model': model_selection,
            'checkpoint': checkpoint_path
        })
    
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        return jsonify({'error': f'Inference error: {str(e)}'}), 500


@app.route('/generate', methods=['POST'])
def generate():
    """
    Generates text from a prompt (Garak-compatible API).
    
    Expected request format:
    {
        "prompt": "text to complete",
        "max_tokens": 20 (optional),
        "temperature": 0.7 (optional),
        "model": "<quantized model>" (optional)
    }
    
    Response format:
    {
        "text": "generated text"
    }
    """
    import time
    start_time = time.time()
    
    try:
        data = request.json
        prompt = data.get('prompt', '')
        temperature = float(data.get('temperature', 0.7))
        max_tokens = int(data.get('max_tokens', 20))
        model_selection = data.get('model', 'Sherlock-tuned')  # Default to the fine-tuned model.
        
        print(f"[/generate] Request: model={model_selection}, max_tokens={max_tokens}, temp={temperature}, prompt_len={len(prompt)}")
        
        if not prompt.strip():
            return jsonify({'error': 'Please provide a prompt'}), 400
        
        checkpoint_map = {
            'Bohemia': ['checkpoints/palmo-bohemia-quantized.pt', 'checkpoints/palmo-bohemia-best.pt'],
            'Bohemia-tuned': ['checkpoints/palmo-bohemia-tuned-quantized.pt', 'checkpoints/palmo-bohemia-tuned-best.pt'],
            'Sherlock': ['checkpoints/palmo-sherlock-quantized.pt', 'checkpoints/palmo-sherlock-best.pt'],
            'Sherlock-tuned': ['checkpoints/palmo-sherlock-tuned-quantized.pt', 'checkpoints/palmo-sherlock-tuned-best.pt'],
        }
        
        checkpoint_paths = checkpoint_map.get(model_selection)
        if not checkpoint_paths:
            return jsonify({'error': f'Invalid model selection: {model_selection}'}), 400
        
        checkpoint_path = None
        for path in checkpoint_paths:
            if os.path.exists(path):
                checkpoint_path = path
                break
        
        if not checkpoint_path:
            return jsonify({'error': f'No checkpoint found for {model_selection}. Tried: {checkpoint_paths}'}), 404
        
        vocab_path = get_vocab_path(model_selection)
        tokenizer = load_tokenizer(vocab_path)
        
        vocab_size = len(tokenizer.vocab)
        model = load_model_checkpoint(checkpoint_path, vocab_size)
        
        generated_text = generate_response(prompt, model, tokenizer, max_tokens=max_tokens, temperature=temperature)
        
        elapsed = time.time() - start_time
        print(f"[/generate] Success: {elapsed:.2f}s, output_len={len(generated_text)}")
        
        # Return the Garak-compatible response format.
        return jsonify({
            'text': generated_text
        })
    
    except FileNotFoundError as e:
        print(f"[/generate] FileNotFound: {e}")
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[/generate] Error after {elapsed:.2f}s: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Generation error: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8001)
