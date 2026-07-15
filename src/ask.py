#!/usr/bin/env python3
"""
Interactive Chatbot Module

Loads the fine-tuned Palmo model and answers questions using different temperatures.
Provides a CLI for interacting with the model.

Author: Fabrizio Soppelsa <fabrizio.soppelsa [@] community.unipa.it>
University of Palermo - Natural Language Processing Course
Academic Year 2025/2026
"""

import os
import json
import torch
from pathlib import Path
from src.transformer import PalmoModel
from src.tokenizer import PalmoTokenizer
from src.pretraining import load_checkpoint


def ask(question, model, tokenizer, max_tokens=50, temperature=0.5):
    """Generate text completion for a given prompt."""
    encoded = tokenizer.encode(question)
    encoded_tensor = torch.tensor([encoded], dtype=torch.long)
    
    with torch.no_grad():
        generated = model.generate(encoded_tensor, max_new_tokens=max_tokens, temperature=temperature)
    
    return tokenizer.decode(generated[0].tolist())


def main():
    # Configuration
    TEXT = "sherlock"
    VOCAB_PATH = f"data/vocab-{TEXT}.json"
    MAX_VOCAB_SIZE = 15000
    
    # Load tokenizer
    print(f"Loading tokenizer from {VOCAB_PATH}...")
    if not os.path.exists(VOCAB_PATH):
        print(f"Error: {VOCAB_PATH} not found!")
        return
    
    with open(VOCAB_PATH, "r") as f:
        vocab_data = json.load(f)
    
    tokenizer = PalmoTokenizer()
    tokenizer.vocab = vocab_data["vocab"]
    tokenizer.reverse_vocab = {int(k): v for k, v in vocab_data["reverse_vocab"].items()}
    tokenizer.merges = [tuple(m) for m in vocab_data["merges"]]
    print(f"Tokenizer loaded: {len(tokenizer.vocab)} tokens\n")
    
    # Model configuration
    config = {
        'vocab_size': MAX_VOCAB_SIZE,
        'embed_dim': 256,
        'n_heads': 8,
        'n_layers': 12,
        'context_length': 256,
        'dropout': 0.1,
    }
    
    # Load model
    model = PalmoModel(config)
    
    # Try to load fine-tuned best model first, fallback to pretrained best
    tuned_best_path = f"checkpoints/palmo-{TEXT}-tuned-best.pt"
    pretrained_path = f"checkpoints/palmo-{TEXT}-best.pt"
    
    checkpoint_path = tuned_best_path if os.path.exists(tuned_best_path) else pretrained_path
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: No checkpoint found!")
        print(f"Tried: {tuned_best_path}")
        print(f"Tried: {pretrained_path}")
        return
    
    print(f"Loading model from {checkpoint_path}...")
    load_checkpoint(checkpoint_path, model)
    model.eval()
    print(f"Model loaded successfully!\n")
    
    # Interactive loop
    print("Enter your prompt (or 'quit' to exit)")
    print("=" * 50)
    
    temperatures = [0.3, 0.6, 0.9]
    
    while True:
        try:
            user_prompt = input("\nYou: ")
            
            if user_prompt.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if not user_prompt.strip():
                continue
            
            print("\nPalmo:")
            for temp in temperatures:
                response = ask(user_prompt, model, tokenizer, max_tokens=50, temperature=temp)
                print(f"  [T={temp}] {response}")
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
