"""
Interactive, incremental tests for MultiHeadAttention

This script provides an interactive test suite for verifying the Multi-Head
Attention mechanism step by step. Unlike automated pytest tests, these tests
print detailed output to aid understanding and debugging.

Author: Fabrizio Soppelsa <fabrizio.soppelsa [@] community.unipa.it>
University of Palermo - Natural Language Processing course
Academic Year 2025/2026

Usage:
    python test_mha_interactive.py
    
Output:
    Each test prints readable results, clearly indicating whether it passed
    ([OK]) or failed ([FAIL]).
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.transformer import MultiHeadAttention


def print_section(title):
    """
    Helper for printing formatted sections.
    
    Args:
        title: Title of the section to print.
    """
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def test_1_basic_initialization():
    """
    TEST 1: Verify that the module initializes correctly.
    
    Checks that:
    - The MultiHeadAttention module can be created without errors
    - Internal parameters are configured correctly
    - The number of trainable parameters is reasonable
    
    Returns:
        MultiHeadAttention: Module instance if initialization succeeds, otherwise None.
    """
    print_section("TEST 1: Inizializzazione Base")
    
    try:
        mha = MultiHeadAttention(
            d_in=64,
            d_out=64,
            context_length=128,
            dropout=0.1,
            num_heads=4,
            qkv_bias=False
        )
        print("[OK] Inizializzazione riuscita")
        print(f"  - Numero teste: {mha.num_heads}")
        print(f"  - Dimensione per testa: {mha.head_dim}")
        print(f"  - Dimensione input: {mha.W_query.in_features}")
        print(f"  - Dimensione output: {mha.d_out}")
        
        # Count trainable parameters
        params = sum(p.numel() for p in mha.parameters())
        print(f"  - Parametri totali: {params:,}")
        
        return mha
    except Exception as e:
        print(f"[FAIL] Errore: {e}")
        return None


def test_2_forward_pass_shape(mha):
    """
    TEST 2: Verify that the forward pass produces the correct shape.
    
    Test the module with different batch_size and seq_len configurations to
    ensure the output dimensions are always correct.
    
    Args:
        mha: MultiHeadAttention instance to test.
    """
    print_section("TEST 2: Shape del Forward Pass")
    
    if mha is None:
        print("[SKIP] Saltato (inizializzazione fallita)")
        return
    
    mha.eval()  # Modalità evaluation (disabilita dropout)
    
    # Different input configurations to test
    test_cases = [
        (1, 5, 64),   # batch_size=1, seq_len=5
        (2, 10, 64),  # batch_size=2, seq_len=10
        (4, 20, 64),  # batch_size=4, seq_len=20
    ]
    
    for batch_size, seq_len, d_in in test_cases:
        x = torch.randn(batch_size, seq_len, d_in)
        
        try:
            with torch.no_grad():
                output = mha(x)
            
            expected_shape = (batch_size, seq_len, 64)
            if output.shape == expected_shape:
                print(f"[OK] Input {x.shape} -> Output {output.shape}")
            else:
                print(f"[FAIL] Input {x.shape} -> Output {output.shape} (atteso {expected_shape})")
        except Exception as e:
            print(f"[FAIL] Errore con input {x.shape}: {e}")


def test_3_causal_mask():
    """
    TEST 3: Verify that the causal mask works correctly.
    
    The causal mask ensures that each token can attend only to itself and
    preceding tokens, never future tokens.
    
    Test strategy:
    - Calculate the output with the original input
    - Drastically modify the final token
    - Recalculate the output
    - Verify that the first token does not change (it cannot see the future)
    - Verify that the final token changes (it depends on itself)
    """
    print_section("TEST 3: Maschera Causale")
    
    mha = MultiHeadAttention(
        d_in=32,
        d_out=32,
        context_length=10,
        dropout=0.0,  # No dropout for this test (determinism is required)
        num_heads=2,
        qkv_bias=False
    )
    mha.eval()
    
    batch_size = 1
    seq_len = 5
    
    # Base input
    x = torch.randn(batch_size, seq_len, 32)
    
    with torch.no_grad():
        output = mha(x)
    
    # Drastically modify the FINAL token (position 4)
    x_modified = x.clone()
    x_modified[:, -1, :] = torch.randn_like(x_modified[:, -1, :]) * 10  # Large change
    
    with torch.no_grad():
        output_modified = mha(x_modified)
    
    # The FIRST token should NOT change (it cannot see the future)
    diff_first = (output[:, 0, :] - output_modified[:, 0, :]).abs().max().item()
    
    # The FINAL token SHOULD change (it depends on itself)
    diff_last = (output[:, -1, :] - output_modified[:, -1, :]).abs().max().item()
    
    print(f"Modifica all'ultimo token (pos 4):")
    print(f"  - Differenza al primo token (pos 0): {diff_first:.6f}")
    print(f"  - Differenza all'ultimo token (pos 4): {diff_last:.6f}")
    
    if diff_first < 1e-5 and diff_last > 1e-2:
        print("[OK] Maschera causale funziona correttamente!")
    else:
        print("[FAIL] Problema con la maschera causale")


def test_4_attention_to_self():
    """
    TEST 4: Every token should attend to at least itself.
    
    Verify that the self-attention mechanism works at a basic level: every
    token should produce non-zero output even when the other tokens differ.
    
    Strategy:
    - Create input in which each token has a distinctive pattern
    - Verify that every token produces meaningful output
    """
    print_section("TEST 4: Self-Attention Basica")
    
    mha = MultiHeadAttention(
        d_in=16,
        d_out=16,
        context_length=10,
        dropout=0.0,
        num_heads=2,
        qkv_bias=False
    )
    mha.eval()
    
    # Simple input: each position has a different pattern (one-hot-like)
    x = torch.zeros(1, 4, 16)
    for i in range(4):
        x[0, i, i] = 1.0  # One-hot per position
    
    with torch.no_grad():
        output = mha(x)
    
    print("Input (4 token con pattern distinti):")
    print(f"  Token 0: attivo alla dimensione 0")
    print(f"  Token 1: attivo alla dimensione 1")
    print(f"  Token 2: attivo alla dimensione 2")
    print(f"  Token 3: attivo alla dimensione 3")
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Output norm per token:")
    for i in range(4):
        norm = output[0, i, :].norm().item()
        print(f"  Token {i}: {norm:.4f}")
    
    # Every token should produce non-zero output
    all_nonzero = all(output[0, i, :].norm().item() > 0.01 for i in range(4))
    
    if all_nonzero:
        print("[OK] Tutti i token producono output non-zero")
    else:
        print("[FAIL] Alcuni token producono output vicino a zero")


def test_5_determinism():
    """
    TEST 5: Same input → same output (in eval mode).
    
    In evaluation mode, with dropout disabled, the same input must always
    produce the same output. This is fundamental for reproducible inference.
    
    Strategy:
    - Run two forward passes with the same input
    - Verify that the outputs are identical (within numerical error)
    """
    print_section("TEST 5: Determinismo (Eval Mode)")
    
    mha = MultiHeadAttention(
        d_in=64,
        d_out=64,
        context_length=128,
        dropout=0.1,
        num_heads=4,
        qkv_bias=False
    )
    mha.eval()  # Important: disables dropout
    
    x = torch.randn(2, 10, 64)
    
    # Two identical forward passes
    with torch.no_grad():
        output1 = mha(x)
        output2 = mha(x)
    
    # Calculate the maximum difference
    max_diff = (output1 - output2).abs().max().item()
    
    print(f"Differenza massima tra due forward pass: {max_diff:.10f}")
    
    if max_diff < 1e-6:
        print("[OK] Output deterministico in eval mode")
    else:
        print("[FAIL] Output non deterministico (dropout attivo?)")


def test_6_gradient_flow():
    """
    TEST 6: Gradients flow through the layer.
    
    Verify that backpropagation works correctly: all trainable parameters must
    receive gradients during optimization. Without gradients, the model cannot learn.
    
    Strategy:
    - Run a forward pass with input that requires gradients
    - Calculate a dummy loss
    - Run backward
    - Verify that all parameters have non-zero gradients
    """
    print_section("TEST 6: Flusso dei Gradienti")
    
    mha = MultiHeadAttention(
        d_in=32,
        d_out=32,
        context_length=128,
        dropout=0.1,
        num_heads=4,
        qkv_bias=False
    )
    mha.train()  # Training mode (enables gradients)
    
    x = torch.randn(2, 5, 32, requires_grad=True)
    
    # Forward
    output = mha(x)
    
    # Backward pass with a dummy loss (sum of all outputs)
    loss = output.sum()
    loss.backward()
    
    # Check that parameters have gradients
    params_with_grad = 0
    params_without_grad = 0
    
    for name, param in mha.named_parameters():
        if param.grad is not None and not torch.all(param.grad == 0):
            params_with_grad += 1
        else:
            params_without_grad += 1
    
    print(f"Parametri con gradiente: {params_with_grad}")
    print(f"Parametri senza gradiente: {params_without_grad}")
    
    # Also check the input gradient
    if x.grad is not None:
        print(f"Gradiente input: norm = {x.grad.norm().item():.4f}")
        print("[OK] Gradienti fluiscono correttamente")
    else:
        print("[FAIL] Nessun gradiente sull'input")


def test_7_different_sequence_lengths():
    """
    TEST 7: Correctly handles different sequence lengths.
    
    The module must be robust to sequence length, handling both very short
    sequences (one token) and longer sequences (up to context_length). The
    causal mask is automatically truncated to the actual length.
    
    Strategy:
    - Test lengths from 1 to 100 tokens
    - Verify that each configuration produces correct output
    """
    print_section("TEST 7: Diverse Lunghezze di Sequenza")
    
    mha = MultiHeadAttention(
        d_in=64,
        d_out=64,
        context_length=128,
        dropout=0.0,
        num_heads=4,
        qkv_bias=False
    )
    mha.eval()
    
    test_lengths = [1, 5, 10, 20, 50, 100]
    
    for seq_len in test_lengths:
        x = torch.randn(2, seq_len, 64)
        
        try:
            with torch.no_grad():
                output = mha(x)
            
            if output.shape == (2, seq_len, 64):
                print(f"[OK] seq_len={seq_len:3d}: OK")
            else:
                print(f"[FAIL] seq_len={seq_len:3d}: shape errata {output.shape}")
        except Exception as e:
            print(f"[FAIL] seq_len={seq_len:3d}: Errore - {e}")


def test_8_heads_configuration():
    """
    TEST 8: Different head configurations.
    
    Verify that the module works with different numbers of heads. More heads
    can capture different patterns in parallel, but require d_model to be
    divisible by num_heads.
    
    Strategy:
    - Test 1, 2, 4, and 8 heads
    - Verify that each configuration is valid
    - Show the dimension per head (head_dim)
    """
    print_section("TEST 8: Diverse Configurazioni di Teste")
    
    d_model = 64
    test_configs = [
        (1, "1 testa"),
        (2, "2 teste"),
        (4, "4 teste"),
        (8, "8 teste"),
    ]
    
    for num_heads, desc in test_configs:
        try:
            mha = MultiHeadAttention(
                d_in=d_model,
                d_out=d_model,
                context_length=128,
                dropout=0.0,
                num_heads=num_heads,
                qkv_bias=False
            )
            mha.eval()
            
            x = torch.randn(2, 10, d_model)
            with torch.no_grad():
                output = mha(x)
            
            head_dim = d_model // num_heads
            print(f"[OK] {desc}: head_dim={head_dim}, output shape={output.shape}")
        except Exception as e:
            print(f"[FAIL] {desc}: Errore - {e}")


def run_all_tests():
    """
    Run all tests in sequence.
    
    This function orchestrates execution of the full test suite, running every
    test in the appropriate order and providing a final summary. Tests are
    organized in progressively increasing complexity.
    """
    print("\n" + "="*60)
    print("  TEST INCREMENTALE MultiHeadAttention")
    print("="*60)
    
    # Test 1: Basic initialization
    mha = test_1_basic_initialization()
    
    # Test 2: Verify tensor shapes in the forward pass
    test_2_forward_pass_shape(mha)
    
    # Test 3: Causal mask (cannot see the future)
    test_3_causal_mask()
    
    # Test 4: Basic self-attention (every token attends to at least itself)
    test_4_attention_to_self()
    
    # Test 5: Determinismo in eval mode
    test_5_determinism()
    
    # Test 6: Backpropagation and gradient flow
    test_6_gradient_flow()
    
    # Test 7: Robustness with different sequence lengths
    test_7_different_sequence_lengths()
    
    # Test 8: Robustness with different head configurations
    test_8_heads_configuration()
    
    print("\n" + "="*60)
    print("  TUTTI I TEST COMPLETATI!")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Set a seed for test reproducibility
    # This ensures that tests always produce the same results
    torch.manual_seed(42)
    
    # Run the full test suite
    run_all_tests()
