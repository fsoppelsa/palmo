"""
Tests for the Multi-Head Attention implementation.

This suite verifies that the multi-head attention mechanism works correctly
in several scenarios: tensor shapes, causal masking, and different batch and
sequence configurations.

Author: Fabrizio Soppelsa <fabrizio.soppelsa [@] community.unipa.it>
University of Palermo - Natural Language Processing course
Academic Year 2025/2026
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.transformer import MultiHeadAttention


class TestMultiHeadAttention:
    """Test suite for the MultiHeadAttention class."""
    
    @pytest.fixture
    def mha_config(self):
        """
        Standard configuration for tests.
        
        Returns:
            dict: Dictionary containing the model configuration parameters.
        """
        return {
            'd_in': 512,           # Input dimension
            'd_out': 512,          # Output dimension
            'context_length': 1024, # Maximum sequence length
            'dropout': 0.1,        # Dropout rate
            'num_heads': 8,        # Number of attention heads
            'qkv_bias': False      # No bias in the Q, K, and V projections
        }
    
    @pytest.fixture
    def mha(self, mha_config):
        """
        Create a MultiHeadAttention instance for tests.
        
        Args:
            mha_config: Configuration from the fixture.
            
        Returns:
            MultiHeadAttention: Attention module instance.
        """
        return MultiHeadAttention(**mha_config)
    
    def test_output_shape(self, mha):
        """
        Test 1: Verify that the output shape is correct.
        
        This test verifies that the Multi-Head Attention module produces
        output with the expected dimensions (batch_size, seq_len, d_out).
        """
        batch_size = 4
        seq_len = 10
        d_in = 512
        d_out = 512
        
        # Create a random input tensor
        x = torch.randn(batch_size, seq_len, d_in)
        
        # Run the forward pass
        output = mha(x)
        
        # Verify that the output shape matches the expected shape
        assert output.shape == (batch_size, seq_len, d_out), \
            f"Forma attesa {(batch_size, seq_len, d_out)}, ottenuta {output.shape}"
    
    def test_heads_divisible_invariant(self):
        """
        Test 2: Verify that num_heads divides d_out evenly.
        
        The number of heads must divide the output dimension exactly to allow
        splitting into head_dim = d_out // num_heads. This test verifies that
        invalid configurations raise an error.
        """
        d_in = 512
        d_out = 512
        num_heads = 7  # NOT divisible by 512 → expected error
        context_length = 1024
        dropout = 0.1
        
        # It must raise an AssertionError
        with pytest.raises(AssertionError, match="d_out must be divisible by num_heads"):
            MultiHeadAttention(d_in=d_in, d_out=d_out, context_length=context_length, 
                             dropout=dropout, num_heads=num_heads)
    
    def test_causal_masking(self, mha):
        """
        Test 3: Verify causal mask behavior.
        
        The causal mask prevents each token from attending to future tokens in
        the sequence. This test verifies that modifying a future token does
        not affect the output of earlier tokens.
        """
        batch_size = 2
        seq_len = 5
        d_in = 512
        d_out = 512
        
        # Set evaluation mode to disable dropout (deterministic test)
        mha.eval()
        
        # Create the input tensor
        x = torch.randn(batch_size, seq_len, d_in)
        
        # Forward pass (the mask is already integrated into the module)
        with torch.no_grad():
            output = mha(x)
        
        # Verify that the output shape is correct
        assert output.shape == (batch_size, seq_len, d_out), \
            f"Forma attesa {(batch_size, seq_len, d_out)}, ottenuta {output.shape}"
        
        # Verify that the output is not all zero (the model actually computed something)
        assert not torch.allclose(output, torch.zeros_like(output)), \
            "L'output non dovrebbe essere tutto zero"
        
        # Verify the causal mask by checking whether changing future tokens affects the current position
        # Output at position i must NOT change when tokens at position > i are modified
        x_modified = x.clone()
        x_modified[:, -1, :] = torch.randn_like(x_modified[:, -1, :])  # Modify the final token
        
        with torch.no_grad():
            output_modified = mha(x_modified)
        
        # The first token must remain unchanged (not affected by the future)
        # Allow small numerical differences
        assert torch.allclose(output[:, 0, :], output_modified[:, 0, :], atol=1e-6), \
            "L'output del primo token non dovrebbe essere influenzato dall'ultimo token per via della maschera causale"
    
    def test_different_batch_sizes(self, mha):
        """
        Test 4: Verify that the model handles different batch sizes.
        
        The module must work correctly with any batch size, from a single
        example to larger batches.
        """
        d_in = 512
        d_out = 512
        seq_len = 8
        
        # Test different batch sizes
        for batch_size in [1, 2, 8, 16]:
            x = torch.randn(batch_size, seq_len, d_in)
            output = mha(x)
            assert output.shape == (batch_size, seq_len, d_out)
    
    def test_different_sequence_lengths(self, mha):
        """
        Test 5: Verify that the model handles different sequence lengths.
        
        The module must work with variable-length sequences, from one token to
        longer sequences (within context_length). The mask is automatically
        truncated to the current length.
        """
        batch_size = 4
        d_in = 512
        d_out = 512
        
        # Test different sequence lengths
        for seq_len in [1, 5, 10, 20]:
            x = torch.randn(batch_size, seq_len, d_in)
            output = mha(x)
            assert output.shape == (batch_size, seq_len, d_out)
