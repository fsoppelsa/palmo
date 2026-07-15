"""
Transformer Architecture Module

Implements the multi-head attention mechanism, Transformer blocks, and the complete Transformer model.

Author: Fabrizio Soppelsa <fabrizio.soppelsa [@] community.unipa.it>
University of Palermo - Natural Language Processing Course
Academic Year 2025/2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.
    
    This module implements multi-head attention, enabling the model to focus
    on different positions in the input sequence simultaneously.
    """
    
    def __init__(self, d_in: int, d_out: int, context_length: int, dropout: float, num_heads: int, qkv_bias: bool = False):
        """
        Initializes the multi-head attention layer.
        
        Args:
            d_in: Input dimension
            d_out: Output dimension (must be divisible by num_heads)
            context_length: Maximum sequence length
            dropout: Dropout rate
            num_heads: Number of attention heads
            qkv_bias: If True, adds bias to Q, K, and V projections
        """
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        
        self.out_proj = nn.Linear(d_out, d_out)
        
        self.dropout = nn.Dropout(dropout)
        
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the multi-head attention mechanism.
        
        Args:
            x: Input tensor with shape (batch_size, num_tokens, d_in)
            
        Returns:
            Output tensor with shape (batch_size, num_tokens, d_out)
        """
        b, num_tokens, d_in = x.shape
        
        # STEP 1: Project the input into separate Query, Key, and Value tensors.
        keys = self.W_key(x)     # Shape: (b, num_tokens, d_out).
        queries = self.W_query(x)
        values = self.W_value(x)
        
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        
        attn_scores = queries @ keys.transpose(2, 3)
        
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Shape: (b, num_heads, num_tokens, head_dim).
        context_vec = (attn_weights @ values).transpose(1, 2)
        
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        
        return context_vec


class PalmoModel(nn.Module):
    """
    Implements a decoder-only architecture for language modeling, composed of
    token and positional embeddings, Transformer blocks, final layer
    normalization, and a prediction output head.
    """
    
    def __init__(self, config: dict):
        """
        Initializes the Palmo model.
        
        Args:
            config: Configuration dictionary containing:
                - vocab_size: Vocabulary size
                - embed_dim: Embedding dimension
                - n_heads: Number of attention heads
                - n_layers: Number of Transformer blocks
                - context_length: Maximum sequence length
                - dropout: Dropout rate
        """
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config['vocab_size'], config['embed_dim'])
        
        self.pos_emb = nn.Embedding(config['context_length'], config['embed_dim'])

        # Dropout for regularization.
        self.drop = nn.Dropout(config['dropout'])

        # Stack of Transformer blocks.
        blocks = []
        for _ in range(config['n_layers']):
            blocks.append(TransformerBlock(config))
        self.blocks = nn.ModuleList(blocks)
        
        # Final layer normalization.
        self.ln_f = LayerNorm(config['embed_dim'])

        self.lm_head = nn.Linear(config['embed_dim'], config['vocab_size'], bias=False)

        self.lm_head.weight = self.tok_emb.weight
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Palmo model.
        
        Args:
            token_ids: Input token IDs with shape (batch_size, seq_len)
            
        Returns:
            Logits with shape (batch_size, seq_len, vocab_size)
        """
        # Read the batch size and the current sequence length from the input.
        # token_ids has shape (batch_size, seq_len).
        batch_size, seq_len = token_ids.shape
        
        # Convert each token ID into its learned embedding vector.
        # tok_embeds[0,0,:] = a 256-dimensional vector that semantically represents "the".
        # self.tok_emb.weight.shape = (vocab_size, embed_dim) = (15000, 256)
        # Result: tok_embeds.shape = (batch_size, seq_len, embed_dim) = (32, 256, 256).
        tok_embeds = self.tok_emb(token_ids)  # (batch_size, seq_len, embed_dim)
        
        # Add learned positional embeddings because attention alone has no
        # inherent notion of token order. Positions range from 0 to seq_len - 1.
        # This model uses learned embeddings rather than sinusoidal embeddings.
        # Result: pos_embeds.shape = (seq_len, embed_dim) = (256, 256).
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=token_ids.device)  # [0, 1, 2, ..., seq_len-1]
        )  # (seq_len, embed_dim)

        # Combine token content and positional information. Broadcasting applies
        # the same positional vectors to every sequence in the batch.
        # Result: x.shape = (batch_size, seq_len, embed_dim) = (32, 256, 256).
        x = tok_embeds + pos_embeds  # (batch_size, seq_len, embed_dim)
        
        # Dropout for regularization: randomly zeroes out some elements.
        x = self.drop(x)
        
        # Pass the representations through each decoder-only Transformer block.
        for block in self.blocks:
            x = block(x)
        
        # Normalize the final hidden states before vocabulary projection.
        x = self.ln_f(x)
        
        # Project hidden states to vocabulary logits. The output and token
        # embedding layers share weights through weight tying.
        logits = self.lm_head(x)  # (batch_size, seq_len, vocab_size)
        
        return logits

    
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        """
        Generates text autoregressively, one token at a time.
        
        Args:
            idx: Initial index sequence with shape (batch_size, seq_len)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (>1 more random, <1 more deterministic)
            
        Returns:
            Extended sequence with shape (batch_size, seq_len + max_new_tokens)
        """
        # Repeat for max_new_tokens iterations.
        for _ in range(max_new_tokens):
            # Limit the input to the maximum context.
            idx_cond = idx[:, -self.config['context_length']:]
            
            logits = self.forward(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            probs = F.softmax(logits, dim=-1)
            
            next_token = torch.multinomial(probs, num_samples=1)
            
            idx = torch.cat((idx, next_token), dim=1)

        return idx


class LayerNorm(nn.Module):
    """
    Layer Normalization.
    
    Normalizes the input for each individual example in the batch, rather than
    across the entire batch as Batch Normalization does.
    """
    
    def __init__(self, normalized_shape: int, epsilon: float = 1e-5):
        """
        Initializes the normalization layer.
        
        Args:
            normalized_shape: Shape dimension to normalize
            epsilon: Epsilon for numerical stability
        """
        super().__init__()
        self.normalized_shape = normalized_shape
        self.epsilon = epsilon
        
        # Learn independent scale and bias values for each normalized feature.
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies layer normalization over the final dimension.
        
        Args:
            x: Input tensor
            
        Returns:
            Normalized tensor
        """
        mean = x.mean(dim=-1, keepdim=True)
        stddev = x.std(dim=-1, keepdim=True, unbiased=False)
        
        x_norm = (x - mean) / (stddev + self.epsilon)
        
        return self.weight * x_norm + self.bias


class FeedForward(nn.Module):
    """
    Feed-forward network (MLP) used in Transformer blocks.
    
    Implements a simple two-layer network with intermediate expansion: the
    dimension is expanded by 4×, then returned to its original size. This
    enables the model to learn complex nonlinear transformations.
    """
    
    def __init__(self, config):
        """
        Initializes the feed-forward network.
        
        Args:
            config: Configuration dictionary containing 'embed_dim'
        """
        super().__init__()
        # First layer: expands from embed_dim to 4 * embed_dim.
        # A 4× expansion is standard in Transformers (see “Attention Is All You Need”).
        self.fc1 = nn.Linear(config['embed_dim'], 4 * config['embed_dim'])
        
        # Second layer: returns from 4 * embed_dim to embed_dim.
        self.fc2 = nn.Linear(4 * config['embed_dim'], config['embed_dim'])

    def forward(self, x):
        """
        Applies the feed-forward transformation.
        
        Args:
            x: Input tensor with shape (batch_size, seq_len, embed_dim)
            
        Returns:
            Output tensor with shape (batch_size, seq_len, embed_dim)
        """
        x = self.fc1(x)
        
        x = F.relu(x)
        
        x = self.fc2(x)
        
        return x


class TransformerBlock(nn.Module):
    """
    Single Transformer block with attention and feed-forward layers.
    
    Implements the standard architecture of a Transformer block with:
    - Multi-head attention
    - Feed-forward network
    - Layer normalization
    - Residual connections
    """
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.attention = MultiHeadAttention(
            d_in=config['embed_dim'],
            d_out=config['embed_dim'],
            context_length=config['context_length'],
            dropout=config['dropout'],
            num_heads=config['n_heads'],
            qkv_bias=True
        )
        self.ff = FeedForward(config)
        self.norm1 = LayerNorm(config['embed_dim'])
        self.norm2 = LayerNorm(config['embed_dim'])
        self.drop_residual = nn.Dropout(config['dropout'])
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the Transformer block with residual connections.
        
        Args:
            x: Input tensor with shape (batch_size, seq_len, embed_dim)
            
        Returns:
            Output tensor with shape (batch_size, seq_len, embed_dim)
        """
        # First step: multi-head attention with a residual connection.
        residual = x
        x = self.norm1(x)
        x = self.attention(x)
        x = self.drop_residual(x)
        x = x + residual

        # Second step: feed-forward network with a residual connection.
        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_residual(x)
        x = x + residual

        return x
