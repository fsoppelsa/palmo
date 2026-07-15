"""
Tests for the BPE Tokenizer implementation.

This suite verifies that the custom BPE tokenizer works correctly.

Author: Fabrizio Soppelsa <fabrizio.soppelsa [@] community.unipa.it>
University of Palermo - Natural Language Processing course
Academic Year 2025/2026
"""
import sys
from pathlib import Path
import pytest
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.tokenizer import PalmoTokenizer

def get_tokens(tokenizer: PalmoTokenizer, text: str) -> list[str]:
    """Helper function to convert a text to a list of token strings."""
    encoded_ids = tokenizer.encode(text)
    return [tokenizer.reverse_vocab.get(i, "ID_NOT_FOUND") for i in encoded_ids]

def test_1():
    # Test 1: Verify that BPE learns whole tokens from a simple sentence
    # Input: "hello world"
    # Expected output: tokens "hello</w>", " ", "world</w>"

    tokenizer = PalmoTokenizer(vocab_size=100)
    corpus = "hello world hello world"
    tokenizer.bpe(corpus)

    test_input = "hello world"
    tokens = tokenizer.encode(test_input)
    decoded_tokens = [tokenizer.reverse_vocab[i] for i in tokens]

    print("Input:", test_input)
    print("Tokens:", decoded_tokens)

    assert decoded_tokens == ["hello</w>", "Ġworld</w>"]

def test_2():
    # Test 2: Verify that BPE learns individual tokens from repeated words
    # Input: "the cat sat"
    # Expected output: tokens "the</w>", " ", "cat</w>", " ", "sat</w>"

    tokenizer = PalmoTokenizer(vocab_size=10000)
    corpus = "the cat sat on the mat the cat was happy"
    tokenizer.bpe(corpus)

    test_input = "the cat sat"
    tokens = tokenizer.encode(test_input)
    decoded_tokens = [tokenizer.reverse_vocab[i] for i in tokens]

    print("Input:", test_input)
    print("Tokens:", decoded_tokens)

    assert decoded_tokens == ["the</w>", "Ġcat</w>", "Ġsat</w>"]

def test_3():
    # Test 3: Verify that BPE discovers common subwords
    # Input: "learning learner"
    # Expected output: tokens "learning</w>", " ", "learner</w>"

    tokenizer = PalmoTokenizer(vocab_size=100)
    corpus = "learning learner learning learner"
    tokenizer.bpe(corpus)

    test_input = "learning learner"
    tokens = tokenizer.encode(test_input)
    decoded_tokens = [tokenizer.reverse_vocab[i] for i in tokens]

    print("Input:", test_input)
    print("Tokens:", decoded_tokens)

    assert decoded_tokens == ["learning</w>", "Ġlearner</w>"]

def test_4():
    # Test 4: Verify that encode and decode return the original text
    # Input: "the quick brown fox"
    # Expected output: reconstructed original text

    tokenizer = PalmoTokenizer(vocab_size=120)
    corpus = "the quick brown fox jumps over the lazy dog"
    tokenizer.bpe(corpus)

    test_input = "the quick brown fox"
    encoded_ids = tokenizer.encode(test_input)
    decoded_text = tokenizer.decode(encoded_ids)

    print("Input:", test_input)
    print("Tokens:", encoded_ids)

    assert decoded_text.replace("</w>", "") == test_input

def test_5():
    # Test 5: Verify handling of unknown tokens (<unk>)
    # Input: "a z"
    # Expected output: tokens "a</w>", " ", "<unk>", "</w>"

    tokenizer = PalmoTokenizer(vocab_size=100)
    corpus = "a b c"
    tokenizer.bpe(corpus)

    test_input = "a z"
    tokens = tokenizer.encode(test_input)
    decoded_tokens = [tokenizer.reverse_vocab[i] for i in tokens]

    print("Input:", test_input)
    print("Tokens:", decoded_tokens)

    assert decoded_tokens == ["a</w>", "Ġ", "<unk>", "</w>"]
