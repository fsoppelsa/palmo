"""
BPE (Byte Pair Encoding) Tokenizer Module

Implements a custom BPE tokenizer for the Palmo project.

To handle spaces between words, it follows the GPT-2 approach:
- The first word starts without a space.
- Subsequent words start with a space.

Author: Fabrizio Soppelsa <fabrizio.soppelsa [@] community.unipa.it>
University of Palermo - Natural Language Processing Course
Academic Year 2025/2026
"""

from typing import List, Dict, Tuple, Optional
from collections import Counter


class PalmoTokenizer:
    def __init__(self, vocab_size=None):
        self.vocab_size = vocab_size

        self.vocab: Dict[str, int] = {}

        self.reverse_vocab: Dict[int, str] = {}

        self.merges: List[Tuple[str, str]] = []

        self.UNK_TOKEN = '<unk>'
        self.EOS_TOKEN = '<eos>'

        # Word-boundary marker (GPT-2 style: Ġ represents the start of a word).
        self.WORD_BOUNDARY = 'Ġ'

    def _get_pair_counts(self, word_splits: Dict[str, List[str]], word_counts: Counter) -> Counter:
        pairs = Counter()
        for word, split in word_splits.items():
            count = word_counts[word]
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pairs[pair] += count
        return pairs

    def _merge_pair(self, word_splits: Dict[str, List[str]], pair_to_merge: Tuple[str, str]) -> Dict[str, List[str]]:
        new_word_splits = {}
        new_token = pair_to_merge[0] + pair_to_merge[1]
        for word, split in word_splits.items():
            new_split = []
            i = 0
            while i < len(split):
                if i < len(split) - 1 and (split[i], split[i+1]) == pair_to_merge:
                    # Skip the two original symbols.
                    new_split.append(new_token)
                    i += 2
                else:
                    new_split.append(split[i])
                    i += 1
            new_word_splits[word] = new_split
        return new_word_splits

    def bpe(self, text: str):
        self.vocab = {self.UNK_TOKEN: 0, self.EOS_TOKEN: 1}

        words = text.split()

        # Preserve word boundaries by adding the Ġ marker 
        # to subsequent words (GPT-2 behavior).
        processed_words = []
        for i, word in enumerate(words):
            processed_words.append(word if i == 0 else self.WORD_BOUNDARY + word)

        word_counts = Counter(processed_words)

        word_splits = {w: list(w) + ['</w>'] for w in word_counts.keys()}

        all_chars = set()
        for split in word_splits.values():
            all_chars.update(split)

        for char in sorted(all_chars):
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)

        # Determine how many merges to run (None = unlimited).
        if self.vocab_size is None or self.vocab_size <= 0:
            max_merges = None
        else:
            max_merges = max(0, self.vocab_size - len(self.vocab))

        merges_done = 0

        # Main BPE merge loop.
        while True:
            pair_counts = self._get_pair_counts(word_splits, word_counts)
            if not pair_counts:
                break

            # Define pairs to discard: avoid merges that include whitespace.
            def is_pair_invalid(pair):
                a, b = pair
                for ch in a:
                    if ch.isspace():
                        return True
                for ch in b:
                    if ch.isspace():
                        return True
                return False

            most_frequent_pair = None
            for pair, _count in pair_counts.most_common():
                if not is_pair_invalid(pair):
                    most_frequent_pair = pair
                    break
            if most_frequent_pair is None:
                break

            word_splits = self._merge_pair(word_splits, most_frequent_pair)

            new_token = most_frequent_pair[0] + most_frequent_pair[1]
            self.vocab[new_token] = len(self.vocab)
            self.merges.append(most_frequent_pair)

            merges_done += 1
            if max_merges is not None and merges_done >= max_merges:
                break

        self.reverse_vocab = {idx: token for token, idx in self.vocab.items()}

    def encode(self, text: str) -> List[int]:
        words = text.split()
        all_token_ids = []
        for i, word in enumerate(words):
            current_word = word if i == 0 else self.WORD_BOUNDARY + word

            tokens = list(current_word) + ['</w>']
            for pair in self.merges:
                new_token = pair[0] + pair[1]
                new_tokens = []
                j = 0
                while j < len(tokens):
                    if j < len(tokens) - 1 and (tokens[j], tokens[j + 1]) == pair:
                        new_tokens.append(new_token)
                        j += 2
                    else:
                        new_tokens.append(tokens[j])
                        j += 1
                tokens = new_tokens
            token_ids = [self.vocab.get(token, self.vocab[self.UNK_TOKEN]) for token in tokens]
            all_token_ids.extend(token_ids)
        return all_token_ids

    def decode(self, token_ids: List[int]) -> str:
        tokens = []
        for idx in token_ids:
            tokens.append(self.reverse_vocab.get(idx, self.UNK_TOKEN))

        text = ''.join(tokens)
        text = text.replace('</w>', '')  # Remove the end-of-word marker.
        text = text.replace(self.WORD_BOUNDARY, ' ')  # Convert Ġ to spaces.
        return text
