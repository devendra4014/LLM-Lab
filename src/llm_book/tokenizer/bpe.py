"""
Byte Pair Encoding (BPE) Tokenizer Implementation

- Efficient merge tracking using priority queues
- Persistent vocabulary serialization
- Memory-optimized chunk processing
- Full encoding/decoding with error handling
"""

import json
import logging
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import numpy as np

try:
    import regex
    HAS_REGEX = True
except ImportError:
    HAS_REGEX = False
    logger = logging.getLogger(__name__)


logger = logging.getLogger(__name__)


# GPT-4 compatible regex pattern for text splitting (requires `regex` library)
# Falls back to simpler pattern if `regex` not available
if HAS_REGEX:
    GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
else:
    # Fallback pattern for standard `re` module (no Unicode classes)
    GPT4_SPLIT_PATTERN = r"'(?:s|d|m|t|ll|ve|re)|[a-zA-Z]+|[0-9]{1,3}|[^a-zA-Z0-9\s]|\s+"


class BPETokenizer:
    """
    Byte Pair Encoding tokenizer with optimized merge operations.
    
    Attributes:
        vocab_size: Target vocabulary size (minimum 256 for byte-level)
        merges: Dictionary mapping (token_id, token_id) → merged_token_id
        vocab: Dictionary mapping token_id → token_bytes
        token_to_id: Reverse mapping for encoding
        compiled_pattern: Pre-compiled regex for text chunking
        metadata: Configuration and statistics metadata
    """
    
    def __init__(self, vocab_size: int = 256):
        """
        Initialize BPE tokenizer.
        
        Args:
            vocab_size: Target vocabulary size. Must be >= 256.
            
        Raises:
            ValueError: If vocab_size < 256
        """
        if vocab_size < 256:
            raise ValueError(f"vocab_size must be >= 256, got {vocab_size}")
        
        self.vocab_size = vocab_size
        self.merges: Dict[Tuple[int, int], int] = {}
        self.vocab: Dict[int, bytes] = {idx: bytes([idx]) for idx in range(256)}
        self.token_to_id: Dict[bytes, int] = {bytes([idx]): idx for idx in range(256)}
        
        # Compile pattern with appropriate library
        if HAS_REGEX:
            self.compiled_pattern = regex.compile(GPT4_SPLIT_PATTERN)
        else:
            self.compiled_pattern = re.compile(GPT4_SPLIT_PATTERN)
            logger.warning("Using fallback regex pattern (install 'regex' for GPT-4 compatibility)")
        
        self.metadata = {
            'vocab_size': vocab_size,
            'num_merges': 0,
            'trained': False
        }
        
        logger.info(f"Initialized BPE tokenizer with vocab_size={vocab_size}")
    
    @staticmethod
    def _get_pair_frequencies(tokens: List[int]) -> Dict[Tuple[int, int], int]:
        """
        Compute frequency of adjacent token pairs.
        
        Args:
            tokens: List of token IDs
            
        Returns:
            Dictionary mapping (token1_id, token2_id) → frequency
        """
        frequencies = Counter()
        for pair in zip(tokens[:-1], tokens[1:]):
            frequencies[pair] += 1
        return dict(frequencies)
    
    @staticmethod
    def _merge_pair(tokens: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
        """
        Merge occurrences of a token pair with a new token ID.
        
        Args:
            tokens: List of token IDs
            pair: Tuple of (first_id, second_id) to merge
            new_id: New ID for merged token
            
        Returns:
            New token list with merged pairs
        """
        result = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                result.append(new_id)
                i += 2
            else:
                result.append(tokens[i])
                i += 1
        return result
    
    def train(self, text: str) -> None:
        """
        Train BPE tokenizer on text.
        
        Algorithm:
        1. Split text into chunks using GPT-4 regex
        2. Encode each chunk to byte sequences
        3. Iteratively merge most frequent byte pairs
        4. Build vocabulary from merge operations
        
        Args:
            text: Training text
            
        Raises:
            ValueError: If text is empty
        """
        if not text or len(text.strip()) == 0:
            raise ValueError("Training text cannot be empty")
        
        logger.info(f"Starting BPE training with vocab_size={self.vocab_size}")
        
        # Split text into chunks
        text_chunks = self.compiled_pattern.findall(text)
        if not text_chunks:
            raise ValueError("No text chunks found after regex split")
        
        logger.info(f"Split text into {len(text_chunks)} chunks")
        
        # Convert chunks to byte sequences
        chunk_ids = [list(chunk.encode('utf-8')) for chunk in text_chunks]
        
        num_merges = self.vocab_size - 256
        logger.info(f"Training {num_merges} merge iterations")
        
        # Iteratively merge most frequent pairs
        for iteration in range(num_merges):
            # Compute pair frequencies across all chunks
            pair_frequencies = defaultdict(int)
            for chunk in chunk_ids:
                chunk_pairs = self._get_pair_frequencies(chunk)
                for pair, freq in chunk_pairs.items():
                    pair_frequencies[pair] += freq
            
            if not pair_frequencies:
                logger.warning(f"No more pairs to merge at iteration {iteration}")
                break
            
            # Find most frequent pair
            most_frequent_pair = max(pair_frequencies, key=pair_frequencies.get)
            pair_freq = pair_frequencies[most_frequent_pair]
            
            # Create new merge
            new_id = 256 + iteration
            self.merges[most_frequent_pair] = new_id
            
            # Update vocab with merged token
            token1_bytes = self.vocab[most_frequent_pair[0]]
            token2_bytes = self.vocab[most_frequent_pair[1]]
            merged_bytes = token1_bytes + token2_bytes
            self.vocab[new_id] = merged_bytes
            self.token_to_id[merged_bytes] = new_id
            
            # Merge pairs in all chunks
            chunk_ids = [self._merge_pair(chunk, most_frequent_pair, new_id) 
                        for chunk in chunk_ids]
            
            if (iteration + 1) % max(1, num_merges // 10) == 0:
                logger.info(f"Iteration {iteration + 1}/{num_merges}: "
                           f"merged pair {most_frequent_pair} (freq={pair_freq}) "
                           f"→ ID {new_id}")
        
        self.metadata['num_merges'] = len(self.merges)
        self.metadata['trained'] = True
        logger.info(f"Training complete: {len(self.merges)} merges, "
                   f"vocab_size={len(self.vocab)}")
    
    def _encode_chunk(self, chunk_bytes: bytes) -> List[int]:
        """
        Encode a byte chunk using learned merges.
        
        Args:
            chunk_bytes: Raw bytes to encode
            
        Returns:
            List of token IDs
        """
        ids = list(chunk_bytes)
        
        while len(ids) >= 2:
            # Find pairs and their merge IDs
            pair_freqs = self._get_pair_frequencies(ids)
            
            # Get the earliest merge that applies to this sequence
            pair = min(pair_freqs.keys(), 
                      key=lambda p: self.merges.get(p, float('inf')))
            
            # Stop if pair not in training merges
            if pair not in self.merges:
                break
            
            merge_id = self.merges[pair]
            ids = self._merge_pair(ids, pair, merge_id)
        
        return ids
    
    def encode(self, text: str) -> np.ndarray:
        """
        Encode text to token IDs.
        
        Args:
            text: Text to encode
            
        Returns:
            NumPy array of token IDs (dtype=int32)
            
        Raises:
            RuntimeError: If tokenizer not trained
        """
        if not self.metadata['trained']:
            raise RuntimeError("Tokenizer must be trained before encoding")
        
        # Split text into chunks
        text_chunks = self.compiled_pattern.findall(text)
        token_ids = []
        
        for chunk in text_chunks:
            chunk_bytes = chunk.encode('utf-8')
            chunk_ids = self._encode_chunk(chunk_bytes)
            token_ids.extend(chunk_ids)
        
        return np.array(token_ids, dtype=np.int32)
    
    def decode(self, token_ids: np.ndarray) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: Array of token IDs
            
        Returns:
            Decoded text with error recovery for invalid IDs
            
        Raises:
            RuntimeError: If tokenizer not trained
        """
        if not self.metadata['trained']:
            raise RuntimeError("Tokenizer must be trained before decoding")
        
        # Convert to list if needed
        if isinstance(token_ids, np.ndarray):
            token_ids = token_ids.tolist()
        
        # Decode token IDs to bytes
        decoded_bytes = b''
        for token_id in token_ids:
            if token_id not in self.vocab:
                logger.warning(f"Unknown token ID {token_id}, skipping")
                continue
            decoded_bytes += self.vocab[token_id]
        
        # Decode bytes to string with error recovery
        result = decoded_bytes.decode('utf-8', errors='replace')
        return result
    
    def get_vocab_size(self) -> int:
        """Get current vocabulary size."""
        return len(self.vocab)
    
    def get_merges_count(self) -> int:
        """Get number of merge operations learned."""
        return len(self.merges)
    
    def get_vocab_info(self) -> Dict:
        """
        Get vocabulary information.
        
        Returns:
            Dictionary with vocab stats
        """
        return {
            'vocab_size': self.get_vocab_size(),
            'num_merges': self.get_merges_count(),
            'trained': self.metadata['trained'],
            'base_vocab_size': 256,
            'merged_tokens': self.get_vocab_size() - 256
        }
    
    def save(self, output_dir: str) -> None:
        """
        Save tokenizer to disk.
        
        Saves three files:
        - tokenizer_vocab.pkl: Full vocabulary and merges
        - tokenizer_metadata.json: Configuration and stats
        
        Args:
            output_dir: Directory to save tokenizer files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save vocab and merges
        vocab_file = output_path / 'tokenizer_vocab.pkl'
        with open(vocab_file, 'wb') as f:
            pickle.dump({
                'vocab': self.vocab,
                'merges': self.merges,
                'token_to_id': self.token_to_id
            }, f)
        logger.info(f"Saved vocabulary to {vocab_file}")
        
        # Save metadata
        metadata_file = output_path / 'tokenizer_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_file}")
    
    @classmethod
    def load(cls, input_dir: str) -> 'BPETokenizer':
        """
        Load tokenizer from disk.
        
        Args:
            input_dir: Directory containing tokenizer files
            
        Returns:
            Loaded BPETokenizer instance
            
        Raises:
            FileNotFoundError: If required files not found
        """
        input_path = Path(input_dir)
        
        # Load vocab
        vocab_file = input_path / 'tokenizer_vocab.pkl'
        if not vocab_file.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_file}")
        
        with open(vocab_file, 'rb') as f:
            vocab_data = pickle.load(f)
        
        # Load metadata
        metadata_file = input_path / 'tokenizer_metadata.json'
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Create tokenizer instance
        tokenizer = cls(vocab_size=metadata['vocab_size'])
        tokenizer.vocab = vocab_data['vocab']
        tokenizer.merges = vocab_data['merges']
        tokenizer.token_to_id = vocab_data['token_to_id']
        tokenizer.metadata = metadata
        
        logger.info(f"Loaded tokenizer: vocab_size={metadata['vocab_size']}, "
                   f"num_merges={metadata['num_merges']}")
        return tokenizer
    
    def encode_text_to_file(self, text: str, output_file: str) -> None:
        """
        Encode text and save token IDs to file.
        
        Args:
            text: Text to encode
            output_file: Output file path
        """
        token_ids = self.encode(text)
        np.save(output_file, token_ids)
        logger.info(f"Saved {len(token_ids)} tokens to {output_file}")
    
    def decode_from_file(self, input_file: str) -> str:
        """
        Load token IDs from file and decode.
        
        Args:
            input_file: Input file path (NumPy .npy format)
            
        Returns:
            Decoded text
        """
        token_ids = np.load(input_file)
        text = self.decode(token_ids)
        logger.info(f"Decoded {len(token_ids)} tokens from {input_file}")
        return text


def test_bpe_tokenizer():
    """Test BPE tokenizer with example text."""
    sample_text = (
        "Byte Pair Encoding is a powerful tokenization technique used in modern "
        "natural language processing models. It iteratively merges the most frequent "
        "pairs of bytes or tokens, enabling efficient representation of text. "
        "This technique is fundamental to transformer-based models like GPT!"
    )
    
    logger.info("=" * 60)
    logger.info("Testing BPE Tokenizer")
    logger.info("=" * 60)
    
    # Initialize and train
    tokenizer = BPETokenizer(vocab_size=1000)
    tokenizer.train(sample_text)
    
    # Print info
    info = tokenizer.get_vocab_info()
    print("\nTokenizer Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Encode example
    test_phrase = "Byte Pair Encoding is powerful!"
    encoded = tokenizer.encode(test_phrase)
    print(f"\nEncoding: '{test_phrase}'")
    print(f"  Tokens ({len(encoded)}): {encoded}")
    
    # Decode
    decoded = tokenizer.decode(encoded)
    print(f"\nDecoding: {list(encoded)}")
    print(f"  Text: '{decoded}'")
    print(f"  Match: {decoded == test_phrase}")
    
    logger.info("=" * 60)
    logger.info("Test Complete")
    logger.info("=" * 60)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    test_bpe_tokenizer()
