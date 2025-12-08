"""
BPE Tokenizer Best Practices and Common Patterns

This module demonstrates recommended usage patterns and best practices
for the BPE tokenizer implementation.
"""

import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple
from llm_book.tokenizer.bpe import BPETokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# BEST PRACTICE 1: Configuration Management
# ============================================================================

class TokenizerConfig:
    """Configuration class for tokenizer setup."""
    
    def __init__(
        self,
        vocab_size: int = 10000,
        tokenizer_dir: str = 'tokenizers',
        name: str = 'default'
    ):
        self.vocab_size = vocab_size
        self.tokenizer_dir = Path(tokenizer_dir) / name
        self.name = name
    
    def get_tokenizer_path(self) -> Path:
        """Get full path to tokenizer."""
        return self.tokenizer_dir
    
    def ensure_dir_exists(self) -> None:
        """Ensure tokenizer directory exists."""
        self.tokenizer_dir.mkdir(parents=True, exist_ok=True)


# ============================================================================
# BEST PRACTICE 2: Context Manager for Resource Management
# ============================================================================

class ManagedTokenizer:
    """Context manager for BPE tokenizer lifecycle."""
    
    def __init__(self, config: TokenizerConfig):
        self.config = config
        self.tokenizer = None
    
    def __enter__(self):
        """Initialize tokenizer on context entry."""
        logger.info(f"Loading tokenizer: {self.config.name}")
        
        tokenizer_path = self.config.get_tokenizer_path()
        if tokenizer_path.exists():
            # Load existing
            self.tokenizer = BPETokenizer.load(str(tokenizer_path))
            logger.info(f"Loaded existing tokenizer from {tokenizer_path}")
        else:
            # Create new
            self.tokenizer = BPETokenizer(vocab_size=self.config.vocab_size)
            logger.info(f"Created new tokenizer (vocab_size={self.config.vocab_size})")
        
        return self.tokenizer
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Save tokenizer on context exit."""
        if self.tokenizer and exc_type is None:
            self.config.ensure_dir_exists()
            self.tokenizer.save(str(self.config.get_tokenizer_path()))
            logger.info(f"Saved tokenizer to {self.config.get_tokenizer_path()}")
        
        return False


# ============================================================================
# BEST PRACTICE 3: Dataset Tokenization Pipeline
# ============================================================================

class DatasetTokenizer:
    """Tokenize entire datasets efficiently."""
    
    def __init__(self, tokenizer: BPETokenizer, batch_size: int = 1000):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
    
    def tokenize_file(self, file_path: str, output_path: str) -> int:
        """
        Tokenize entire file and save to output.
        
        Args:
            file_path: Input text file
            output_path: Output NumPy file
            
        Returns:
            Total number of tokens
        """
        logger.info(f"Tokenizing {file_path}")
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        
        tokens = self.tokenizer.encode(text)
        np.save(output_path, tokens)
        
        logger.info(f"Saved {len(tokens)} tokens to {output_path}")
        return len(tokens)
    
    def tokenize_corpus(
        self,
        file_paths: List[str],
        output_dir: str
    ) -> Tuple[int, List[str]]:
        """
        Tokenize corpus of multiple files.
        
        Args:
            file_paths: List of input files
            output_dir: Output directory
            
        Returns:
            Tuple of (total_tokens, output_files)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        total_tokens = 0
        output_files = []
        
        for file_path in file_paths:
            output_file = output_dir / f"{Path(file_path).stem}_tokens.npy"
            tokens = self.tokenize_file(file_path, str(output_file))
            total_tokens += tokens
            output_files.append(str(output_file))
        
        logger.info(f"Total tokens: {total_tokens}")
        return total_tokens, output_files


# ============================================================================
# BEST PRACTICE 4: Batch Encoding with Padding
# ============================================================================

class BatchEncoder:
    """Efficiently encode multiple texts with padding."""
    
    def __init__(self, tokenizer: BPETokenizer, pad_token_id: int = 0):
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id
    
    def encode_batch(
        self,
        texts: List[str],
        max_length: int = None,
        padding: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode batch of texts with optional padding.
        
        Args:
            texts: List of texts to encode
            max_length: Max length (or None for auto)
            padding: Whether to pad sequences
            
        Returns:
            Tuple of (batch_tokens, batch_mask)
        """
        # Encode all texts
        encodings = [self.tokenizer.encode(text) for text in texts]
        
        # Determine max length
        if max_length is None:
            max_length = max(len(e) for e in encodings)
        
        # Pad sequences
        batch_tokens = []
        batch_mask = []
        
        for encoding in encodings:
            # Truncate if too long
            if len(encoding) > max_length:
                encoding = encoding[:max_length]
            
            # Pad if too short
            if padding and len(encoding) < max_length:
                pad_length = max_length - len(encoding)
                encoding = np.pad(
                    encoding,
                    (0, pad_length),
                    'constant',
                    constant_values=self.pad_token_id
                )
            
            batch_tokens.append(encoding)
            
            # Create attention mask (1 for real tokens, 0 for padding)
            mask = np.ones(len(encoding), dtype=np.int32)
            if padding and len(encoding) < max_length:
                mask = np.pad(mask, (0, max_length - len(mask)), constant_values=0)
            batch_mask.append(mask)
        
        return np.stack(batch_tokens), np.stack(batch_mask)


# ============================================================================
# BEST PRACTICE 5: Efficient Token Caching
# ============================================================================

class TokenCache:
    """Cache frequently tokenized texts."""
    
    def __init__(self, tokenizer: BPETokenizer, max_cache_size: int = 1000):
        self.tokenizer = tokenizer
        self.max_cache_size = max_cache_size
        self.cache = {}
    
    def encode(self, text: str) -> np.ndarray:
        """Encode text with caching."""
        if text in self.cache:
            return self.cache[text].copy()
        
        tokens = self.tokenizer.encode(text)
        
        # Add to cache if not full
        if len(self.cache) < self.max_cache_size:
            self.cache[text] = tokens.copy()
        
        return tokens
    
    def clear_cache(self) -> None:
        """Clear cache."""
        self.cache.clear()
    
    def cache_stats(self) -> dict:
        """Get cache statistics."""
        return {
            'size': len(self.cache),
            'max_size': self.max_cache_size,
            'full': len(self.cache) >= self.max_cache_size
        }


# ============================================================================
# BEST PRACTICE 6: Tokenizer Validation
# ============================================================================

class TokenizerValidator:
    """Validate tokenizer quality and performance."""
    
    def __init__(self, tokenizer: BPETokenizer):
        self.tokenizer = tokenizer
    
    def validate_roundtrip(self, test_texts: List[str]) -> dict:
        """
        Validate encode/decode round-trips.
        
        Returns:
            Results dict with success rate and failures
        """
        successes = 0
        failures = []
        
        for text in test_texts:
            try:
                encoded = self.tokenizer.encode(text)
                decoded = self.tokenizer.decode(encoded)
                
                if decoded == text:
                    successes += 1
                else:
                    failures.append({
                        'text': text,
                        'original_len': len(text),
                        'decoded': decoded,
                        'decoded_len': len(decoded)
                    })
            except Exception as e:
                failures.append({'text': text, 'error': str(e)})
        
        return {
            'total': len(test_texts),
            'successes': successes,
            'failures': len(failures),
            'success_rate': successes / len(test_texts) if test_texts else 0.0,
            'failed_cases': failures
        }
    
    def compression_analysis(self, test_texts: List[str]) -> dict:
        """Analyze compression performance."""
        char_count = 0
        token_count = 0
        
        for text in test_texts:
            char_count += len(text)
            encoded = self.tokenizer.encode(text)
            token_count += len(encoded)
        
        compression_ratio = (1 - token_count / char_count) * 100 if char_count > 0 else 0
        
        return {
            'total_characters': char_count,
            'total_tokens': token_count,
            'compression_ratio': compression_ratio,
            'avg_chars_per_token': char_count / token_count if token_count > 0 else 0
        }


# ============================================================================
# BEST PRACTICE 7: Common Usage Pattern
# ============================================================================

def complete_workflow_example():
    """Demonstrate recommended complete workflow."""
    logger.info("=" * 60)
    logger.info("Complete Workflow Example")
    logger.info("=" * 60)
    
    # 1. Setup configuration
    config = TokenizerConfig(vocab_size=1000, name='example_tokenizer')
    
    # 2. Use context manager for lifecycle
    with ManagedTokenizer(config) as tokenizer:
        # 3. Train if new
        if not tokenizer.metadata['trained']:
            training_text = "Machine learning is powerful. Python is great." * 10
            logger.info("Training tokenizer...")
            tokenizer.train(training_text)
        
        # 4. Use batch encoder for efficient processing
        batch_encoder = BatchEncoder(tokenizer)
        texts = ["Hello world", "Python is great", "Tokenization is fun"]
        tokens, masks = batch_encoder.encode_batch(texts)
        
        logger.info(f"Batch shape: {tokens.shape}")
        logger.info(f"Mask shape: {masks.shape}")
        
        # 5. Use token cache for repeated texts
        cache = TokenCache(tokenizer)
        repeated_text = "hello world hello world"
        encoded1 = cache.encode(repeated_text)
        encoded2 = cache.encode(repeated_text)  # From cache
        
        logger.info(f"Cache stats: {cache.cache_stats()}")
        
        # 6. Validate tokenizer
        validator = TokenizerValidator(tokenizer)
        validation = validator.validate_roundtrip(texts)
        logger.info(f"Validation results: {validation['success_rate']*100:.1f}% success")
        
        compression = validator.compression_analysis(texts)
        logger.info(f"Compression ratio: {compression['compression_ratio']:.1f}%")


# ============================================================================
# BEST PRACTICE 8: Error Handling Pattern
# ============================================================================

def safe_tokenization(
    tokenizer: BPETokenizer,
    text: str,
    default_value: np.ndarray = None
) -> np.ndarray:
    """
    Safely tokenize with fallback.
    
    Args:
        tokenizer: BPE tokenizer
        text: Text to tokenize
        default_value: Default if tokenization fails
        
    Returns:
        Token array or default
    """
    try:
        if not tokenizer.metadata.get('trained'):
            logger.warning("Tokenizer not trained, returning empty array")
            return default_value or np.array([], dtype=np.int32)
        
        if not text or len(text.strip()) == 0:
            logger.debug("Empty text, returning empty array")
            return np.array([], dtype=np.int32)
        
        return tokenizer.encode(text)
    
    except Exception as e:
        logger.error(f"Tokenization failed: {e}")
        return default_value or np.array([], dtype=np.int32)


if __name__ == '__main__':
    complete_workflow_example()
