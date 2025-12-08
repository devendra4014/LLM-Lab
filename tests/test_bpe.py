"""
Unit tests for BPE Tokenizer implementation.

Tests cover:
- Tokenizer initialization and validation
- Training on various text samples
- Encoding/decoding round-trips
- Vocabulary management
- Persistence (save/load)
- Error handling
"""

import json
import logging
import numpy as np
import os
import pickle
import pytest
import tempfile
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from llm_book.tokenizer.bpe import BPETokenizer


logger = logging.getLogger(__name__)


class TestBPEInitialization:
    """Test BPE tokenizer initialization."""
    
    def test_init_default(self):
        """Test initialization with default vocab size."""
        tokenizer = BPETokenizer()
        assert tokenizer.vocab_size == 256
        assert len(tokenizer.vocab) == 256
        assert len(tokenizer.merges) == 0
        assert not tokenizer.metadata['trained']
    
    def test_init_custom_vocab_size(self):
        """Test initialization with custom vocab size."""
        vocab_size = 1000
        tokenizer = BPETokenizer(vocab_size=vocab_size)
        assert tokenizer.vocab_size == vocab_size
        assert len(tokenizer.vocab) == 256  # Not expanded until trained
    
    def test_init_invalid_vocab_size(self):
        """Test that vocab_size < 256 raises error."""
        with pytest.raises(ValueError, match="vocab_size must be >= 256"):
            BPETokenizer(vocab_size=100)


class TestBPETraining:
    """Test BPE training functionality."""
    
    def test_train_simple_text(self):
        """Test training on simple text."""
        text = "hello hello world"
        tokenizer = BPETokenizer(vocab_size=300)
        tokenizer.train(text)
        
        assert tokenizer.metadata['trained']
        assert tokenizer.get_vocab_size() > 256
        assert tokenizer.get_merges_count() > 0
    
    def test_train_complex_text(self):
        """Test training on longer text."""
        text = (
            "The quick brown fox jumps over the lazy dog. "
            "This is a longer text with multiple sentences. "
            "It should create more token merges."
        )
        tokenizer = BPETokenizer(vocab_size=512)
        tokenizer.train(text)
        
        assert tokenizer.get_vocab_size() > 256
        info = tokenizer.get_vocab_info()
        assert info['trained']
        assert info['num_merges'] > 0
    
    def test_train_empty_text(self):
        """Test that training on empty text raises error."""
        tokenizer = BPETokenizer()
        with pytest.raises(ValueError, match="Training text cannot be empty"):
            tokenizer.train("")
    
    def test_train_whitespace_only(self):
        """Test that training on whitespace-only text raises error."""
        tokenizer = BPETokenizer()
        with pytest.raises(ValueError, match="Training text cannot be empty"):
            tokenizer.train("   \n\t  ")


class TestBPEEncoding:
    """Test BPE encoding functionality."""
    
    def setup_method(self):
        """Setup tokenizer for each test."""
        text = "The quick brown fox jumps over the lazy dog"
        self.tokenizer = BPETokenizer(vocab_size=300)
        self.tokenizer.train(text)
    
    def test_encode_basic(self):
        """Test basic encoding."""
        text = "hello"
        encoded = self.tokenizer.encode(text)
        
        assert isinstance(encoded, np.ndarray)
        assert encoded.dtype == np.int32
        assert len(encoded) > 0
    
    def test_encode_returns_int32(self):
        """Test that encode returns int32 dtype."""
        text = "test encoding"
        encoded = self.tokenizer.encode(text)
        
        assert encoded.dtype == np.int32
    
    def test_encode_before_training(self):
        """Test that encoding before training raises error."""
        tokenizer = BPETokenizer()
        with pytest.raises(RuntimeError, match="must be trained"):
            tokenizer.encode("test")
    
    def test_encode_special_characters(self):
        """Test encoding with special characters."""
        text = "Hello! @#$% 123"
        encoded = self.tokenizer.encode(text)
        
        assert isinstance(encoded, np.ndarray)
        assert len(encoded) > 0
    
    def test_encode_unicode(self):
        """Test encoding with unicode characters."""
        text = "Café résumé naïve"
        encoded = self.tokenizer.encode(text)
        
        assert isinstance(encoded, np.ndarray)
        assert len(encoded) > 0


class TestBPEDecoding:
    """Test BPE decoding functionality."""
    
    def setup_method(self):
        """Setup tokenizer for each test."""
        text = "The quick brown fox jumps over the lazy dog"
        self.tokenizer = BPETokenizer(vocab_size=300)
        self.tokenizer.train(text)
    
    def test_decode_basic(self):
        """Test basic decoding."""
        encoded = np.array([72, 101, 108, 108, 111], dtype=np.int32)  # "hello"
        decoded = self.tokenizer.decode(encoded)
        
        assert isinstance(decoded, str)
        assert len(decoded) > 0
    
    def test_decode_before_training(self):
        """Test that decoding before training raises error."""
        tokenizer = BPETokenizer()
        ids = np.array([100, 101], dtype=np.int32)
        
        with pytest.raises(RuntimeError, match="must be trained"):
            tokenizer.decode(ids)
    
    def test_decode_unknown_token(self):
        """Test decoding with unknown token ID (should log warning)."""
        # Use an ID outside the vocab range
        encoded = np.array([10000], dtype=np.int32)
        decoded = self.tokenizer.decode(encoded)
        
        # Should return empty or partial result with error recovery
        assert isinstance(decoded, str)


class TestBPERoundTrip:
    """Test encode/decode round-trips."""
    
    def setup_method(self):
        """Setup tokenizer for each test."""
        text = ("The quick brown fox jumps over the lazy dog. "
                "Python is great for text processing. "
                "Machine learning requires data preprocessing.")
        self.tokenizer = BPETokenizer(vocab_size=500)
        self.tokenizer.train(text)
    
    def test_roundtrip_simple(self):
        """Test round-trip for simple text."""
        original = "hello world"
        encoded = self.tokenizer.encode(original)
        decoded = self.tokenizer.decode(encoded)
        
        assert decoded == original
    
    def test_roundtrip_complex(self):
        """Test round-trip for complex text."""
        original = "The quick brown fox!"
        encoded = self.tokenizer.encode(original)
        decoded = self.tokenizer.decode(encoded)
        
        assert decoded == original
    
    def test_roundtrip_unicode(self):
        """Test round-trip with unicode."""
        original = "Café naïve"
        encoded = self.tokenizer.encode(original)
        decoded = self.tokenizer.decode(encoded)
        
        assert decoded == original
    
    def test_roundtrip_special_chars(self):
        """Test round-trip with special characters."""
        original = "Email: test@example.com (123)"
        encoded = self.tokenizer.encode(original)
        decoded = self.tokenizer.decode(encoded)
        
        assert decoded == original


class TestBPEVocabulary:
    """Test vocabulary management."""
    
    def test_vocab_size_tracking(self):
        """Test vocabulary size tracking."""
        text = "hello hello hello"
        tokenizer = BPETokenizer(vocab_size=300)
        initial_size = tokenizer.get_vocab_size()
        
        tokenizer.train(text)
        trained_size = tokenizer.get_vocab_size()
        
        assert trained_size >= initial_size
    
    def test_merges_count(self):
        """Test merge count tracking."""
        text = "aaa bbb ccc"
        tokenizer = BPETokenizer(vocab_size=300)
        tokenizer.train(text)
        
        merges = tokenizer.get_merges_count()
        assert merges > 0
    
    def test_vocab_info(self):
        """Test vocabulary info method."""
        text = "test text"
        tokenizer = BPETokenizer(vocab_size=300)
        tokenizer.train(text)
        
        info = tokenizer.get_vocab_info()
        
        assert 'vocab_size' in info
        assert 'num_merges' in info
        assert 'trained' in info
        assert 'base_vocab_size' in info
        assert 'merged_tokens' in info
        assert info['base_vocab_size'] == 256


class TestBPEPersistence:
    """Test tokenizer persistence (save/load)."""
    
    def setup_method(self):
        """Setup tokenizer for each test."""
        text = "The quick brown fox jumps over the lazy dog"
        self.tokenizer = BPETokenizer(vocab_size=300)
        self.tokenizer.train(text)
    
    def test_save_creates_files(self):
        """Test that save creates required files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.tokenizer.save(tmpdir)
            
            assert (Path(tmpdir) / 'tokenizer_vocab.pkl').exists()
            assert (Path(tmpdir) / 'tokenizer_metadata.json').exists()
    
    def test_load_and_encode(self):
        """Test loading and encoding with loaded tokenizer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            self.tokenizer.save(tmpdir)
            
            # Load
            loaded = BPETokenizer.load(tmpdir)
            
            # Test encoding
            text = "hello world"
            encoded_orig = self.tokenizer.encode(text)
            encoded_load = loaded.encode(text)
            
            assert np.array_equal(encoded_orig, encoded_load)
    
    def test_load_and_decode(self):
        """Test loading and decoding with loaded tokenizer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            self.tokenizer.save(tmpdir)
            
            # Load
            loaded = BPETokenizer.load(tmpdir)
            
            # Test decoding
            encoded = np.array([84, 101, 115, 116], dtype=np.int32)
            decoded_orig = self.tokenizer.decode(encoded)
            decoded_load = loaded.decode(encoded)
            
            assert decoded_orig == decoded_load
    
    def test_load_missing_vocab(self):
        """Test that load raises error if vocab file missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create only metadata file
            metadata = {'vocab_size': 300, 'num_merges': 10, 'trained': True}
            with open(Path(tmpdir) / 'tokenizer_metadata.json', 'w') as f:
                json.dump(metadata, f)
            
            with pytest.raises(FileNotFoundError):
                BPETokenizer.load(tmpdir)
    
    def test_load_missing_metadata(self):
        """Test that load raises error if metadata file missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create only vocab file
            vocab_data = {
                'vocab': {i: bytes([i]) for i in range(256)},
                'merges': {},
                'token_to_id': {bytes([i]): i for i in range(256)}
            }
            with open(Path(tmpdir) / 'tokenizer_vocab.pkl', 'wb') as f:
                pickle.dump(vocab_data, f)
            
            with pytest.raises(FileNotFoundError):
                BPETokenizer.load(tmpdir)


class TestBPEFileIO:
    """Test file I/O operations."""
    
    def setup_method(self):
        """Setup tokenizer for each test."""
        text = "The quick brown fox jumps"
        self.tokenizer = BPETokenizer(vocab_size=300)
        self.tokenizer.train(text)
    
    def test_encode_to_file(self):
        """Test encoding text to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / 'tokens.npy'
            text = "hello world"
            
            self.tokenizer.encode_text_to_file(text, str(output_file))
            
            assert output_file.exists()
            loaded = np.load(output_file)
            assert isinstance(loaded, np.ndarray)
    
    def test_decode_from_file(self):
        """Test decoding from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / 'tokens.npy'
            
            # Save tokens
            tokens = np.array([72, 101, 108, 108, 111], dtype=np.int32)
            np.save(input_file, tokens)
            
            # Decode from file
            decoded = self.tokenizer.decode_from_file(str(input_file))
            
            assert isinstance(decoded, str)
            assert len(decoded) > 0


class TestBPEEdgeCases:
    """Test edge cases and error conditions."""
    
    def setup_method(self):
        """Setup tokenizer for each test."""
        text = "test"
        self.tokenizer = BPETokenizer(vocab_size=300)
        self.tokenizer.train(text)
    
    def test_encode_empty_string(self):
        """Test encoding empty string."""
        encoded = self.tokenizer.encode("")
        
        assert isinstance(encoded, np.ndarray)
        assert len(encoded) == 0
    
    def test_decode_empty_array(self):
        """Test decoding empty array."""
        encoded = np.array([], dtype=np.int32)
        decoded = self.tokenizer.decode(encoded)
        
        assert decoded == ""
    
    def test_encode_very_long_text(self):
        """Test encoding very long text."""
        text = "hello " * 10000
        encoded = self.tokenizer.encode(text)
        
        assert isinstance(encoded, np.ndarray)
        assert len(encoded) > 0
    
    def test_list_to_ndarray_conversion(self):
        """Test that decode works with list input."""
        token_list = [84, 101, 115, 116]
        decoded = self.tokenizer.decode(token_list)
        
        assert isinstance(decoded, str)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
