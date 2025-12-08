"""
Vocabulary Management Module for LLM Learning Project

This module provides vocabulary creation, encoding, and decoding utilities.
Shared across data processing and dataset utilities.

Author: LLM Learning Project
Date: 2025
"""

import pickle
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class VocabularyManager:
    """
    Manages vocabulary creation, encoding, and decoding.
    
    Provides bidirectional character-to-index and index-to-character mappings.
    Used during data processing and inference.
    """
    
    def __init__(self):
        """Initialize VocabularyManager."""
        self.char_to_idx = {}
        self.idx_to_char = {}
        logger.info("VocabularyManager initialized")
    
    def build(self, text: str) -> Dict[str, int]:
        """
        Build vocabulary from text.
        
        Args:
            text (str): Text to extract vocabulary from
            
        Returns:
            Dict[str, int]: Character to index mapping
        """
        try:
            # Get unique characters sorted for consistency
            unique_chars = sorted(set(text))
            
            # Create bidirectional mappings
            self.char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}
            self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
            
            logger.info(f"Vocabulary built - Unique characters: {len(unique_chars)}")
            logger.debug(f"Sample vocab: {dict(list(self.char_to_idx.items())[:10])}")
            
            return self.char_to_idx
            
        except Exception as e:
            logger.error(f"Error building vocabulary: {str(e)}")
            raise
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to numerical sequence.
        
        Args:
            text (str): Text to encode
            
        Returns:
            List[int]: Encoded sequence
            
        Raises:
            ValueError: If vocabulary not built or unknown character found
        """
        try:
            if not self.char_to_idx:
                raise ValueError("Vocabulary not built. Call build() first.")
            
            # Encode character by character
            encoded = []
            unknown_count = 0
            
            for char in text:
                if char in self.char_to_idx:
                    encoded.append(self.char_to_idx[char])
                else:
                    # Handle unknown characters gracefully
                    logger.warning(f"Unknown character encountered: {repr(char)}")
                    unknown_count += 1
            
            if unknown_count > 0:
                logger.warning(f"Total unknown characters: {unknown_count}")
            
            logger.debug(f"Encoded {len(text)} characters to {len(encoded)} tokens")
            return encoded
            
        except Exception as e:
            logger.error(f"Error encoding text: {str(e)}")
            raise
    
    def decode(self, indices: List[int]) -> str:
        """
        Decode numerical sequence to text.
        
        Args:
            indices (List[int]): Encoded sequence
            
        Returns:
            str: Decoded text
            
        Raises:
            ValueError: If vocabulary not built or invalid index
        """
        try:
            if not self.idx_to_char:
                raise ValueError("Vocabulary not built. Call build() first.")
            
            decoded = []
            for idx in indices:
                if idx in self.idx_to_char:
                    decoded.append(self.idx_to_char[idx])
                else:
                    logger.warning(f"Invalid index: {idx}")
            
            return ''.join(decoded)
            
        except Exception as e:
            logger.error(f"Error decoding sequence: {str(e)}")
            raise
    
    def save(self, path: str) -> None:
        """
        Save vocabulary to pickle file.
        
        Args:
            path (str): Path to save vocabulary
        """
        try:
            import os
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            
            vocab_data = {
                'char_to_idx': self.char_to_idx,
                'idx_to_char': self.idx_to_char
            }
            
            with open(path, 'wb') as f:
                pickle.dump(vocab_data, f)
            
            logger.info(f"Vocabulary saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving vocabulary: {str(e)}")
            raise
    
    def load(self, path: str) -> None:
        """
        Load vocabulary from pickle file.
        
        Args:
            path (str): Path to vocabulary file
        """
        try:
            with open(path, 'rb') as f:
                vocab_data = pickle.load(f)
            
            self.char_to_idx = vocab_data['char_to_idx']
            self.idx_to_char = vocab_data['idx_to_char']
            
            logger.info(f"Vocabulary loaded from {path}")
            logger.info(f"Vocabulary size: {len(self.char_to_idx)}")
            
        except Exception as e:
            logger.error(f"Error loading vocabulary: {str(e)}")
            raise
    
    def get_vocab_size(self) -> int:
        """
        Get vocabulary size.
        
        Returns:
            int: Number of unique characters
        """
        return len(self.char_to_idx)
    
    def get_info(self) -> Dict:
        """
        Get vocabulary information and statistics.
        
        Returns:
            Dict: Vocabulary statistics
        """
        info = {
            'vocab_size': len(self.char_to_idx),
            'sample_chars': list(self.char_to_idx.keys())[:20],
            'min_idx': min(self.idx_to_char.keys()) if self.idx_to_char else None,
            'max_idx': max(self.idx_to_char.keys()) if self.idx_to_char else None,
        }
        
        logger.info(f"Vocabulary info - Size: {info['vocab_size']}")
        return info
