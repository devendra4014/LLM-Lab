"""
This module handles the complete data pipeline for processing raw text files
into encoded datasets ready for model training. It includes text normalization,
vocabulary building, encoding, and dataset creation.
"""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter

import numpy as np

from .vocabulary import VocabularyManager


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TextNormalizer:
    """
    Handles text normalization and cleaning operations.
    
    Attributes:
        lowercase (bool): Whether to convert text to lowercase
        remove_special (bool): Whether to remove special characters
    """
    
    def __init__(self, lowercase: bool = False, remove_special: bool = False):
        """
        Initialize TextNormalizer.
        
        Args:
            lowercase (bool): Convert text to lowercase. Default: False
            remove_special (bool): Remove special characters. Default: False
        """
        self.lowercase = lowercase
        self.remove_special = remove_special
        logger.info(f"TextNormalizer initialized - lowercase: {lowercase}, "
                   f"remove_special: {remove_special}")
    
    def normalize(self, text: str) -> str:
        """
        Normalize text by cleaning and standardizing.
        
        Args:
            text (str): Raw text to normalize
            
        Returns:
            str: Normalized text
        """
        try:
            # Handle encoding artifacts
            text = self.handle_encoding(text)
            
            # Normalize line breaks and whitespace
            text = text.replace('\r\n', '\n')  # Windows line breaks
            text = text.replace('\r', '\n')     # Mac line breaks
            
            # Remove excessive whitespace while preserving structure
            lines = text.split('\n')
            lines = [line.rstrip() for line in lines]  # Remove trailing spaces
            text = '\n'.join(lines)
            
            # Remove leading/trailing whitespace
            text = text.strip()
            
            # Apply optional transformations
            if self.lowercase:
                text = text.lower()
                logger.debug("Text converted to lowercase")
            
            if self.remove_special:
                # Keep alphanumeric, spaces, and basic punctuation
                text = ''.join(
                    char if char.isalnum() or char in ' \n.,!?;:\'"' 
                    else '' 
                    for char in text
                )
                logger.debug("Special characters removed")
            
            logger.info("Text normalization completed")
            return text
            
        except Exception as e:
            logger.error(f"Error during normalization: {str(e)}")
            raise
    
    def handle_encoding(self, text: str) -> str:
        """
        Handle encoding artifacts and invalid characters.
        
        Args:
            text (str): Text potentially with encoding issues
            
        Returns:
            str: Cleaned text
        """
        try:
            # Replace common encoding artifacts
            replacements = {
                '\ufeff': '',      # BOM
                '\u200b': '',      # Zero-width space
                '\u200c': '',      # Zero-width non-joiner
                '\\n': '\n',       # Escaped newlines
                '\\t': '\t',       # Escaped tabs
            }
            
            for artifact, replacement in replacements.items():
                text = text.replace(artifact, replacement)
            
            return text
            
        except Exception as e:
            logger.warning(f"Error handling encoding: {str(e)}")
            return text


class DataProcessor:
    """
    Main data processing pipeline for converting raw text to training datasets.
    
    Handles loading, cleaning, encoding, and saving of text data for LLM training.
    """
    
    # Default configuration
    DEFAULT_CONFIG = {
        'chunk_size': 1000,
        'seq_length': 128,
        'validation_split': 0.1,
        'lowercase': False,
        'encoding': 'utf-8',
        'remove_special_chars': False,
        'vocab_size': None,
    }
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize DataProcessor.
        
        Args:
            config (Dict, optional): Configuration dictionary. Uses defaults if not provided.
        """
        self.config = {**self.DEFAULT_CONFIG}
        if config:
            self.config.update(config)
        
        self.normalizer = TextNormalizer(
            lowercase=self.config['lowercase'],
            remove_special=self.config['remove_special_chars']
        )
        self.vocab_manager = VocabularyManager()
        self.stats = {}
        
        logger.info(f"DataProcessor initialized with config: {self.config}")
    
    def load_raw_text(self, file_path: str) -> str:
        """
        Load raw text from file.
        
        Args:
            file_path (str): Path to raw text file
            
        Returns:
            str: Raw text content
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            file_size = os.path.getsize(file_path)
            logger.info(f"Loading file: {file_path} (Size: {file_size / 1024:.2f} KB)")
            
            with open(file_path, 'r', encoding=self.config['encoding']) as f:
                text = f.read()
            
            logger.info(f"File loaded successfully. Characters: {len(text)}")
            return text
            
        except UnicodeDecodeError as e:
            logger.error(f"Encoding error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error loading file: {str(e)}")
            raise
    
    def load_and_clean(self, file_path: str) -> str:
        """
        Load and clean text in one step.
        
        Args:
            file_path (str): Path to raw text file
            
        Returns:
            str: Cleaned text
        """
        raw_text = self.load_raw_text(file_path)
        cleaned_text = self.normalizer.normalize(raw_text)
        return cleaned_text
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text to numerical sequence using vocabulary.
        
        Args:
            text (str): Text to encode
            
        Returns:
            np.ndarray: Encoded sequence
        """
        try:
            encoded_list = self.vocab_manager.encode(text)
            encoded_array = np.array(encoded_list, dtype=np.int32)
            logger.info(f"Text encoded - Shape: {encoded_array.shape}, Dtype: {encoded_array.dtype}")
            return encoded_array
            
        except Exception as e:
            logger.error(f"Error encoding text: {str(e)}")
            raise
    
    def create_dataset(self, encoded_text: np.ndarray, 
                      seq_length: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create training dataset with input-target pairs.
        
        Args:
            encoded_text (np.ndarray): Encoded text sequence
            seq_length (int, optional): Sequence length. Uses config if not provided.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (input_sequences, target_sequences)
        """
        seq_length = seq_length or self.config['seq_length']
        
        try:
            # Create overlapping sequences
            inputs = []
            targets = []
            
            for i in range(len(encoded_text) - seq_length):
                inputs.append(encoded_text[i:i + seq_length])
                targets.append(encoded_text[i + seq_length])
            
            inputs = np.array(inputs, dtype=np.int32)
            targets = np.array(targets, dtype=np.int32)
            
            logger.info(f"Dataset created - Inputs shape: {inputs.shape}, "
                       f"Targets shape: {targets.shape}")
            
            return inputs, targets
            
        except Exception as e:
            logger.error(f"Error creating dataset: {str(e)}")
            raise
    
    def split_dataset(self, inputs: np.ndarray, targets: np.ndarray,
                     validation_split: Optional[float] = None) -> Dict:
        """
        Split dataset into train and validation sets.
        
        Args:
            inputs (np.ndarray): Input sequences
            targets (np.ndarray): Target sequences
            validation_split (float, optional): Fraction for validation. Uses config if not provided.
            
        Returns:
            Dict: Dictionary with 'train' and 'validation' splits
        """
        validation_split = validation_split or self.config['validation_split']
        
        try:
            split_idx = int(len(inputs) * (1 - validation_split))
            
            splits = {
                'train': {
                    'inputs': inputs[:split_idx],
                    'targets': targets[:split_idx]
                },
                'validation': {
                    'inputs': inputs[split_idx:],
                    'targets': targets[split_idx:]
                }
            }
            
            logger.info(f"Dataset split - Train: {splits['train']['inputs'].shape[0]}, "
                       f"Validation: {splits['validation']['inputs'].shape[0]}")
            
            return splits
            
        except Exception as e:
            logger.error(f"Error splitting dataset: {str(e)}")
            raise
    
    def validate_and_report(self, data: np.ndarray) -> Dict:
        """
        Validate data and generate statistics report.
        
        Args:
            data (np.ndarray): Data to validate
            
        Returns:
            Dict: Validation report with statistics
        """
        try:
            report = {
                'shape': data.shape,
                'dtype': str(data.dtype),
                'min_value': int(np.min(data)),
                'max_value': int(np.max(data)),
                'mean_value': float(np.mean(data)),
                'unique_values': int(len(np.unique(data))),
                'memory_usage_mb': data.nbytes / (1024 * 1024)
            }
            
            logger.info("Data Validation Report:")
            for key, value in report.items():
                logger.info(f"  {key}: {value}")
            
            return report
            
        except Exception as e:
            logger.error(f"Error validating data: {str(e)}")
            raise
    
    def save_processed_data(self, data: Dict, output_dir: str, dataset_name: str) -> None:
        """
        Save processed data and metadata to disk.
        
        Args:
            data (Dict): Dictionary containing processed data
            output_dir (str): Output directory path
            dataset_name (str): Name of the dataset
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save encoded data
            encoded_path = os.path.join(output_dir, f'{dataset_name}_encoded.npy')
            np.save(encoded_path, data['encoded'])
            logger.info(f"Encoded data saved to {encoded_path}")
            
            # Save dataset splits
            if 'splits' in data:
                splits_path = os.path.join(output_dir, f'{dataset_name}_splits.pkl')
                with open(splits_path, 'wb') as f:
                    pickle.dump(data['splits'], f)
                logger.info(f"Dataset splits saved to {splits_path}")
            
            # Save vocabulary
            vocab_path = os.path.join(output_dir, f'{dataset_name}_vocab.pkl')
            self.vocab_manager.save(vocab_path)
            logger.info(f"Vocabulary saved to {vocab_path}")
            
            # Save metadata
            metadata = {
                'dataset_name': dataset_name,
                'config': self.config,
                'vocab_size': len(self.vocab_manager.char_to_idx),
                'encoded_shape': data['encoded'].shape if isinstance(data['encoded'], np.ndarray) else None,
                'stats': self.stats
            }
            
            metadata_path = os.path.join(output_dir, f'{dataset_name}_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Metadata saved to {metadata_path}")
            
        except Exception as e:
            logger.error(f"Error saving processed data: {str(e)}")
            raise
    
    def process_pipeline(self, input_file: str, output_dir: str, 
                        dataset_name: Optional[str] = None) -> Dict:
        """
        Complete processing pipeline from raw text to training dataset.
        
        Args:
            input_file (str): Path to raw text file
            output_dir (str): Output directory for processed data
            dataset_name (str, optional): Name for output files. Uses input filename if not provided.
            
        Returns:
            Dict: Processing summary and statistics
        """
        try:
            logger.info("=" * 60)
            logger.info("Starting data processing pipeline")
            logger.info("=" * 60)
            
            # Determine dataset name
            if dataset_name is None:
                dataset_name = Path(input_file).stem
            
            # Step 1: Load and clean
            logger.info("\nStep 1: Loading and cleaning text...")
            text = self.load_and_clean(input_file)
            logger.info(f"Cleaned text size: {len(text)} characters")
            
            # Step 2: Build vocabulary
            logger.info("\nStep 2: Building vocabulary...")
            self.vocab_manager.build(text)
            vocab_size = len(self.vocab_manager.char_to_idx)
            logger.info(f"Vocabulary size: {vocab_size}")
            
            # Step 3: Encode text
            logger.info("\nStep 3: Encoding text...")
            encoded_text = self.encode_text(text)
            
            # Step 4: Create dataset
            logger.info("\nStep 4: Creating training dataset...")
            inputs, targets = self.create_dataset(encoded_text)
            
            # Step 5: Split dataset
            logger.info("\nStep 5: Splitting into train/validation...")
            splits = self.split_dataset(inputs, targets)
            
            # Step 6: Validate data
            logger.info("\nStep 6: Validating data...")
            input_stats = self.validate_and_report(inputs)
            target_stats = self.validate_and_report(targets)
            
            self.stats = {
                'input_stats': input_stats,
                'target_stats': target_stats,
                'vocab_size': vocab_size,
                'text_length': len(text),
                'total_sequences': len(inputs)
            }
            
            # Step 7: Save processed data
            logger.info("\nStep 7: Saving processed data...")
            data_to_save = {
                'encoded': encoded_text,
                'splits': splits
            }
            self.save_processed_data(data_to_save, output_dir, dataset_name)
            
            # Generate summary
            summary = {
                'dataset_name': dataset_name,
                'input_file': input_file,
                'output_dir': output_dir,
                'status': 'success',
                'stats': self.stats,
                'config': self.config
            }
            
            logger.info("\n" + "=" * 60)
            logger.info("Pipeline completed successfully!")
            logger.info("=" * 60)
            logger.info(f"\nSummary:")
            logger.info(f"  Original text: {len(text)} characters")
            logger.info(f"  Vocabulary size: {vocab_size}")
            logger.info(f"  Total training sequences: {len(inputs)}")
            logger.info(f"  Train sequences: {splits['train']['inputs'].shape[0]}")
            logger.info(f"  Validation sequences: {splits['validation']['inputs'].shape[0]}")
            
            return summary
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise


# Example usage and testing
if __name__ == "__main__":
    # Example configuration
    config = {
        'chunk_size': 1000,
        'seq_length': 128,
        'validation_split': 0.1,
        'lowercase': False,
        'encoding': 'utf-8',
        'remove_special_chars': False,
    }
    
    # Initialize processor
    processor = DataProcessor(config)
    
    # Process example datasets
    raw_data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'raw'
    processed_data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'processed'
    
    if raw_data_dir.exists():
        for text_file in raw_data_dir.glob('*.txt'):
            try:
                logger.info(f"\nProcessing: {text_file.name}")
                result = processor.process_pipeline(
                    input_file=str(text_file),
                    output_dir=str(processed_data_dir),
                    dataset_name=text_file.stem
                )
                logger.info(f"Result: {result}")
            except Exception as e:
                logger.error(f"Failed to process {text_file.name}: {str(e)}")
    else:
        logger.warning(f"Raw data directory not found: {raw_data_dir}")
