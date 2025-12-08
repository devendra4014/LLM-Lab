"""
This module provides efficient data loading, batching, validation, and utility functions
for training language models. It bridges processed data with training pipelines.
"""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterator, Union
from collections import Counter

import numpy as np

from .vocabulary import VocabularyManager


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataConfig:
    """
    Configuration class for data processing and loading.
    
    Encapsulates all data-related configurations to ensure consistency
    between processing and training stages.
    """
    
    # Default configuration
    DEFAULTS = {
        'chunk_size': 1000,
        'seq_length': 128,
        'validation_split': 0.1,
        'lowercase': False,
        'encoding': 'utf-8',
        'remove_special_chars': False,
        'vocab_size': None,
        'batch_size': 32,
        'shuffle': True,
        'drop_last': False,
        'seed': 42,
    }
    
    def __init__(self, **kwargs):
        """
        Initialize DataConfig with custom or default values.
        
        Args:
            **kwargs: Configuration parameters to override defaults
        """
        self.config = {**self.DEFAULTS}
        self.config.update(kwargs)
        
        logger.info(f"DataConfig initialized with {len(kwargs)} custom parameters")
        logger.debug(f"Configuration: {self.config}")
    
    def update(self, updates: Dict) -> None:
        """
        Update configuration.
        
        Args:
            updates (Dict): Dictionary of updates
        """
        self.config.update(updates)
        logger.info(f"Configuration updated")
    
    def get(self, key: str, default=None):
        """
        Get configuration value.
        
        Args:
            key (str): Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return self.config.get(key, default)
    
    def __getitem__(self, key: str):
        """Access config via bracket notation."""
        return self.config[key]
    
    def __setitem__(self, key: str, value):
        """Set config via bracket notation."""
        self.config[key] = value
    
    def to_dict(self) -> Dict:
        """
        Get configuration as dictionary.
        
        Returns:
            Dict: Configuration dictionary
        """
        return self.config.copy()
    
    @staticmethod
    def from_metadata(metadata: Dict) -> 'DataConfig':
        """
        Create DataConfig from metadata dictionary.
        
        Args:
            metadata (Dict): Metadata from processed data
            
        Returns:
            DataConfig: Configuration instance
        """
        config_from_metadata = metadata.get('config', {})
        return DataConfig(**config_from_metadata)


class DatasetLoader:
    """
    Loads processed datasets and vocabulary from disk.
    
    Handles loading of encoded sequences, vocabulary mappings, and metadata.
    Provides convenient access to dataset information.
    """
    
    def __init__(self, data_path: str, vocab_path: str, metadata_path: Optional[str] = None,
                 splits_path: Optional[str] = None):
        """
        Initialize DatasetLoader.
        
        Args:
            data_path (str): Path to encoded data (.npy file)
            vocab_path (str): Path to vocabulary file (.pkl file)
            metadata_path (str, optional): Path to metadata file (.json file)
            splits_path (str, optional): Path to pre-split data (.pkl file)
        """
        self.data_path = data_path
        self.vocab_path = vocab_path
        self.metadata_path = metadata_path
        self.splits_path = splits_path
        
        self.data = None
        self.vocab = None
        self.metadata = None
        self.splits = None
        
        logger.info(f"DatasetLoader initialized")
        logger.debug(f"Data path: {data_path}")
        logger.debug(f"Vocab path: {vocab_path}")
        if splits_path:
            logger.debug(f"Splits path: {splits_path}")
    
    def load_data(self) -> np.ndarray:
        """
        Load encoded data from disk.
        
        Returns:
            np.ndarray: Encoded sequences
            
        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If data is corrupted
        """
        try:
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
            self.data = np.load(self.data_path)
            logger.info(f"Data loaded successfully - Shape: {self.data.shape}, "
                       f"Dtype: {self.data.dtype}")
            
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def load_vocab(self) -> Dict:
        """
        Load vocabulary from disk.
        
        Returns:
            Dict: Dictionary with 'char_to_idx' and 'idx_to_char' mappings
            
        Raises:
            FileNotFoundError: If vocabulary file doesn't exist
        """
        try:
            if not os.path.exists(self.vocab_path):
                raise FileNotFoundError(f"Vocabulary file not found: {self.vocab_path}")
            
            with open(self.vocab_path, 'rb') as f:
                self.vocab = pickle.load(f)
            
            logger.info(f"Vocabulary loaded - Size: {len(self.vocab.get('char_to_idx', {}))}")
            
            return self.vocab
            
        except Exception as e:
            logger.error(f"Error loading vocabulary: {str(e)}")
            raise
    
    def load_metadata(self) -> Dict:
        """
        Load metadata from disk.
        
        Returns:
            Dict: Metadata dictionary
            
        Raises:
            FileNotFoundError: If metadata file doesn't exist
        """
        try:
            if self.metadata_path is None:
                logger.warning("Metadata path not provided")
                return {}
            
            if not os.path.exists(self.metadata_path):
                logger.warning(f"Metadata file not found: {self.metadata_path}")
                return {}
            
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            logger.info(f"Metadata loaded successfully")
            
            return self.metadata
            
        except Exception as e:
            logger.error(f"Error loading metadata: {str(e)}")
            raise
    
    def load_splits(self) -> Dict:
        """
        Load pre-split train/validation data from disk.
        
        Returns:
            Dict: Dictionary with 'train' and 'validation' splits
                 Format: {
                     'train': {'inputs': np.ndarray, 'targets': np.ndarray},
                     'validation': {'inputs': np.ndarray, 'targets': np.ndarray}
                 }
            
        Raises:
            FileNotFoundError: If splits file doesn't exist
            ValueError: If splits not provided during initialization
        """
        try:
            if self.splits_path is None:
                raise ValueError("Splits path not provided during initialization")
            
            if not os.path.exists(self.splits_path):
                raise FileNotFoundError(f"Splits file not found: {self.splits_path}")
            
            with open(self.splits_path, 'rb') as f:
                self.splits = pickle.load(f)
            
            logger.info(f"Splits loaded successfully")
            logger.info(f"  Train samples: {len(self.splits['train']['inputs'])}")
            logger.info(f"  Validation samples: {len(self.splits['validation']['inputs'])}")
            
            return self.splits
            
        except Exception as e:
            logger.error(f"Error loading splits: {str(e)}")
            raise
    
    def get_train_val_data(self) -> Tuple[Tuple[np.ndarray, np.ndarray], 
                                          Tuple[np.ndarray, np.ndarray]]:
        """
        Get pre-split train and validation data.
        
        Returns:
            Tuple: ((train_inputs, train_targets), (val_inputs, val_targets))
            
        Raises:
            ValueError: If splits not loaded
        """
        if self.splits is None:
            self.load_splits()
        
        train_inputs = self.splits['train']['inputs']
        train_targets = self.splits['train']['targets']
        val_inputs = self.splits['validation']['inputs']
        val_targets = self.splits['validation']['targets']
        
        logger.info(f"Train/val data retrieved - Train: {train_inputs.shape}, Val: {val_inputs.shape}")
        
        return (train_inputs, train_targets), (val_inputs, val_targets)
    
    def get_vocab_size(self) -> int:
        """
        Get vocabulary size.
        
        Returns:
            int: Number of unique tokens in vocabulary
        """
        if self.vocab is None:
            self.load_vocab()
        
        return len(self.vocab.get('char_to_idx', {}))
    
    def get_dataset_size(self) -> int:
        """
        Get total number of samples in dataset.
        
        Returns:
            int: Number of samples
        """
        if self.data is None:
            self.load_data()
        
        return len(self.data)
    
    def get_info(self) -> Dict:
        """
        Get comprehensive dataset information.
        
        Returns:
            Dict: Dictionary with dataset information
        """
        if self.data is None:
            self.load_data()
        if self.vocab is None:
            self.load_vocab()
        
        info = {
            'data_shape': self.data.shape,
            'data_dtype': str(self.data.dtype),
            'vocab_size': len(self.vocab.get('char_to_idx', {})),
            'total_samples': len(self.data),
            'memory_usage_mb': self.data.nbytes / (1024 * 1024)
        }
        
        logger.info(f"Dataset info: {info}")
        return info


class BatchGenerator:
    """
    Generates mini-batches from data for training.
    
    Supports shuffling, efficient iteration, and epoch-based training loops.
    """
    
    def __init__(self, inputs: np.ndarray, targets: np.ndarray,
                 batch_size: int = 32, shuffle: bool = True,
                 drop_last: bool = False, seed: Optional[int] = None):
        """
        Initialize BatchGenerator.
        
        Args:
            inputs (np.ndarray): Input sequences
            targets (np.ndarray): Target sequences
            batch_size (int): Mini-batch size. Default: 32
            shuffle (bool): Shuffle batches each epoch. Default: True
            drop_last (bool): Drop incomplete final batch. Default: False
            seed (int, optional): Random seed for reproducibility
        """
        assert len(inputs) == len(targets), "Inputs and targets must have same length"
        
        self.inputs = inputs
        self.targets = targets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
        
        self.indices = np.arange(len(inputs))
        self.current_epoch = 0
        
        logger.info(f"BatchGenerator initialized - Batch size: {batch_size}, "
                   f"Total samples: {len(inputs)}, Shuffle: {shuffle}")
    
    def __len__(self) -> int:
        """Get number of batches per epoch."""
        if self.drop_last:
            return len(self.inputs) // self.batch_size
        else:
            return (len(self.inputs) + self.batch_size - 1) // self.batch_size
    
    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Iterate over batches."""
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        for batch_idx in range(len(self)):
            start_idx = batch_idx * self.batch_size
            end_idx = start_idx + self.batch_size
            
            # Skip last batch if drop_last is True and it's incomplete
            if self.drop_last and end_idx > len(self.inputs):
                break
            
            batch_indices = self.indices[start_idx:end_idx]
            yield (self.inputs[batch_indices], self.targets[batch_indices])
        
        self.current_epoch += 1
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get specific batch by index.
        
        Args:
            idx (int): Batch index
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Batch (inputs, targets)
        """
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.inputs))
        
        batch_indices = self.indices[start_idx:end_idx]
        return (self.inputs[batch_indices], self.targets[batch_indices])
    
    def reset(self) -> None:
        """Reset generator state."""
        self.indices = np.arange(len(self.inputs))
        self.current_epoch = 0
        logger.info("BatchGenerator reset")
    
    def get_batches(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Get all batches as a list.
        
        Returns:
            List[Tuple]: List of (inputs, targets) batches
        """
        batches = []
        for batch in self:
            batches.append(batch)
        return batches


class DataSplitter:
    """
    Utilities for splitting datasets into train/val/test sets.
    """
    
    @staticmethod
    def train_val_split(data: np.ndarray, targets: np.ndarray,
                       split_ratio: float = 0.8,
                       seed: Optional[int] = None) -> Tuple[Tuple, Tuple]:
        """
        Split data into training and validation sets.
        
        Args:
            data (np.ndarray): Input data
            targets (np.ndarray): Target data
            split_ratio (float): Ratio for training set. Default: 0.8
            seed (int, optional): Random seed for reproducibility
            
        Returns:
            Tuple[Tuple, Tuple]: ((train_data, train_targets), (val_data, val_targets))
        """
        try:
            if seed is not None:
                np.random.seed(seed)
            
            assert len(data) == len(targets), "Data and targets must have same length"
            assert 0 < split_ratio < 1, "Split ratio must be between 0 and 1"
            
            indices = np.arange(len(data))
            np.random.shuffle(indices)
            
            split_idx = int(len(data) * split_ratio)
            train_indices = indices[:split_idx]
            val_indices = indices[split_idx:]
            
            logger.info(f"Train/Val split - Train: {len(train_indices)}, "
                       f"Val: {len(val_indices)}")
            
            return (
                (data[train_indices], targets[train_indices]),
                (data[val_indices], targets[val_indices])
            )
            
        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            raise
    
    @staticmethod
    def train_val_test_split(data: np.ndarray, targets: np.ndarray,
                            ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
                            seed: Optional[int] = None) -> Tuple[Tuple, Tuple, Tuple]:
        """
        Split data into training, validation, and test sets.
        
        Args:
            data (np.ndarray): Input data
            targets (np.ndarray): Target data
            ratios (Tuple[float, float, float]): (train, val, test) ratios. Default: (0.7, 0.15, 0.15)
            seed (int, optional): Random seed for reproducibility
            
        Returns:
            Tuple[Tuple, Tuple, Tuple]: ((train_data, train_targets), 
                                         (val_data, val_targets),
                                         (test_data, test_targets))
        """
        try:
            if seed is not None:
                np.random.seed(seed)
            
            train_ratio, val_ratio, test_ratio = ratios
            assert abs(sum(ratios) - 1.0) < 1e-6, "Ratios must sum to 1.0"
            assert len(data) == len(targets), "Data and targets must have same length"
            
            indices = np.arange(len(data))
            np.random.shuffle(indices)
            
            train_idx = int(len(data) * train_ratio)
            val_idx = train_idx + int(len(data) * val_ratio)
            
            train_indices = indices[:train_idx]
            val_indices = indices[train_idx:val_idx]
            test_indices = indices[val_idx:]
            
            logger.info(f"Train/Val/Test split - Train: {len(train_indices)}, "
                       f"Val: {len(val_indices)}, Test: {len(test_indices)}")
            
            return (
                (data[train_indices], targets[train_indices]),
                (data[val_indices], targets[val_indices]),
                (data[test_indices], targets[test_indices])
            )
            
        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            raise
    
    @staticmethod
    def stratified_split(data: np.ndarray, labels: np.ndarray,
                        split_ratio: float = 0.8,
                        seed: Optional[int] = None) -> Tuple[Tuple, Tuple]:
        """
        Split data maintaining label distribution (stratified split).
        
        Args:
            data (np.ndarray): Input data
            labels (np.ndarray): Label/target data
            split_ratio (float): Training set ratio. Default: 0.8
            seed (int, optional): Random seed for reproducibility
            
        Returns:
            Tuple[Tuple, Tuple]: ((train_data, train_labels), (val_data, val_labels))
        """
        try:
            if seed is not None:
                np.random.seed(seed)
            
            train_indices = []
            val_indices = []
            
            # Get unique labels
            unique_labels = np.unique(labels)
            
            # Split each class separately
            for label in unique_labels:
                label_indices = np.where(labels == label)[0]
                split_idx = int(len(label_indices) * split_ratio)
                
                np.random.shuffle(label_indices)
                train_indices.extend(label_indices[:split_idx])
                val_indices.extend(label_indices[split_idx:])
            
            train_indices = np.array(train_indices)
            val_indices = np.array(val_indices)
            
            logger.info(f"Stratified split - Train: {len(train_indices)}, "
                       f"Val: {len(val_indices)}")
            
            return (
                (data[train_indices], labels[train_indices]),
                (data[val_indices], labels[val_indices])
            )
            
        except Exception as e:
            logger.error(f"Error stratified split: {str(e)}")
            raise


class DataNormalizer:
    """
    Normalizes/scales numerical data.
    
    Supports fitting on training data and transforming new data.
    """
    
    def __init__(self, method: str = 'minmax'):
        """
        Initialize DataNormalizer.
        
        Args:
            method (str): Normalization method ('minmax' or 'zscore'). Default: 'minmax'
        """
        self.method = method
        self.min_val = None
        self.max_val = None
        self.mean_val = None
        self.std_val = None
        self.fitted = False
        
        logger.info(f"DataNormalizer initialized with method: {method}")
    
    def fit(self, data: np.ndarray) -> None:
        """
        Fit normalizer on training data.
        
        Args:
            data (np.ndarray): Training data
        """
        try:
            if self.method == 'minmax':
                self.min_val = np.min(data)
                self.max_val = np.max(data)
                logger.info(f"Fitted MinMax - Min: {self.min_val}, Max: {self.max_val}")
            
            elif self.method == 'zscore':
                self.mean_val = np.mean(data)
                self.std_val = np.std(data)
                logger.info(f"Fitted Z-score - Mean: {self.mean_val}, Std: {self.std_val}")
            
            self.fitted = True
            
        except Exception as e:
            logger.error(f"Error fitting normalizer: {str(e)}")
            raise
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted parameters.
        
        Args:
            data (np.ndarray): Data to transform
            
        Returns:
            np.ndarray: Transformed data
        """
        try:
            if not self.fitted:
                raise ValueError("Normalizer not fitted. Call fit() first.")
            
            if self.method == 'minmax':
                transformed = (data - self.min_val) / (self.max_val - self.min_val + 1e-8)
            
            elif self.method == 'zscore':
                transformed = (data - self.mean_val) / (self.std_val + 1e-8)
            
            logger.debug(f"Data transformed - Shape: {transformed.shape}")
            return transformed
            
        except Exception as e:
            logger.error(f"Error transforming data: {str(e)}")
            raise
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Fit and transform data in one step.
        
        Args:
            data (np.ndarray): Data to fit and transform
            
        Returns:
            np.ndarray: Transformed data
        """
        self.fit(data)
        return self.transform(data)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Reverse normalization transformation.
        
        Args:
            data (np.ndarray): Normalized data
            
        Returns:
            np.ndarray: Original scale data
        """
        try:
            if not self.fitted:
                raise ValueError("Normalizer not fitted. Call fit() first.")
            
            if self.method == 'minmax':
                original = data * (self.max_val - self.min_val) + self.min_val
            
            elif self.method == 'zscore':
                original = data * self.std_val + self.mean_val
            
            logger.debug(f"Data inverse transformed - Shape: {original.shape}")
            return original
            
        except Exception as e:
            logger.error(f"Error inverse transforming data: {str(e)}")
            raise


class DataAugmentation:
    """
    Data augmentation utilities for regularization.
    """
    
    @staticmethod
    def add_noise(data: np.ndarray, noise_level: float = 0.01,
                 seed: Optional[int] = None) -> np.ndarray:
        """
        Add Gaussian noise to data.
        
        Args:
            data (np.ndarray): Input data
            noise_level (float): Standard deviation of noise. Default: 0.01
            seed (int, optional): Random seed for reproducibility
            
        Returns:
            np.ndarray: Data with added noise
        """
        try:
            if seed is not None:
                np.random.seed(seed)
            
            noise = np.random.normal(0, noise_level, data.shape)
            augmented = data + noise
            
            logger.debug(f"Noise added - Level: {noise_level}")
            return augmented
            
        except Exception as e:
            logger.error(f"Error adding noise: {str(e)}")
            raise
    
    @staticmethod
    def random_masking(data: np.ndarray, mask_ratio: float = 0.1,
                      mask_value: int = 0,
                      seed: Optional[int] = None) -> np.ndarray:
        """
        Randomly mask positions in sequences.
        
        Args:
            data (np.ndarray): Input sequences
            mask_ratio (float): Fraction to mask. Default: 0.1
            mask_value (int): Value to use for masking. Default: 0
            seed (int, optional): Random seed for reproducibility
            
        Returns:
            np.ndarray: Data with masked positions
        """
        try:
            if seed is not None:
                np.random.seed(seed)
            
            augmented = data.copy()
            mask = np.random.rand(*data.shape) < mask_ratio
            augmented[mask] = mask_value
            
            logger.debug(f"Random masking applied - Ratio: {mask_ratio}")
            return augmented
            
        except Exception as e:
            logger.error(f"Error applying masking: {str(e)}")
            raise
    
    @staticmethod
    def sequence_shuffling(data: np.ndarray, shuffle_fraction: float = 0.1,
                          seed: Optional[int] = None) -> np.ndarray:
        """
        Shuffle random subsequences.
        
        Args:
            data (np.ndarray): Input sequences
            shuffle_fraction (float): Fraction to shuffle. Default: 0.1
            seed (int, optional): Random seed for reproducibility
            
        Returns:
            np.ndarray: Data with shuffled subsequences
        """
        try:
            if seed is not None:
                np.random.seed(seed)
            
            augmented = data.copy()
            seq_length = data.shape[-1] if len(data.shape) > 1 else len(data)
            shuffle_len = max(1, int(seq_length * shuffle_fraction))
            
            for i in range(len(augmented)):
                start_idx = np.random.randint(0, seq_length - shuffle_len)
                indices = np.arange(start_idx, start_idx + shuffle_len)
                np.random.shuffle(indices)
                augmented[i][indices] = data[i][np.sort(indices)]
            
            logger.debug(f"Sequence shuffling applied - Fraction: {shuffle_fraction}")
            return augmented
            
        except Exception as e:
            logger.error(f"Error shuffling sequences: {str(e)}")
            raise


def validate_dataset(inputs: np.ndarray, targets: np.ndarray) -> Dict:
    """
    Validate dataset quality and consistency.
    
    Args:
        inputs (np.ndarray): Input sequences
        targets (np.ndarray): Target sequences
        
    Returns:
        Dict: Validation report
    """
    try:
        report = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check lengths match
        if len(inputs) != len(targets):
            report['errors'].append(f"Length mismatch: inputs={len(inputs)}, targets={len(targets)}")
            report['valid'] = False
        
        # Check dtypes
        if inputs.dtype != targets.dtype:
            report['warnings'].append(f"Dtype mismatch: inputs={inputs.dtype}, targets={targets.dtype}")
        
        # Check for NaN values
        if np.any(np.isnan(inputs)):
            report['errors'].append("NaN values found in inputs")
            report['valid'] = False
        if np.any(np.isnan(targets)):
            report['errors'].append("NaN values found in targets")
            report['valid'] = False
        
        # Check for infinite values
        if np.any(np.isinf(inputs)):
            report['errors'].append("Infinite values found in inputs")
            report['valid'] = False
        if np.any(np.isinf(targets)):
            report['errors'].append("Infinite values found in targets")
            report['valid'] = False
        
        # Check value ranges
        if len(inputs) > 0:
            input_min, input_max = np.min(inputs), np.max(inputs)
            target_min, target_max = np.min(targets), np.max(targets)
            
            report['input_range'] = (float(input_min), float(input_max))
            report['target_range'] = (float(target_min), float(target_max))
        
        logger.info(f"Dataset validation - Valid: {report['valid']}, "
                   f"Errors: {len(report['errors'])}, Warnings: {len(report['warnings'])}")
        
        return report
        
    except Exception as e:
        logger.error(f"Error validating dataset: {str(e)}")
        raise


def compute_statistics(data: np.ndarray) -> Dict:
    """
    Compute detailed statistics on data.
    
    Args:
        data (np.ndarray): Input data
        
    Returns:
        Dict: Statistics dictionary
    """
    try:
        stats = {
            'shape': data.shape,
            'dtype': str(data.dtype),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'mean': float(np.mean(data)),
            'median': float(np.median(data)),
            'std': float(np.std(data)),
            'variance': float(np.var(data)),
            'unique_values': int(len(np.unique(data))),
        }
        
        logger.info(f"Statistics computed - Mean: {stats['mean']:.4f}, "
                   f"Std: {stats['std']:.4f}, Unique: {stats['unique_values']}")
        
        return stats
        
    except Exception as e:
        logger.error(f"Error computing statistics: {str(e)}")
        raise


def compute_sequence_length_distribution(data: np.ndarray) -> Dict:
    """
    Analyze sequence length distribution.
    
    Args:
        data (np.ndarray): Sequence data (1D or 2D)
        
    Returns:
        Dict: Distribution statistics
    """
    try:
        if len(data.shape) == 1:
            lengths = [len(data)]
        else:
            lengths = [len(seq) for seq in data]
        
        distribution = {
            'min_length': int(np.min(lengths)),
            'max_length': int(np.max(lengths)),
            'mean_length': float(np.mean(lengths)),
            'median_length': float(np.median(lengths)),
            'std_length': float(np.std(lengths)),
        }
        
        logger.info(f"Sequence length distribution - Mean: {distribution['mean_length']:.2f}, "
                   f"Range: {distribution['min_length']}-{distribution['max_length']}")
        
        return distribution
        
    except Exception as e:
        logger.error(f"Error computing sequence distribution: {str(e)}")
        raise


def detect_data_leakage(train_data: np.ndarray, val_data: np.ndarray,
                       threshold: int = 5) -> Dict:
    """
    Detect potential data leakage between splits.
    
    Args:
        train_data (np.ndarray): Training data
        val_data (np.ndarray): Validation data
        threshold (int): Minimum sequence length to check. Default: 5
        
    Returns:
        Dict: Leakage report
    """
    try:
        report = {
            'has_leakage': False,
            'duplicates_found': 0,
            'warning': None
        }
        
        # Convert to tuples for comparison (if sequences are fixed length)
        if len(train_data.shape) == 1 and len(val_data.shape) == 1:
            train_set = set(train_data)
            val_set = set(val_data)
            overlap = train_set.intersection(val_set)
            
            if len(overlap) > 0:
                report['has_leakage'] = True
                report['duplicates_found'] = len(overlap)
                report['warning'] = f"Found {len(overlap)} duplicate values between splits"
        
        logger.warning(f"Data leakage check - Leakage detected: {report['has_leakage']}")
        
        return report
        
    except Exception as e:
        logger.error(f"Error detecting leakage: {str(e)}")
        raise


def encode_text(text: str, vocab: Dict[str, Dict]) -> np.ndarray:
    """
    Encode text using vocabulary.
    
    Args:
        text (str): Text to encode
        vocab (Dict): Vocabulary in format:
                     {'char_to_idx': {...}, 'idx_to_char': {...}}
        
    Returns:
        np.ndarray: Encoded sequence as integer array
    """
    try:
        if not isinstance(vocab, dict) or 'char_to_idx' not in vocab:
            raise ValueError("Vocabulary must be dict with 'char_to_idx' key")
        
        char_to_idx = vocab['char_to_idx']
        encoded = []
        
        for char in text:
            if char in char_to_idx:
                encoded.append(char_to_idx[char])
            else:
                logger.warning(f"Unknown character: {repr(char)}")
        
        result = np.array(encoded, dtype=np.int32)
        logger.info(f"Text encoded successfully - Length: {len(encoded)}, Shape: {result.shape}")
        return result
        
    except Exception as e:
        logger.error(f"Error encoding text: {str(e)}")
        raise


def decode_sequence(indices: np.ndarray, vocab: Dict[str, Dict]) -> str:
    """
    Decode sequence using vocabulary.
    
    Args:
        indices (np.ndarray): Encoded sequence
        vocab (Dict): Vocabulary in format:
                     {'char_to_idx': {...}, 'idx_to_char': {...}}
        
    Returns:
        str: Decoded text
    """
    try:
        if not isinstance(vocab, dict) or 'idx_to_char' not in vocab:
            raise ValueError("Vocabulary must be dict with 'idx_to_char' key")
        
        idx_to_char = vocab['idx_to_char']
        decoded = []
        
        # Convert indices to list if numpy array
        if isinstance(indices, np.ndarray):
            indices = indices.tolist()
        
        for idx in indices:
            idx_int = int(idx)  # Ensure integer
            if idx_int in idx_to_char:
                decoded.append(idx_to_char[idx_int])
            else:
                logger.warning(f"Invalid index: {idx}")
        
        text = ''.join(decoded)
        logger.info(f"Sequence decoded - Length: {len(text)}")
        return text
        
    except Exception as e:
        logger.error(f"Error decoding sequence: {str(e)}")
        raise


def get_vocab_info(vocab: Dict[str, Dict]) -> Dict:
    """
    Get vocabulary information and statistics.
    
    Args:
        vocab (Dict): Vocabulary in format:
                     {'char_to_idx': {...}, 'idx_to_char': {...}}
        
    Returns:
        Dict: Vocabulary statistics
    """
    try:
        if not isinstance(vocab, dict) or 'char_to_idx' not in vocab:
            raise ValueError("Vocabulary must be dict with 'char_to_idx' key")
        
        char_to_idx = vocab['char_to_idx']
        
        info = {
            'vocab_size': len(char_to_idx),
            'sample_chars': list(char_to_idx.keys())[:20],
            'min_idx': min(char_to_idx.values()) if char_to_idx else None,
            'max_idx': max(char_to_idx.values()) if char_to_idx else None,
        }
        
        logger.info(f"Vocabulary info - Size: {info['vocab_size']}")
        
        return info
        
    except Exception as e:
        logger.error(f"Error getting vocab info: {str(e)}")
        raise


# Example usage
if __name__ == "__main__":
    logger.info("Dataset utilities module loaded successfully")
    
    # Example: Create dummy data
    dummy_inputs = np.random.randint(0, 100, (1000, 128))
    dummy_targets = np.random.randint(0, 100, 1000)
    
    # Test validation
    validation = validate_dataset(dummy_inputs, dummy_targets)
    logger.info(f"Validation result: {validation}")
    
    # Test statistics
    stats = compute_statistics(dummy_inputs)
    logger.info(f"Statistics: {stats}")
    
    # Test batch generator
    batch_gen = BatchGenerator(dummy_inputs, dummy_targets, batch_size=32)
    logger.info(f"Total batches: {len(batch_gen)}")
    
    # Test train/val split
    (train_x, train_y), (val_x, val_y) = DataSplitter.train_val_split(
        dummy_inputs, dummy_targets
    )
    logger.info(f"Train shape: {train_x.shape}, Val shape: {val_x.shape}")
