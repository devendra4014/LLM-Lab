# Data Processing & Loading Integration Guide

## Overview

This guide demonstrates the complete data pipeline workflow integrating `process_raw_data.py`, `dataset_utils.py`, and `vocabulary.py`.

---

## Complete Workflow

### Step 1: Configure Data Processing

```python
from llm_book.data.dataset_utils import DataConfig
from llm_book.data.process_raw_data import DataProcessor

# Create configuration
config = DataConfig(
    seq_length=128,
    validation_split=0.1,
    lowercase=False,
    encoding='utf-8',
    batch_size=32,
    seed=42
)

# Initialize processor with configuration
processor = DataProcessor(config.to_dict())
```

### Step 2: Process Raw Text Files

```python
from pathlib import Path

# Define paths
raw_data_dir = Path('data/raw')
processed_data_dir = Path('data/processed')

# Process all text files
for text_file in raw_data_dir.glob('*.txt'):
    print(f"\nProcessing: {text_file.name}")
    
    result = processor.process_pipeline(
        input_file=str(text_file),
        output_dir=str(processed_data_dir),
        dataset_name=text_file.stem
    )
    
    print(f"✓ Dataset: {result['dataset_name']}")
    print(f"  Original text: {result['stats']['text_length']} chars")
    print(f"  Vocabulary: {result['stats']['vocab_size']} tokens")
    print(f"  Sequences: {result['stats']['total_sequences']}")
```

**Output files created:**
- `{dataset_name}_encoded.npy` - Encoded sequences
- `{dataset_name}_vocab.pkl` - Vocabulary mappings
- `{dataset_name}_metadata.json` - Processing metadata & config
- `{dataset_name}_splits.pkl` - Pre-split train/validation data

---

### Step 3: Load Processed Data

```python
from llm_book.data.dataset_utils import DatasetLoader

# Initialize loader with all paths
loader = DatasetLoader(
    data_path='data/processed/shakespeare_encoded.npy',
    vocab_path='data/processed/shakespeare_vocab.pkl',
    metadata_path='data/processed/shakespeare_metadata.json',
    splits_path='data/processed/shakespeare_splits.pkl'
)

# Load all components
encoded_data = loader.load_data()
vocabulary = loader.load_vocab()
metadata = loader.load_metadata()
splits = loader.load_splits()

# Get dataset information
info = loader.get_info()
print(f"Vocabulary size: {info['vocab_size']}")
print(f"Total sequences: {info['total_samples']}")
print(f"Memory usage: {info['memory_usage_mb']:.2f} MB")
```

---

### Step 4: Get Pre-Split Train/Validation Data

```python
# Retrieve pre-split data (created during processing)
(train_inputs, train_targets), (val_inputs, val_targets) = loader.get_train_val_data()

print(f"Training set: {train_inputs.shape}")
print(f"Validation set: {val_inputs.shape}")
```

---

### Step 5: Create Batch Generator

```python
from llm_book.data.dataset_utils import BatchGenerator

# Create batch generator for training
train_batch_gen = BatchGenerator(
    inputs=train_inputs,
    targets=train_targets,
    batch_size=32,
    shuffle=True,
    seed=42
)

# Create batch generator for validation
val_batch_gen = BatchGenerator(
    inputs=val_inputs,
    targets=val_targets,
    batch_size=32,
    shuffle=False
)

# Iterate over batches
for epoch in range(3):
    print(f"\nEpoch {epoch + 1}")
    
    for batch_idx, (batch_inputs, batch_targets) in enumerate(train_batch_gen):
        print(f"  Batch {batch_idx}: inputs {batch_inputs.shape}, targets {batch_targets.shape}")
        # Train model on batch...
```

---

### Step 6: Validate Data Quality

```python
from llm_book.data.dataset_utils import validate_dataset, compute_statistics

# Validate data integrity
validation_report = validate_dataset(train_inputs, train_targets)
print(f"Data valid: {validation_report['valid']}")
print(f"Input range: {validation_report['input_range']}")
print(f"Target range: {validation_report['target_range']}")

# Compute statistics
stats = compute_statistics(train_inputs)
print(f"Mean value: {stats['mean']:.4f}")
print(f"Std deviation: {stats['std']:.4f}")
print(f"Unique tokens: {stats['unique_values']}")
```

---

### Step 7: Use Vocabulary for Encoding/Decoding

```python
from llm_book.data.dataset_utils import encode_text, decode_sequence
import numpy as np

# Encode new text using the learned vocabulary
text_sample = "Hello, world!"
encoded = encode_text(text_sample, vocabulary)
print(f"Encoded: {encoded}")

# Decode sequences back to text
decoded = decode_sequence(encoded, vocabulary)
print(f"Decoded: {decoded}")
```

---

### Step 8: Load Configuration from Metadata

```python
from llm_book.data.dataset_utils import DataConfig

# Recreate config from processing metadata
config = DataConfig.from_metadata(metadata)

# Verify config matches original
print(f"Sequence length: {config['seq_length']}")
print(f"Validation split: {config['validation_split']}")
print(f"Batch size: {config['batch_size']}")
```

---

## Advanced: Custom Data Augmentation

```python
from llm_book.data.dataset_utils import DataAugmentation

# Create augmented training data
augmented_inputs = DataAugmentation.random_masking(
    train_inputs, 
    mask_ratio=0.1
)

augmented_inputs = DataAugmentation.add_noise(
    augmented_inputs,
    noise_level=0.02
)

# Use augmented data for training
batch_gen_augmented = BatchGenerator(augmented_inputs, train_targets, batch_size=32)
```

---

## Key Integration Points

### Data Flow:
```
Raw Text Files
      ↓
process_raw_data.py (DataProcessor)
      ↓
[.npy, .pkl, .json files in data/processed/]
      ↓
dataset_utils.py (DatasetLoader)
      ↓
BatchGenerator → Training Loop
```

### Consistency Guarantees:
- ✅ Vocabulary format standardized: `{'char_to_idx': {...}, 'idx_to_char': {...}}`
- ✅ Configuration persisted in metadata and accessible via `DataConfig.from_metadata()`
- ✅ Data split consistency via pre-computed train/val splits
- ✅ Vocabulary consistency ensured through `vocabulary.py` module
- ✅ Type consistency: encoding returns `np.ndarray`, decoding accepts `np.ndarray`

---

## Troubleshooting

### Issue: "Vocabulary not built"
**Solution**: Ensure `load_splits()` or `load_data()` is called before encoding

```python
loader.load_splits()  # or loader.load_data()
vocab = loader.load_vocab()
```

### Issue: Data shape mismatch
**Solution**: Verify configuration matches between processing and loading:

```python
meta_config = config.to_dict()
print(meta_config['seq_length'])  # Should match batch sequence length
```

### Issue: Unknown characters in encoding
**Solution**: Vocabulary built on training data; new text may have unknown chars:

```python
# encode_text() handles gracefully with warnings
# Use same text preprocessing/cleaning as in DataProcessor
```

---

## Complete Example Script

```python
#!/usr/bin/env python
"""Complete data pipeline example."""

from pathlib import Path
from llm_book.data.dataset_utils import (
    DataConfig, DatasetLoader, BatchGenerator, 
    validate_dataset, compute_statistics
)
from llm_book.data.process_raw_data import DataProcessor

def main():
    # Step 1: Configure
    config = DataConfig(seq_length=128, batch_size=32, seed=42)
    
    # Step 2: Process
    processor = DataProcessor(config.to_dict())
    raw_file = Path('data/raw/shakespeare.txt')
    processed_dir = Path('data/processed')
    
    result = processor.process_pipeline(
        str(raw_file), str(processed_dir), 'shakespeare'
    )
    print(f"✓ Processed {result['dataset_name']}")
    
    # Step 3: Load
    loader = DatasetLoader(
        data_path=str(processed_dir / 'shakespeare_encoded.npy'),
        vocab_path=str(processed_dir / 'shakespeare_vocab.pkl'),
        metadata_path=str(processed_dir / 'shakespeare_metadata.json'),
        splits_path=str(processed_dir / 'shakespeare_splits.pkl')
    )
    
    # Step 4: Get splits
    (train_x, train_y), (val_x, val_y) = loader.get_train_val_data()
    print(f"✓ Loaded splits: train {train_x.shape}, val {val_x.shape}")
    
    # Step 5: Validate
    validation = validate_dataset(train_x, train_y)
    print(f"✓ Validation: {validation['valid']}")
    
    # Step 6: Create batches
    batch_gen = BatchGenerator(train_x, train_y, batch_size=32, shuffle=True)
    print(f"✓ Batch generator ready: {len(batch_gen)} batches")
    
    # Step 7: Get one batch
    first_batch = next(iter(batch_gen))
    print(f"✓ First batch: {first_batch[0].shape}, {first_batch[1].shape}")

if __name__ == '__main__':
    main()
```

---

## Summary

This integration ensures:
1. **Consistency**: Same vocabulary, config, and data formats across all operations
2. **Reproducibility**: Seeds, configs, and splits are persisted
3. **Efficiency**: Pre-computed splits avoid redundant processing
4. **Type Safety**: Standardized array types and vocabulary formats
5. **Debugging**: Comprehensive logging and validation utilities

---

**Last Updated**: December 7, 2025
