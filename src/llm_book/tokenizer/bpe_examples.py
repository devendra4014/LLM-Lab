"""
BPE Tokenizer Integration Examples

Practical examples showing how to use the BPE tokenizer with the data pipeline.
"""

import logging
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llm_book.tokenizer.bpe import BPETokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_basic_usage():
    """Example 1: Basic encode/decode workflow."""
    logger.info("=" * 60)
    logger.info("Example 1: Basic Encode/Decode")
    logger.info("=" * 60)
    
    # Training text
    training_text = (
        "The quick brown fox jumps over the lazy dog. "
        "Python is a powerful programming language. "
        "Machine learning requires data preprocessing."
    )
    
    # Initialize and train
    tokenizer = BPETokenizer(vocab_size=500)
    tokenizer.train(training_text)
    
    # Encode
    text = "Python is powerful!"
    encoded = tokenizer.encode(text)
    print(f"\nOriginal: '{text}'")
    print(f"Encoded: {encoded}")
    print(f"Token count: {len(encoded)}")
    
    # Decode
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: '{decoded}'")
    print(f"Match: {decoded == text}")
    
    # Vocabulary info
    info = tokenizer.get_vocab_info()
    print(f"\nTokenizer Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")


def example_2_compression_analysis():
    """Example 2: Analyze compression ratio."""
    logger.info("=" * 60)
    logger.info("Example 2: Compression Analysis")
    logger.info("=" * 60)
    
    training_text = """
    The field of machine learning has experienced exponential growth over the past decade.
    Deep neural networks have revolutionized computer vision, natural language processing,
    and reinforcement learning. Transformer architectures like BERT and GPT have achieved
    state-of-the-art results across numerous benchmarks. The scalability and efficiency
    of large language models continue to improve with new techniques and optimizations.
    """ * 10
    
    tokenizer = BPETokenizer(vocab_size=512)
    tokenizer.train(training_text)
    
    # Test on different texts
    test_texts = [
        "Machine learning",
        "Deep neural networks",
        "Transformer architectures",
        training_text[:100]
    ]
    
    print("\nCompression Analysis:")
    for test_text in test_texts:
        encoded = tokenizer.encode(test_text)
        # Character count vs token count
        char_count = len(test_text)
        token_count = len(encoded)
        compression = (1 - token_count / char_count) * 100
        
        print(f"\nText: '{test_text[:50]}...'")
        print(f"  Characters: {char_count}")
        print(f"  Tokens: {token_count}")
        print(f"  Compression: {compression:.1f}%")


def example_3_persistence():
    """Example 3: Save and load tokenizer."""
    logger.info("=" * 60)
    logger.info("Example 3: Persistence (Save/Load)")
    logger.info("=" * 60)
    
    import tempfile
    
    training_text = "hello world hello python world python"
    
    # Train and save
    print("\n1. Training tokenizer...")
    tokenizer1 = BPETokenizer(vocab_size=300)
    tokenizer1.train(training_text)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save
        print(f"2. Saving to {tmpdir}...")
        tokenizer1.save(tmpdir)
        
        files = list(Path(tmpdir).glob('*'))
        print(f"   Files created: {[f.name for f in files]}")
        
        # Load
        print("3. Loading tokenizer...")
        tokenizer2 = BPETokenizer.load(tmpdir)
        
        # Verify
        print("4. Verifying round-trip...")
        test_text = "hello python"
        encoded1 = tokenizer1.encode(test_text)
        encoded2 = tokenizer2.encode(test_text)
        
        print(f"\n   Original encoding:  {encoded1}")
        print(f"   Loaded encoding:    {encoded2}")
        print(f"   Match: {np.array_equal(encoded1, encoded2)}")


def example_4_special_cases():
    """Example 4: Handling special cases."""
    logger.info("=" * 60)
    logger.info("Example 4: Special Cases & Edge Cases")
    logger.info("=" * 60)
    
    training_text = "Hello, world! 123 café naïve" * 5
    
    tokenizer = BPETokenizer(vocab_size=512)
    tokenizer.train(training_text)
    
    # Case 1: Empty string
    print("\n1. Empty string:")
    encoded = tokenizer.encode("")
    print(f"   Encoded: {encoded}")
    print(f"   Length: {len(encoded)}")
    
    # Case 2: Special characters
    print("\n2. Special characters:")
    text = "Email: test@example.com (555-1234)"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    print(f"   Original: '{text}'")
    print(f"   Decoded:  '{decoded}'")
    print(f"   Match: {decoded == text}")
    
    # Case 3: Numbers
    print("\n3. Numbers:")
    text = "Cost: $99.99 or €85.50"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    print(f"   Original: '{text}'")
    print(f"   Decoded:  '{decoded}'")
    print(f"   Match: {decoded == text}")
    
    # Case 4: Unicode
    print("\n4. Unicode characters:")
    text = "Hello 世界 مرحبا мир"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    print(f"   Original: '{text}'")
    print(f"   Decoded:  '{decoded}'")
    # Note: May not match due to character-to-bytes encoding


def example_5_vocab_comparison():
    """Example 5: Compare different vocabulary sizes."""
    logger.info("=" * 60)
    logger.info("Example 5: Vocabulary Size Comparison")
    logger.info("=" * 60)
    
    training_text = (
        "The quick brown fox jumps over the lazy dog. " * 20
    )
    
    vocab_sizes = [256, 512, 1024, 2048]
    test_text = "The quick brown fox jumps"
    
    print(f"\nTest text: '{test_text}'")
    print(f"Original length: {len(test_text)} characters\n")
    
    results = []
    for vocab_size in vocab_sizes:
        tokenizer = BPETokenizer(vocab_size=vocab_size)
        tokenizer.train(training_text)
        
        encoded = tokenizer.encode(test_text)
        info = tokenizer.get_vocab_info()
        
        compression = (1 - len(encoded) / len(test_text)) * 100
        
        results.append({
            'vocab_size': vocab_size,
            'actual_vocab': info['vocab_size'],
            'merges': info['num_merges'],
            'tokens': len(encoded),
            'compression': compression
        })
    
    # Print comparison table
    print(f"{'Target Vocab':<15} {'Actual Vocab':<15} {'Merges':<10} {'Tokens':<10} {'Compression':<15}")
    print("-" * 65)
    for r in results:
        print(f"{r['vocab_size']:<15} {r['actual_vocab']:<15} {r['merges']:<10} "
              f"{r['tokens']:<10} {r['compression']:.1f}%")


def example_6_file_io():
    """Example 6: File-based encoding/decoding."""
    logger.info("=" * 60)
    logger.info("Example 6: File I/O Operations")
    logger.info("=" * 60)
    
    import tempfile
    
    training_text = "Python programming is fun and powerful"
    test_text = "Python is powerful"
    
    tokenizer = BPETokenizer(vocab_size=300)
    tokenizer.train(training_text)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Encode to file
        token_file = Path(tmpdir) / 'tokens.npy'
        print(f"\n1. Encoding to file: {token_file}")
        tokenizer.encode_text_to_file(test_text, str(token_file))
        
        # Check file
        file_size = token_file.stat().st_size
        print(f"   File size: {file_size} bytes")
        
        # Decode from file
        print("2. Decoding from file...")
        decoded_text = tokenizer.decode_from_file(str(token_file))
        
        print(f"\n   Original text: '{test_text}'")
        print(f"   Decoded text:  '{decoded_text}'")
        print(f"   Match: {decoded_text == test_text}")


def example_7_batch_processing():
    """Example 7: Batch processing multiple texts."""
    logger.info("=" * 60)
    logger.info("Example 7: Batch Processing")
    logger.info("=" * 60)
    
    training_text = "The quick brown fox jumps over the lazy dog" * 10
    
    tokenizer = BPETokenizer(vocab_size=500)
    tokenizer.train(training_text)
    
    # Multiple texts
    texts = [
        "The quick fox",
        "brown dog",
        "jumps over"
    ]
    
    print("\nBatch Processing:")
    encodings = []
    for i, text in enumerate(texts):
        encoded = tokenizer.encode(text)
        encodings.append(encoded)
        print(f"\n{i+1}. Text: '{text}'")
        print(f"   Encoded: {encoded} (length: {len(encoded)})")
    
    # Could pad and stack for batch processing
    max_len = max(len(e) for e in encodings)
    print(f"\nMax sequence length: {max_len}")
    
    # Pad to same length
    padded = []
    for enc in encodings:
        padded_enc = np.pad(enc, (0, max_len - len(enc)), 'constant', constant_values=0)
        padded.append(padded_enc)
    
    batch = np.stack(padded)
    print(f"Batch shape: {batch.shape}")
    print(f"Batch:\n{batch}")


if __name__ == '__main__':
    # Run all examples
    example_1_basic_usage()
    example_2_compression_analysis()
    example_3_persistence()
    example_4_special_cases()
    example_5_vocab_comparison()
    example_6_file_io()
    example_7_batch_processing()
    
    logger.info("\n" + "=" * 60)
    logger.info("All examples completed successfully!")
    logger.info("=" * 60)
