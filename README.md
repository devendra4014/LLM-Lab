# Large Language Models: A Learning Project

A comprehensive personal learning repository for understanding Large Language Models (LLMs) from first principles. This project implements fundamental LLM concepts, including n-gram models, bigram models, character-level GPT, and tokenizer implementations, complemented by detailed documentation and practical training scripts.

## 🎯 Project Overview

This repository documents my journey through the depths of Large Language Models. Rather than using black-box implementations, this project focuses on building language models from scratch to understand the underlying mechanics of transformers, attention mechanisms, and modern LLM architectures.

### Key Features

- **Model Implementations**: N-gram models, Bigram models, and character-level GPT architecture
- **Tokenization**: BPE (Byte Pair Encoding) and other tokenizer implementations
- **Training Infrastructure**: End-to-end scripts for training small language models
- **Educational Content**: Detailed notebooks, documentation, and diagrams explaining transformer and attention mechanisms
- **Datasets**: Curated text datasets (Shakespeare, Star Wars, legal text) for experimentation
- **Test Suite**: Comprehensive unit tests for validating model implementations

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip or conda package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/llm-learning.git
   cd llm-learning
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package in development mode**
   ```bash
   pip install -e .
   ```

## 📊 Datasets

The project includes three curated text datasets:

- **shakespeare.txt**: Works of William Shakespeare (~5MB)
- **starwars.txt**: Star Wars movie scripts (~2MB)
- **the-verdict.txt**: Legal text examples (~1MB)

To download additional datasets or update existing ones:

```bash
bash scripts/download_data.sh
```

## 🎯 Implementation Roadmap

This project is under active development. The following components are planned and will be implemented progressively:

### Phase 1: Data Processing
- [ ] **Raw Data Processing**: Build pipeline to process raw text files (`src/llm_book/data/process_raw_data.py`)
- [ ] **Dataset Utilities**: Create dataset loaders and preprocessing tools (`src/llm_book/data/dataset_utils.py`)
- [ ] **Data Validation**: Ensure data quality and format consistency

### Phase 2: Tokenization
- [ ] **BPE Tokenizer**: Implement Byte Pair Encoding (`src/llm_book/tokenizer/bpe.py`)
- [ ] **Tokenizer Utils**: Helper functions for vocabulary building and encoding/decoding

### Phase 3: Training Infrastructure
- [ ] **Training Utilities**: Common loss functions, metrics, and logging (`src/llm_book/training/utils.py`)
- [ ] **Checkpointing**: Model saving and resuming capabilities
- [ ] **Hyperparameter Management**: Configuration handling for experiments

### Phase 4: Foundation Models
- [ ] **N-gram Model**: Static probability-based language model
- [ ] **Bigram Model**: Build and train a bigram model (`src/llm_book/models/bigram.py`)
- [ ] **Bigram Training**: Implement training pipeline (`src/llm_book/training/train_bigram.py`)

### Phase 5: Neural Language Models
- [ ] **Character-Level GPT**: Implement transformer-based GPT (`src/llm_book/models/char_gpt.py`)
- [ ] **GPT Training**: Create training script with optimization (`src/llm_book/training/train_char_gpt.py`)
- [ ] **Attention Mechanisms**: Implement multi-head self-attention

### Phase 6: Testing & Validation
- [ ] **Unit Tests**: Test bigram model (`tests/test_bigram.py`)
- [ ] **GPT Tests**: Validate GPT implementation (`tests/test_char_gpt.py`)
- [ ] **Tokenizer Tests**: Verify tokenization correctness (`tests/test_tokenizer.py`)

### Phase 7: Evaluation & Analysis
- [ ] **Evaluation Scripts**: Perplexity, loss analysis, and text generation (`scripts/evaluate.sh`)
- [ ] **Model Comparison**: Benchmarking different architectures

### Phase 8: Documentation & Learning
- [ ] **Comprehensive Notes**: Create `docs/LLMBook.md` with detailed explanations
- [ ] **Visual Diagrams**: Architecture diagrams and attention visualization in `docs/images/`
- [ ] **Jupyter Notebooks**: Interactive notebooks for experimentation in `notebooks/`
- [ ] **Concept Explanations**: Transformers, attention mechanisms, and training strategies

### Phase 9: Automation
- [ ] **Data Download Script**: Automated dataset fetching (`scripts/download_data.sh`)
- [ ] **Training Orchestration**: Master training script (`scripts/train.sh`)

## 📚 Learning Topics Covered

This project will progressively cover:

- Fundamentals of language modeling and probability distributions
- N-gram and neural language models
- Transformer architecture and self-attention mechanisms
- Subword tokenization strategies (BPE)
- Character-level modeling with neural networks
- Training loops, optimization, and hyperparameter tuning
- Model evaluation metrics and analysis techniques
- Text generation and sampling strategies

## 🛠️ Technologies & Libraries

- **PyTorch**: Deep learning framework
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **scikit-learn**: Machine learning utilities
- **Jupyter**: Interactive notebooks
- **pytest**: Testing framework

## 📝 Notes

- This is a **learning project** designed for educational purposes, not production deployment
- Model sizes are intentionally small to allow training on CPU/limited GPU resources
- Emphasis is placed on understanding concepts rather than achieving state-of-the-art performance

---

Happy Learning! 🚀
