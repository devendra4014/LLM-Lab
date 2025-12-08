# BigramLanguageModel

This document describes the `BigramLanguageModel` implemented in `src/llm_book/models/bigram.py`.
It is a simple, educational first-order Markov model that counts adjacent token pairs and exposes
probabilities, sampling, and persistence.

## Key API

- `BigramLanguageModel(vocab_size, smoothing=1e-6)`
  - Create a new model. `vocab_size` is required.

- `fit(tokens)`
  - Fit the model from a sequence of integer token ids (1-D array-like).
  - Builds pair counts and computes conditional probabilities with additive smoothing.

- `predict_next_distribution(token_id)`
  - Returns a 1-D NumPy array of probabilities for `P(next | token_id)`.

- `log_prob_sequence(tokens)`
  - Returns the natural log probability of the sequence under the model.

- `perplexity(tokens)`
  - Returns perplexity for the sequence: `exp(-log P / N)` where `N = len(tokens) - 1`.

- `sample(start_token, length, random_state=None)`
  - Simple ancestral sampling from the conditional distributions.

- `sample_top_k(start_token, length, k=10, temperature=1.0, random_state=None)`
  - Top-k sampling at each step. Temperature controls randomness.

- `sample_top_p(start_token, length, p=0.9, temperature=1.0, random_state=None)`
  - Nucleus (top-p) sampling at each step.

- `save(path)` / `load(path)`
  - Persist the model to disk as a JSON metadata file and an `.npz` archive containing arrays.

## Examples

Create and fit a model:

```python
from src.llm_book.models.bigram import BigramLanguageModel
import numpy as np

model = BigramLanguageModel(vocab_size=100)
text_tokens = np.array([0,1,2,3,2,3,4,5,2,3], dtype=np.int64)
model.fit(text_tokens)
```

Compute perplexity:

```python
ppl = model.perplexity(text_tokens)
print(f"Perplexity: {ppl:.3f}")
```

Top-k sampling:

```python
from numpy.random import default_rng
rng = default_rng(42)
seq = model.sample_top_k(start_token=2, length=12, k=5, temperature=0.8, random_state=rng)
print(seq)
```

Save and load:

```python
model.save("models/my_bigram")
loaded = BigramLanguageModel.load("models/my_bigram")
```

## Notes & Tips

- The model assumes token ids are integers in range `[0, vocab_size)`.
- Use `smoothing` to avoid zero-probability transitions; default is `1e-6`.
- `perplexity` returns `inf` if any transition has zero probability under the model.
- `sample_top_k` and `sample_top_p` apply temperature by converting probabilities to logits
  via `log(p)` and dividing by the temperature; small numerical adjustments are made for stability.

## Quick test command

Run the focused bigram tests:

```powershell
python -m pytest tests/test_bigram.py -q
```
