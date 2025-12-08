"""Bigram language model (simple, NumPy-based).

This module provides a lightweight bigram language model useful for
education and small-scale experiments. It counts adjacent token pairs
and exposes probabilities, log-probabilities, sampling, and simple
persistence helpers.

The model expects token IDs to be integers in range [0, vocab_size).
"""

from __future__ import annotations

from typing import Optional, Sequence
import numpy as np
import json
from pathlib import Path


class BigramLanguageModel:
	"""A simple bigram (first-order Markov) language model.

	Attributes:
		vocab_size: Size of vocabulary (number of distinct token ids).
		counts: 2D array of shape (vocab_size, vocab_size) with pair counts.
		probs: 2D array of conditional probabilities P(next | current).
	"""

	def __init__(self, vocab_size: int, smoothing: float = 1e-6) -> None:
		if vocab_size <= 0:
			raise ValueError("vocab_size must be a positive integer")
		self.vocab_size = int(vocab_size)
		self.smoothing = float(smoothing)
		self.counts = np.zeros((self.vocab_size, self.vocab_size), dtype=np.float64)
		self.probs: Optional[np.ndarray] = None

	def fit(self, tokens: Sequence[int]) -> None:
		"""Fit the bigram model from a sequence of token ids.

		Args:
			tokens: Iterable of integer token ids (1D sequence).
		"""
		tokens = np.asarray(tokens, dtype=np.int64)
		if tokens.ndim != 1:
			raise ValueError("tokens must be a 1-D sequence of token ids")
		if tokens.size < 2:
			# nothing to learn from fewer than 2 tokens
			self.counts.fill(0)
			self.probs = None
			return

		# Validate tokens are within vocabulary
		if tokens.min() < 0 or tokens.max() >= self.vocab_size:
			raise ValueError("token ids must be in range [0, vocab_size)")

		# Efficient counting of adjacent pairs
		a = tokens[:-1].astype(np.int64)
		b = tokens[1:].astype(np.int64)
		# Reset counts
		self.counts.fill(0)
		for i, j in zip(a, b):
			self.counts[i, j] += 1.0

		# Convert counts to conditional probabilities with smoothing
		row_sums = self.counts.sum(axis=1, keepdims=True)
		# Avoid division by zero with smoothing
		self.probs = (self.counts + self.smoothing) / (row_sums + self.smoothing * self.vocab_size)

	def predict_next_distribution(self, token_id: int) -> np.ndarray:
		"""Return probability distribution over next tokens given current token.

		Args:
			token_id: Current token id.

		Returns:
			1-D NumPy array of length `vocab_size` with probabilities summing to 1.
		"""
		if token_id < 0 or token_id >= self.vocab_size:
			raise ValueError("token_id out of range")
		if self.probs is None:
			# If model not fitted, return uniform distribution
			return np.ones(self.vocab_size, dtype=np.float64) / float(self.vocab_size)
		return self.probs[token_id].copy()

	def log_prob_sequence(self, tokens: Sequence[int]) -> float:
		"""Compute the log-probability (natural log) of a token sequence under the model.

		Returns -inf for sequences containing out-of-vocabulary tokens.
		"""
		tokens = np.asarray(tokens, dtype=np.int64)
		if tokens.ndim != 1 or tokens.size < 2:
			return 0.0
		if tokens.min() < 0 or tokens.max() >= self.vocab_size:
			return -np.inf
		if self.probs is None:
			# uniform model
			return -np.log(self.vocab_size) * (tokens.size - 1)

		logp = 0.0
		for prev, nxt in zip(tokens[:-1], tokens[1:]):
			p = self.probs[prev, nxt]
			if p <= 0.0:
				return -np.inf
			logp += float(np.log(p))
		return logp

	def sample(self, start_token: int, length: int, random_state: Optional[np.random.Generator] = None) -> np.ndarray:
		"""Sample a sequence of token ids from the model.

		Args:
			start_token: Initial token id to start sampling from.
			length: Total length of sequence to produce (including start_token). Must be >=1.
			random_state: Optional NumPy Generator for reproducibility.

		Returns:
			NumPy array of sampled token ids of shape (length,).
		"""
		if length <= 0:
			raise ValueError("length must be >= 1")
		if start_token < 0 or start_token >= self.vocab_size:
			raise ValueError("start_token out of range")
		rng = random_state or np.random.default_rng()
		seq = np.empty(length, dtype=np.int64)
		seq[0] = int(start_token)
		for i in range(1, length):
			dist = self.predict_next_distribution(int(seq[i - 1]))
			seq[i] = int(rng.choice(self.vocab_size, p=dist))
		return seq

	def perplexity(self, tokens: Sequence[int]) -> float:
		"""Compute the perplexity of a token sequence under the model.

		Perplexity is defined as exp(-1/N * log P(tokens)), where N is the
		number of predicted transitions (len(tokens)-1). Returns ``np.inf``
		for sequences containing out-of-vocabulary tokens or zero-probability
		transitions.
		"""
		tokens = np.asarray(tokens, dtype=np.int64)
		if tokens.ndim != 1 or tokens.size < 2:
			return float('nan')
		N = tokens.size - 1
		logp = self.log_prob_sequence(tokens)
		if not np.isfinite(logp):
			return float('inf')
		return float(np.exp(-logp / float(N)))

	def sample_top_k(self, start_token: int, length: int, k: int = 10, temperature: float = 1.0, random_state: Optional[np.random.Generator] = None) -> np.ndarray:
		"""Sample a sequence using top-k sampling at each step.

		Args:
			start_token: initial token id
			length: total sequence length including start
			k: keep top-k candidates
			temperature: sampling temperature (>0, 1.0 is default)
			random_state: optional NumPy Generator
		"""
		if k <= 0:
			raise ValueError("k must be > 0")
		rng = random_state or np.random.default_rng()
		seq = np.empty(length, dtype=np.int64)
		seq[0] = int(start_token)
		for i in range(1, length):
			dist = self.predict_next_distribution(int(seq[i - 1]))
			# apply temperature
			if temperature <= 0.0:
				raise ValueError("temperature must be > 0")
			if temperature != 1.0:
				logits = np.log(dist + 1e-20) / float(temperature)
				exp_logits = np.exp(logits - np.max(logits))
				dist = exp_logits / exp_logits.sum()

			# select top-k indices
			k = min(k, self.vocab_size)
			topk_idx = np.argpartition(-dist, k - 1)[:k]
			topk_probs = dist[topk_idx]
			if topk_probs.sum() <= 0:
				# fallback to uniform over vocab
				topk_probs = np.ones_like(topk_probs, dtype=np.float64)
			topk_probs = topk_probs / float(topk_probs.sum())
			chosen = rng.choice(topk_idx, p=topk_probs)
			seq[i] = int(chosen)
		return seq

	def sample_top_p(self, start_token: int, length: int, p: float = 0.9, temperature: float = 1.0, random_state: Optional[np.random.Generator] = None) -> np.ndarray:
		"""Sample a sequence using nucleus (top-p) sampling at each step.

		Args:
			p: cumulative probability threshold (0 < p <= 1)
		"""
		if not (0.0 < p <= 1.0):
			raise ValueError("p must be in (0, 1]")
		rng = random_state or np.random.default_rng()
		seq = np.empty(length, dtype=np.int64)
		seq[0] = int(start_token)
		for i in range(1, length):
			dist = self.predict_next_distribution(int(seq[i - 1]))
			if temperature <= 0.0:
				raise ValueError("temperature must be > 0")
			if temperature != 1.0:
				logits = np.log(dist + 1e-20) / float(temperature)
				exp_logits = np.exp(logits - np.max(logits))
				dist = exp_logits / exp_logits.sum()

			# sort probabilities descending
			sorted_idx = np.argsort(-dist)
			sorted_probs = dist[sorted_idx]
			cum = np.cumsum(sorted_probs)
			# find minimal set with cumulative prob >= p
			cutoff = np.searchsorted(cum, p) + 1
			cutoff = max(1, cutoff)
			chosen_idx = sorted_idx[:cutoff]
			chosen_probs = dist[chosen_idx]
			if chosen_probs.sum() <= 0:
				chosen_probs = np.ones_like(chosen_probs, dtype=np.float64)
			chosen_probs = chosen_probs / float(chosen_probs.sum())
			choice = rng.choice(chosen_idx, p=chosen_probs)
			seq[i] = int(choice)
		return seq

	def save(self, path: str) -> None:
		"""Save model to disk as JSON + NumPy .npz for counts/probs."""
		p = Path(path)
		p.parent.mkdir(parents=True, exist_ok=True)
		meta = {
			'vocab_size': self.vocab_size,
			'smoothing': float(self.smoothing)
		}
		with open(p.with_suffix('.json'), 'w', encoding='utf8') as f:
			json.dump(meta, f)
		np.savez_compressed(p.with_suffix('.npz'), counts=self.counts, probs=self.probs if self.probs is not None else np.zeros_like(self.counts))

	@classmethod
	def load(cls, path: str) -> 'BigramLanguageModel':
		p = Path(path)
		with open(p.with_suffix('.json'), 'r', encoding='utf8') as f:
			meta = json.load(f)
		arr = np.load(p.with_suffix('.npz'))
		model = cls(vocab_size=int(meta['vocab_size']), smoothing=float(meta.get('smoothing', 1e-6)))
		model.counts = arr['counts'].astype(np.float64)
		probs = arr.get('probs')
		model.probs = probs.astype(np.float64) if probs is not None else None
		return model


__all__ = ["BigramLanguageModel"]

