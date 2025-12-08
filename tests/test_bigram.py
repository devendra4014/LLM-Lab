import numpy as np

from src.llm_book.models.bigram import BigramLanguageModel


def test_fit_and_predict_distribution():
	vocab_size = 5
	model = BigramLanguageModel(vocab_size=vocab_size, smoothing=1e-6)
	# sequence: 0->1, 1->2 (twice), 2->3, 3->4
	tokens = np.array([0, 1, 2, 1, 2, 3, 4], dtype=np.int64)
	model.fit(tokens)

	# probs should be defined and rows sum to ~1
	dist = model.predict_next_distribution(1)
	assert dist.shape == (vocab_size,)
	assert np.isclose(dist.sum(), 1.0)

	# most likely next token after 1 should be 2 (it appears twice in training)
	pred = int(np.argmax(dist))
	assert pred == 2


def test_log_prob_and_sampling():
	vocab_size = 4
	model = BigramLanguageModel(vocab_size=vocab_size, smoothing=1e-6)
	tokens = np.array([0, 1, 2, 3, 0, 1], dtype=np.int64)
	model.fit(tokens)

	lp = model.log_prob_sequence(tokens)
	assert np.isfinite(lp)

	seq = model.sample(start_token=0, length=6, random_state=np.random.default_rng(42))
	assert len(seq) == 6
	assert seq.dtype == np.int64
	assert seq.min() >= 0 and seq.max() < vocab_size


def test_save_and_load(tmp_path):
	vocab_size = 6
	model = BigramLanguageModel(vocab_size=vocab_size)
	tokens = np.array([0, 1, 2, 3, 4, 5, 0, 1], dtype=np.int64)
	model.fit(tokens)

	p = tmp_path / "bigram_model"
	model.save(str(p))

	loaded = BigramLanguageModel.load(str(p))
	# counts should match
	assert np.allclose(model.counts, loaded.counts)
	# probs should match (or both be None)
	if model.probs is None:
		assert loaded.probs is None
	else:
		assert np.allclose(model.probs, loaded.probs)

