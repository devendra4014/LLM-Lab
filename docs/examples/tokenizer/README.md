# Tokenizer Examples & Best Practices

This folder contains runnable examples and a converted Best Practices guide for the BPE tokenizer.

Files
- `BPE_BEST_PRACTICES.md` — Human-readable best-practices guide (converted from `bpe_best_practices.py`).
- `bpe_examples.py` — Runnable examples demonstrating common workflows (kept in `src/llm_book/tokenizer` as the canonical runnable module).

Quick Start

1. Install dependencies in the project virtual environment:

```powershell
E:/code/DL/LLM/repo-root/.venv/Scripts/python.exe -m pip install -r requirements.txt
```

2. Run the examples (from repo root):

```powershell
python src/llm_book/tokenizer/bpe_examples.py
```

3. Run unit tests:

```powershell
E:/code/DL/LLM/repo-root/.venv/Scripts/python.exe -m pytest -q
```

Notes
- `bpe_examples.py` is intentionally kept runnable in `src/llm_book/tokenizer/` so contributors can execute examples quickly.
- The markdown guide `BPE_BEST_PRACTICES.md` is the authored documentation and should be used as the primary reference for patterns and integration tips.

Suggested next steps
- Move selected example snippets into a Jupyter notebook under `docs/examples/notebooks/` for interactive exploration.
- Add a short index to `docs/README.md` linking to this folder.

