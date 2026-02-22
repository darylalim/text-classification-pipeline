# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Streamlit web app that classifies text sentiment (reviews, tweets, social media posts, etc.) as positive/negative using SiEBERT (`siebert/sentiment-roberta-large-english`), a RoBERTa-based sequence classification model. Users upload a CSV, select the text column, and download results with "Sentiment" and "Confidence" columns.

## Commands

```bash
# Setup
uv sync

# Run
uv run streamlit run streamlit_app.py

# Lint
uv run ruff check .

# Format
uv run ruff format .

# Type check
uv run ty check .

# Test
uv run pytest                                          # all tests
uv run pytest tests/test_streamlit_app.py              # single file
uv run pytest tests/test_streamlit_app.py::test_name   # single test
```

Use `ruff` for all linting and formatting. Run `uv run ruff check --fix .` to auto-fix lint issues. Use `ty` for type checking. Use `pytest` for unit testing.

## Architecture

Single-file application (`streamlit_app.py`, ~150 lines):

1. **Device detection** (`get_device`) — selects MPS, CUDA, or CPU
2. **Model loading** (`load_model`) — cached via `@st.cache_resource`, float16 precision, authenticates with `HF_TOKEN` env var
3. **Batch processing** (`process_dataframe`) — chunks texts by `BATCH_SIZE` (default 8), tokenizes raw text, runs forward pass, extracts labels and confidence scores from logits via softmax, skips empty/whitespace-only texts
4. **UI** — file upload → column selection → classify → preview → CSV download

## Key Patterns

- Model and tokenizer loaded once per session via `@st.cache_resource`
- `torch.inference_mode()` for inference
- Confidence scores via `torch.softmax(logits, dim=-1).max(dim=-1)` — labels from `model.config.id2label`
- Empty/whitespace-only texts get sentiment `""` and confidence `0.0`
- Tokenizer called with `truncation=True` (RoBERTa max 512 tokens) and `padding=True` for batching
- `process_dataframe` returns a copy — input DataFrame is not mutated
- Dependencies managed by `uv` with lockfile (`uv.lock`) for reproducible installs

## Tests

- `tests/test_streamlit_app.py` — unit tests covering `get_device`, `load_model`, `process_dataframe`, and `BATCH_SIZE` with mocked model/tokenizer/Streamlit dependencies
- `tests/data/csv/customer_reviews.csv` — 100 reviews (PRODUCT, DATE, SUMMARY, SENTIMENT_SCORE, Order ID)
- `tests/data/csv/customer_reviews_sample.csv` — 11-row subset for quick testing
