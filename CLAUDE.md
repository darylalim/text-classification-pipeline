# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Streamlit web app that classifies text sentiment as positive/negative using SiEBERT (`siebert/sentiment-roberta-large-english`). Users upload a CSV, select the text column, and download results with "Sentiment" and "Confidence" columns.

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

Single-file application (`streamlit_app.py`, ~120 lines):

1. **`get_device`** — selects MPS, CUDA, or CPU
2. **`load_model`** — loads model/tokenizer once via `@st.cache_resource` in float16; authenticates with `HF_TOKEN`
3. **`process_dataframe`** — pre-filters blanks, batches valid texts (`BATCH_SIZE=8`), classifies via softmax over logits
4. **UI** — file upload → column selection → classify → preview → CSV download

## Key Patterns

- `torch.inference_mode()` for all inference
- `hf_logging.set_verbosity_error()` suppresses expected checkpoint warnings
- Confidence via `torch.softmax(logits, dim=-1).max(dim=-1)`; labels from `model.config.id2label`
- Empty/whitespace-only texts skipped; get sentiment `""` and confidence `0.0`
- Tokenizer uses `truncation=True` (512 token limit) and `padding=True`
- `process_dataframe` returns a copy; input DataFrame is not mutated
- Dependencies managed by `uv` with lockfile (`uv.lock`)

## Tests

- `tests/test_streamlit_app.py` — unit tests for `get_device`, `load_model`, `process_dataframe`, and `BATCH_SIZE` with mocked dependencies
- `tests/data/csv/product_reviews.csv` — 40 e-commerce product reviews
- `tests/data/csv/movie_reviews.csv` — 40 film and TV opinions
- `tests/data/csv/social_media.csv` — 40 tweets and social media posts
- `tests/data/csv/restaurant_reviews.csv` — 40 dining and food service reviews
- `tests/data/csv/app_reviews.csv` — 40 mobile/web app store reviews
- `tests/data/csv/mixed_sample.csv` — 20-row sample (4 from each domain) for quick testing
