# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Streamlit web app that classifies customer review sentiments (positive/negative) using the IBM Granite 4.0 Tiny model (`ibm-granite/granite-4.0-h-tiny`). Users upload a CSV, select the text column, and download results with a new "Sentiment" column.

## Commands

```bash
# Setup
python3.12 -m venv streamlit_env
source streamlit_env/bin/activate
pip install -r requirements.txt

# Run
streamlit run streamlit_app.py

# Lint
ruff check .

# Format
ruff format .

# Type check
ty check .

# Test
pytest                                          # all tests
pytest tests/test_streamlit_app.py              # single file
pytest tests/test_streamlit_app.py::test_name   # single test
```

Use `ruff` for all linting and formatting. Run `ruff check --fix .` to auto-fix lint issues. Use `ty` for type checking. Use `pytest` for unit testing.

## Architecture

Single-file application (`streamlit_app.py`, ~150 lines):

1. **Device detection** (`get_device`) — selects MPS, CUDA, or CPU
2. **Model loading** (`load_model`) — cached via `@st.cache_resource`, float16 precision, left-padding for batched generation, authenticates with `HF_TOKEN` env var
3. **Batch processing** (`process_dataframe`) — chunks reviews by `BATCH_SIZE` (default 8), applies `PROMPT_TEMPLATE`, generates once per batch, decodes only new tokens
4. **UI** — file upload → column selection (defaults to "SUMMARY") → classify → preview → CSV download

## Key Patterns

- Model and tokenizer loaded once per session via `@st.cache_resource`
- `torch.inference_mode()` for inference
- Left-padded batched generation for GPU/MPS parallelism
- Only new tokens decoded; "negative" in output → negative, otherwise positive
- `process_dataframe` returns a copy — input DataFrame is not mutated
- Few-shot prompt defined as module-level `PROMPT_TEMPLATE` constant
- MPS constant-padding warning suppressed (PyTorch fallback works correctly)
- Dependencies unpinned in `requirements.txt`

## Tests

- `tests/test_streamlit_app.py` — 25 unit tests covering `get_device`, `load_model`, `process_dataframe`, `PROMPT_TEMPLATE`, and `BATCH_SIZE` with mocked model/tokenizer/Streamlit dependencies
- `tests/data/csv/customer_reviews.csv` — 100 reviews (PRODUCT, DATE, SUMMARY, SENTIMENT_SCORE, Order ID)
- `tests/data/csv/customer_reviews_sample.csv` — 11-row subset for quick testing
