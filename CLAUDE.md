# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Streamlit web app that classifies text sentiment as positive/negative using SiEBERT (`siebert/sentiment-roberta-large-english`). Users upload a CSV (or try built-in sample data), select the text column, classify, and download results with "Sentiment" and "Confidence" columns. Guided step-by-step UI with sidebar instructions, auto-detected text columns, summary metrics, and color-coded results. Supports light/dark theme switching with muted colors and comfort typography for extended use.

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

Single-file application (`streamlit_app.py`, ~250 lines):

1. **`get_device`** — selects MPS, CUDA, or CPU
2. **`detect_text_column`** — returns first string-dtype column name for auto-selection
3. **`highlight_sentiment`** — returns theme-aware CSS for Pandas Styler (muted green/red for light, deep green/red for dark); accepts `dark` keyword-only parameter
4. **`load_model`** — loads model/tokenizer once via `@st.cache_resource` in float16; authenticates with `HF_TOKEN`
5. **`process_dataframe`** — pre-filters blanks, batches valid texts (`BATCH_SIZE=8`), classifies via softmax over logits
6. **UI** — guided step-by-step flow: sidebar with instructions and theme hint → comfort CSS injection → file upload or sample data → column auto-detect and preview → classify → summary metrics → color-coded results table → CSV download

## Key Patterns

- `torch.inference_mode()` for all inference
- `hf_logging.set_verbosity_error()` suppresses expected checkpoint warnings
- Confidence via `torch.softmax(logits, dim=-1).max(dim=-1)`; labels from `model.config.id2label`
- Empty/whitespace-only texts skipped; get sentiment `""` and confidence `0.0`
- Tokenizer uses `truncation=True` (512 token limit) and `padding=True`
- `process_dataframe` returns a copy; input DataFrame is not mutated
- `st.session_state` persists loaded DataFrame across Streamlit reruns (buttons reset on rerun)
- `SAMPLE_DATA_PATH` points to `tests/data/csv/mixed_sample.csv` for the "Try with sample data" button
- `.streamlit/config.toml` sets only `primaryColor` and `font`; Streamlit manages light/dark backgrounds and text colors via its built-in theme toggle (Settings menu)
- Theme detection via `st.get_option("theme.base")` with `functools.partial` to pass `dark` flag to `highlight_sentiment` in `style.map`
- Custom CSS injection (`st.markdown`) for 16px font size and 1.6 line height for reading comfort
- Dependencies managed by `uv` with lockfile (`uv.lock`)

## Tests

- `tests/test_streamlit_app.py` — unit tests for `get_device`, `detect_text_column`, `highlight_sentiment`, `load_model`, `process_dataframe`, `BATCH_SIZE`, and `SAMPLE_DATA_PATH` with mocked dependencies
- `tests/data/csv/product_reviews.csv` — 40 e-commerce product reviews
- `tests/data/csv/movie_reviews.csv` — 40 film and TV opinions
- `tests/data/csv/social_media.csv` — 40 tweets and social media posts
- `tests/data/csv/restaurant_reviews.csv` — 40 dining and food service reviews
- `tests/data/csv/app_reviews.csv` — 40 mobile/web app store reviews
- `tests/data/csv/mixed_sample.csv` — 20-row sample (4 from each domain) for quick testing
