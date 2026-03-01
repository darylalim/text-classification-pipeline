# Text Classification Pipeline

Classify sentiment in English text using [SiEBERT](https://huggingface.co/siebert/sentiment-roberta-large-english), a RoBERTa-based model trained on ~1.4M diverse texts.

## Features

- Upload a CSV or try built-in sample data
- Auto-detects the text column; select any column to classify
- Binary sentiment (positive/negative) with confidence scores
- Summary metrics: total rows, positive/negative counts, average confidence
- Results table with CSV download
- Batched GPU/MPS/CPU inference
- Handles empty, whitespace-only, and malformed input

## Setup

```bash
uv sync
```

Set a [Hugging Face token](https://huggingface.co/settings/tokens) for authenticated model downloads:

```bash
export HF_TOKEN=hf_...
```

## Usage

```bash
uv run streamlit run streamlit_app.py
```

## Testing

```bash
uv run pytest
```
