# Text Classification Pipeline

Classify sentiment in English text using [SiEBERT](https://huggingface.co/siebert/sentiment-roberta-large-english), a RoBERTa-based model trained on ~1.4M diverse texts.

## Features

- Upload a CSV, select the text column, classify, and download results
- Binary sentiment (positive/negative) with confidence scores
- Batched GPU/MPS inference
- Handles empty and whitespace-only text

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
