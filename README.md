# Text Classification Pipeline

Classify sentiment in English text (reviews, tweets, social media posts, etc.) using [SiEBERT](https://huggingface.co/siebert/sentiment-roberta-large-english), a RoBERTa-based model trained on ~1.4M diverse English texts.

## Features

- Binary sentiment classification (positive/negative) with confidence scores
- Batched inference (`BATCH_SIZE` default 8) for GPU/MPS parallelism
- Handles empty and whitespace-only text entries
- Upload a CSV, select the text column, download results with "Sentiment" and "Confidence" columns

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
