# Text Classification Pipeline

Classify sentiments in customer reviews with IBM Granite 4.0 language models.

## Features

- Batched inference (`BATCH_SIZE` default 8) for GPU/MPS parallelism
- Few-shot prompt with positive and negative examples
- Upload a CSV, select the text column, download results with a "Sentiment" column

## Setup

```bash
python3.12 -m venv streamlit_env
source streamlit_env/bin/activate
pip install -r requirements.txt
```

Set a [Hugging Face token](https://huggingface.co/settings/tokens) for authenticated model downloads:

```bash
export HF_TOKEN=hf_...
```

## Usage

```bash
streamlit run streamlit_app.py
```

## Testing

```bash
pytest
```
