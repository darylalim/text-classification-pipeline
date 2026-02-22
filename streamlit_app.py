import os

import pandas as pd
import streamlit as st
import torch
from dotenv import load_dotenv
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    logging as hf_logging,
)

load_dotenv()

BATCH_SIZE = 8


def get_device():
    """Automatically detect the best available device in order of priority: MPS, CUDA, CPU."""
    return (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


@st.cache_resource
def load_model(device):
    """Load model and tokenizer at application startup."""
    model_path = "siebert/sentiment-roberta-large-english"
    token = os.environ.get("HF_TOKEN")
    hf_logging.set_verbosity_error()
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, dtype=torch.float16, token=token
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
    return model, tokenizer


def process_dataframe(df, text_column, model, tokenizer, device):
    """Process a dataframe in batches with progress updates."""
    texts = df[text_column].astype(str).tolist()
    sentiments = [""] * len(texts)
    confidences = [0.0] * len(texts)
    progress_bar = st.progress(0)

    valid = [(i, t) for i, t in enumerate(texts) if t.strip()]

    if valid:
        id2label = model.config.id2label
        indices, valid_texts = zip(*valid)

        for start in range(0, len(valid_texts), BATCH_SIZE):
            batch = list(valid_texts[start : start + BATCH_SIZE])
            inputs = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True
            ).to(device)

            with torch.inference_mode():
                probs = torch.softmax(model(**inputs).logits, dim=-1)
                max_probs, preds = probs.max(dim=-1)

            for j, idx in enumerate(indices[start : start + BATCH_SIZE]):
                sentiments[idx] = id2label[preds[j].item()].lower()
                confidences[idx] = round(max_probs[j].item(), 4)

            progress_bar.progress((start + len(batch)) / len(valid_texts))
    else:
        progress_bar.progress(1.0)

    result = df.copy()
    result["Sentiment"] = sentiments
    result["Confidence"] = confidences
    return result


st.title("Text Classification Pipeline")
st.write("Classify sentiment in text with SiEBERT.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

device = get_device()

with st.spinner(f"Loading model on {device.upper()}..."):
    model, tokenizer = load_model(device)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("Preview")
    st.dataframe(df.head())

    columns = df.columns.tolist()
    text_column = st.selectbox(
        "Select text column",
        options=columns,
        index=0,
    )

    if st.button("Classify", type="primary"):
        st.write("Classifying...")
        result_df = process_dataframe(df, text_column, model, tokenizer, device)

        st.success("Done.")

        st.write("Preview")
        st.dataframe(result_df.head())

        csv_data = result_df.to_csv(index=False)

        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}_sentiment.csv",
            mime="text/csv",
        )
