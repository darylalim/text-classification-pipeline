import os
from pathlib import Path

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
SAMPLE_DATA_PATH = Path(__file__).parent / "tests" / "data" / "csv" / "mixed_sample.csv"


def get_device():
    """Automatically detect the best available device in order of priority: MPS, CUDA, CPU."""
    return (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def detect_text_column(df: pd.DataFrame) -> str | None:
    """Return the name of the first string-dtype column, or None."""
    for col in df.columns:
        if df[col].dtype == "object":
            return col
    return None


def highlight_sentiment(val: str) -> str:
    """Return CSS background color for a sentiment value."""
    if val == "positive":
        return "background-color: #d4edda"
    if val == "negative":
        return "background-color: #f8d7da"
    return ""


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


st.set_page_config(
    page_title="Text Classification Pipeline",
    page_icon=":mag:",
    layout="wide",
)

device = get_device()

with st.spinner(f"Loading model on {device.upper()}..."):
    model, tokenizer = load_model(device)

with st.sidebar:
    st.header("How it works")
    st.markdown(
        "1. **Upload** a CSV file (or try sample data)\n"
        "2. **Select** the column containing text\n"
        "3. **Classify** and download results"
    )
    st.divider()
    st.caption("Powered by SiEBERT (RoBERTa-large)")
    st.caption(f"Running on {device.upper()}")

st.title("Text Classification Pipeline")
st.write("Classify the sentiment of text in your CSV as positive or negative.")

col_upload, col_sample = st.columns(2)

with col_upload:
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

with col_sample:
    st.write("")
    st.write("")
    use_sample = st.button("Try with sample data")

st.caption("Supports CSV files. Your data is processed locally and never stored.")

if use_sample:
    st.session_state["df"] = pd.read_csv(SAMPLE_DATA_PATH)
    st.session_state["source_name"] = "mixed_sample"
elif uploaded_file is not None:
    try:
        st.session_state["df"] = pd.read_csv(uploaded_file)
        st.session_state["source_name"] = uploaded_file.name.rsplit(".", 1)[0]
    except pd.errors.ParserError:
        st.error("Could not read this file. Please check it's a valid CSV.")

df = st.session_state.get("df")
source_name = st.session_state.get("source_name", "")

if df is not None:
    if df.empty:
        st.warning("This CSV has no rows. Please upload a file with data.")
    else:
        if detect_text_column(df) is None:
            st.warning("No text columns detected. Please check your CSV.")
        else:
            st.subheader("Select the column containing text to classify")

            default_col = detect_text_column(df)
            columns = df.columns.tolist()
            default_index = columns.index(default_col) if default_col else 0

            text_column = st.selectbox(
                "Text column",
                options=columns,
                index=default_index,
            )

            st.write("Preview of selected column")
            st.dataframe(df[[text_column]].head(), use_container_width=True)

            if st.button("Classify", type="primary"):
                with st.spinner("Classifying..."):
                    result_df = process_dataframe(
                        df, text_column, model, tokenizer, device
                    )

                all_blank = result_df["Sentiment"].eq("").all()

                if all_blank:
                    st.info(
                        "All values in this column are empty. "
                        "No classification was performed."
                    )
                    csv_data = result_df.to_csv(index=False)
                    st.download_button(
                        label="Download results as CSV",
                        data=csv_data,
                        file_name=f"{source_name}_sentiment.csv",
                        mime="text/csv",
                    )
                else:
                    st.success("Classification complete!")

                    total = len(result_df)
                    classified = result_df[result_df["Sentiment"] != ""]
                    pos_count = int((classified["Sentiment"] == "positive").sum())
                    neg_count = int((classified["Sentiment"] == "negative").sum())
                    avg_conf = (
                        classified["Confidence"].mean() if len(classified) > 0 else 0.0
                    )

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total rows", total)
                    m2.metric(
                        "Positive", f"{pos_count} ({pos_count / total * 100:.0f}%)"
                    )
                    m3.metric(
                        "Negative", f"{neg_count} ({neg_count / total * 100:.0f}%)"
                    )
                    m4.metric("Avg confidence", f"{avg_conf:.1%}")

                    styled = result_df.style.map(
                        highlight_sentiment, subset=["Sentiment"]
                    )
                    st.dataframe(styled, use_container_width=True)

                    csv_data = result_df.to_csv(index=False)
                    st.download_button(
                        label="Download results as CSV",
                        data=csv_data,
                        file_name=f"{source_name}_sentiment.csv",
                        mime="text/csv",
                    )
