import os
import warnings

import pandas as pd
import streamlit as st
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

warnings.filterwarnings(
    "ignore", message="MPS:.*constant padding.*not currently supported"
)

BATCH_SIZE = 8

PROMPT_TEMPLATE = """Classify the sentiment of the customer reviews as positive or negative.
Your response should only include the answer. Do not provide any further explanation.

Here are some examples, complete the last one:
Review:
Oh, where do I even begin? This product is a tour de force that has left me utterly captivated, enchanted, and spellbound. Every aspect of this purchase was nothing short of pure excellence, deserving nothing less than a perfect 10 out of 10 rating!
Sentiment:
positive

Review:
It's a shame because I had high hopes. This product did not deliver. The quality is poor, the design is flawed, and customer service was unhelpful. There were so many issues that could have been avoided. So many promises were made, but none were kept.
Sentiment:
negative

Review:
{}
Sentiment:"""


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
    model_path = "ibm-granite/granite-4.0-h-tiny"
    token = os.environ.get("HF_TOKEN")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float16, token=token, low_cpu_mem_usage=True
    ).half().to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def process_dataframe(df, text_column, model, tokenizer, device):
    """Process a dataframe in batches with progress updates."""
    reviews = df[text_column].tolist()
    all_sentiments = []
    total = len(reviews)
    progress_bar = st.progress(0)

    if not reviews:
        progress_bar.progress(1.0)
        df = df.copy()
        df["Sentiment"] = all_sentiments
        return df

    for start in range(0, total, BATCH_SIZE):
        batch = reviews[start : start + BATCH_SIZE]
        chats = [
            [{"role": "user", "content": PROMPT_TEMPLATE.format(review)}]
            for review in batch
        ]
        texts = [
            tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
            for chat in chats
        ]
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)
        input_length = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            outputs = model.generate(
                **inputs, max_new_tokens=10, pad_token_id=tokenizer.pad_token_id
            )

        for output in outputs:
            new_tokens = tokenizer.decode(
                output[input_length:], skip_special_tokens=True
            )
            all_sentiments.append(
                "negative" if "negative" in new_tokens.lower() else "positive"
            )

        progress_bar.progress(min(start + len(batch), total) / total)

    result = df.copy()
    result["Sentiment"] = all_sentiments
    return result


st.title("Text Classification Pipeline")
st.write(
    "Classify sentiments in customer reviews with IBM Granite 4.0 language models."
)

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
        "Select column for review text",
        options=columns,
        index=columns.index("SUMMARY") if "SUMMARY" in columns else 0,
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
