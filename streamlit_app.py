import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_device():
    """Automatically detect the best available device in order of priority: MPS, CUDA, CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

@st.cache_resource
def load_model(device):
    """Load model and tokenizer at application startup."""
    model_path = "ibm-granite/granite-4.0-h-tiny"
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def create_prompt(review_text):
    """Create the classification prompt for a review."""
    prompt = f"""Classify the sentiment of the customer reviews as positive or negative.
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
{review_text}
Sentiment:"""
    return prompt

def classify_review(review, model, tokenizer, device):
    """Classify a single review and return the sentiment label."""
    prompt = create_prompt(review)
    chat = [{"role": "user", "content": prompt}]
    chat_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    
    input_tokens = tokenizer(chat_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = model.generate(
            **input_tokens,
            max_new_tokens=10,
            pad_token_id=tokenizer.eos_token_id
        )
    
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    
    response_lower = decoded.lower()
    if "negative" in response_lower.split("sentiment:")[-1]:
        return "negative"
    return "positive"


def process_dataframe(df, text_column, model, tokenizer, device):
    """Process a dataframe with per review progress updates."""
    reviews = df[text_column].tolist()
    all_sentiments = []
    
    total_reviews = len(reviews)
    progress_bar = st.progress(0)
    
    for i, review in enumerate(reviews):
        sentiment = classify_review(review, model, tokenizer, device)
        all_sentiments.append(sentiment)
        progress_bar.progress((i + 1) / total_reviews)
    
    df["Sentiment"] = all_sentiments
    return df

st.title("Text Classification Pipeline")
st.write("Classify sentiments in customer reviews with IBM Granite 4.0 language models.")

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
        index=columns.index("SUMMARY") if "SUMMARY" in columns else 0
    )
    
    if st.button("Classify", type="primary"):
        st.write("Classifying...")
        result_df = process_dataframe(df, text_column, model, tokenizer, device)
        
        st.success("Done.")
        
        st.write("Preview")
        st.dataframe(result_df.head())
        
        # Prepare CSV for download
        csv_data = result_df.to_csv(index=False)
        
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}_sentiment.csv",
            mime="text/csv"
        )
