import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

st.set_page_config(page_title="Sentiment Classifier")


def get_device():
    """Detect the best available device: CUDA → MPS → CPU"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


@st.cache_resource
def load_model():
    """Load model and tokenizer at application startup"""
    device = get_device()
    model_path = "ibm-granite/granite-4.0-h-tiny"
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16)
    model = model.to(device)
    
    return model, tokenizer, device


def create_prompt(review_text):
    """Create the classification prompt for a single review"""
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


def classify_batch(reviews, model, tokenizer, device):
    """Classify a batch of reviews and return sentiment labels"""
    sentiments = []
    
    for review in reviews:
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
        
        # Extract sentiment from the response
        response_lower = decoded.lower()
        if "negative" in response_lower.split("sentiment:")[-1]:
            sentiments.append("negative")
        else:
            sentiments.append("positive")
    
    return sentiments


def process_dataframe(df, text_column, model, tokenizer, device, batch_size=16):
    """Process the entire dataframe in batches"""
    reviews = df[text_column].tolist()
    all_sentiments = []
    
    total_batches = (len(reviews) + batch_size - 1) // batch_size
    progress_bar = st.progress(0)
    
    for i in range(0, len(reviews), batch_size):
        batch = reviews[i:i + batch_size]
        batch_sentiments = classify_batch(batch, model, tokenizer, device)
        all_sentiments.extend(batch_sentiments)
        
        # Update progress
        current_batch = (i // batch_size) + 1
        progress_bar.progress(current_batch / total_batches)
    
    df["Sentiment"] = all_sentiments
    return df

st.title("Customer Review Sentiment Classifier")

with st.spinner("Loading model..."):
    model, tokenizer, device = load_model()
st.success(f"Model loaded on {device.upper()}")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.write("**File preview**")
    st.dataframe(df.head())
    
    # Column selection dropdown
    columns = df.columns.tolist()
    text_column = st.selectbox(
        "Select column for review text",
        options=columns,
        index=columns.index("SUMMARY") if "SUMMARY" in columns else 0
    )
    
    # Process button
    if st.button("Classify"):
        with st.spinner("Classifying..."):
            result_df = process_dataframe(df, text_column, model, tokenizer, device, batch_size=16)
        
        st.success("Done")
        
        st.write("**Sentiment preview**")
        st.dataframe(result_df.head())
        
        # Prepare download
        original_filename = uploaded_file.name.rsplit(".", 1)[0]
        output_filename = f"{original_filename}_sentiment.csv"
        
        csv_data = result_df.to_csv(index=False)
        
        st.download_button(
            label="Download",
            data=csv_data,
            file_name=output_filename,
            mime="text/csv"
        )
