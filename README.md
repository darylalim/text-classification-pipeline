# Text Classification Pipeline

A Streamlit app that classifies customer review sentiments using an IBM Granite 4.0 language model.

## Features

- Drag and drop CSV file upload
- Automatic device detection (CUDA → MPS → CPU)
- Batch processing for efficient classification
- Adds a "Sentiment" column with positive/negative labels
- Download results as CSV

## Usage

1. Create and activate a virtual environment:
   ```bash
   python3.12 -m venv streamlit_env
   source streamlit_env/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run streamlit_app.py
   ```

4. Upload a CSV file, select the column containing review text, and click "Classify".
