# UI Redesign Design

## Goal

Redesign the Streamlit UI to be simple, friendly, and delightful. Target audience is both technical and non-technical users. Priority is clarity and guidance — the user should always know what to do next.

## Approach

Guided step-by-step flow with progressive disclosure. Each stage reveals only when the previous stage is complete.

## Page Config and Theming

- `st.set_page_config(page_title="Text Classification Pipeline", page_icon, layout="wide")`
- `.streamlit/config.toml` with a clean, neutral color scheme
- Sidebar contains:
  - "How it works" — 3 numbered steps (upload, select column, classify)
  - Model info: "Powered by SiEBERT (RoBERTa-large)"
  - Device indicator: "Running on MPS/CUDA/CPU"

## Landing State

When no file is uploaded:

- Title: "Text Classification Pipeline"
- Subtitle: "Classify the sentiment of text in your CSV as positive or negative."
- Two columns side by side:
  - Left: file uploader (CSV only)
  - Right: "Try with sample data" button (loads `tests/data/csv/mixed_sample.csv`)
- Below: "Supports CSV files. Your data is processed locally and never stored."

## Column Selection and Preview

Once CSV is loaded:

- Subheader: "Select the column containing text to classify"
- Column selector with auto-detection (picks first `object`-dtype column)
- Preview of selected column's first 5 values
- "Classify" button (primary) — only appears after column is selected

## Results Display

After classification:

- Success message
- Summary metrics row (`st.metric` in columns):
  - Total rows processed
  - Positive count (with percentage)
  - Negative count (with percentage)
  - Average confidence
- Results table with color-highlighted Sentiment column (green=positive, red=negative) via Pandas Styler
- "Download results as CSV" button
- "Start over" button to clear session state

## Error Handling

- Empty CSV: `st.warning("This CSV has no rows. Please upload a file with data.")`
- No string columns: `st.warning("No text columns detected. Please check your CSV.")`
- All blank texts: `st.info("All values in this column are empty. No classification was performed.")`
- Malformed CSV: try/except around `pd.read_csv` with `st.error("Could not read this file. Please check it's a valid CSV.")`
- Progress: keep existing progress bar + `st.spinner` wrapper

## Files Changed

- `streamlit_app.py` — significant rewrite of UI section (lines 79-118), minor changes to `process_dataframe` for session state
- `.streamlit/config.toml` — new file for theming
- Tests updated to cover new UI behavior and error handling
