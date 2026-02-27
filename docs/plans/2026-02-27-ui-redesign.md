# UI Redesign Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Redesign the Streamlit UI into a guided step-by-step flow with progressive disclosure, sample data, summary metrics, and color-coded results.

**Architecture:** Keep existing backend functions (`get_device`, `load_model`, `process_dataframe`) unchanged. Add two new pure helper functions (`detect_text_column`, `highlight_sentiment`) and a `SAMPLE_DATA_PATH` constant. Rewrite the UI section (lines 79-118) to use Streamlit session state for progressive disclosure. Add `.streamlit/config.toml` for theming.

**Tech Stack:** Streamlit (page config, session state, columns, metrics, sidebar), Pandas Styler, pathlib

---

### Task 1: Create Streamlit theme config

**Files:**
- Create: `.streamlit/config.toml`

**Step 1: Create the config file**

```toml
[theme]
primaryColor = "#4A90D9"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F5F7FA"
textColor = "#1A1A2E"
font = "sans serif"
```

**Step 2: Verify the directory and file exist**

Run: `ls -la .streamlit/config.toml`
Expected: File exists with correct contents

**Step 3: Commit**

```bash
git add .streamlit/config.toml
git commit -m "feat: add Streamlit theme config"
```

---

### Task 2: Add `SAMPLE_DATA_PATH` constant

**Files:**
- Modify: `streamlit_app.py:1-15`
- Test: `tests/test_streamlit_app.py`

**Step 1: Write the failing test**

Add to `tests/test_streamlit_app.py` after the `BATCH_SIZE` test (line 15):

```python
# --- SAMPLE_DATA_PATH ---


def test_sample_data_path_exists():
    from streamlit_app import SAMPLE_DATA_PATH

    assert SAMPLE_DATA_PATH.exists()
    assert SAMPLE_DATA_PATH.suffix == ".csv"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_streamlit_app.py::test_sample_data_path_exists -v`
Expected: FAIL with ImportError (SAMPLE_DATA_PATH not defined)

**Step 3: Write minimal implementation**

Add to `streamlit_app.py` after `BATCH_SIZE = 8` (line 15):

```python
from pathlib import Path

SAMPLE_DATA_PATH = Path(__file__).parent / "tests" / "data" / "csv" / "mixed_sample.csv"
```

Note: add `from pathlib import Path` to the imports at the top of the file.

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_streamlit_app.py::test_sample_data_path_exists -v`
Expected: PASS

**Step 5: Commit**

```bash
git add streamlit_app.py tests/test_streamlit_app.py
git commit -m "feat: add SAMPLE_DATA_PATH constant"
```

---

### Task 3: Add `detect_text_column` function

**Files:**
- Modify: `streamlit_app.py` (add function after `get_device`)
- Test: `tests/test_streamlit_app.py`

**Step 1: Write the failing tests**

Add a new test class to `tests/test_streamlit_app.py`:

```python
# --- detect_text_column ---


class TestDetectTextColumn:
    def test_returns_first_object_column(self):
        from streamlit_app import detect_text_column

        df = pd.DataFrame({"id": [1, 2], "review": ["good", "bad"], "score": [5, 1]})
        assert detect_text_column(df) == "review"

    def test_skips_non_object_columns(self):
        from streamlit_app import detect_text_column

        df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
        assert detect_text_column(df) is None

    def test_returns_first_when_multiple_object_columns(self):
        from streamlit_app import detect_text_column

        df = pd.DataFrame({"name": ["Alice", "Bob"], "text": ["hi", "bye"]})
        assert detect_text_column(df) == "name"

    def test_returns_none_for_empty_dataframe(self):
        from streamlit_app import detect_text_column

        df = pd.DataFrame()
        assert detect_text_column(df) is None
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_streamlit_app.py::TestDetectTextColumn -v`
Expected: FAIL with ImportError

**Step 3: Write minimal implementation**

Add to `streamlit_app.py` after `get_device()`:

```python
def detect_text_column(df):
    """Return the name of the first string-dtype column, or None."""
    for col in df.columns:
        if df[col].dtype == "object":
            return col
    return None
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_streamlit_app.py::TestDetectTextColumn -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add streamlit_app.py tests/test_streamlit_app.py
git commit -m "feat: add detect_text_column helper"
```

---

### Task 4: Add `highlight_sentiment` function

**Files:**
- Modify: `streamlit_app.py` (add function after `detect_text_column`)
- Test: `tests/test_streamlit_app.py`

**Step 1: Write the failing tests**

Add a new test class to `tests/test_streamlit_app.py`:

```python
# --- highlight_sentiment ---


class TestHighlightSentiment:
    def test_positive_returns_green(self):
        from streamlit_app import highlight_sentiment

        result = highlight_sentiment("positive")
        assert "background-color" in result
        assert "#d4edda" in result

    def test_negative_returns_red(self):
        from streamlit_app import highlight_sentiment

        result = highlight_sentiment("negative")
        assert "background-color" in result
        assert "#f8d7da" in result

    def test_empty_returns_empty_string(self):
        from streamlit_app import highlight_sentiment

        assert highlight_sentiment("") == ""

    def test_unknown_returns_empty_string(self):
        from streamlit_app import highlight_sentiment

        assert highlight_sentiment("neutral") == ""
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_streamlit_app.py::TestHighlightSentiment -v`
Expected: FAIL with ImportError

**Step 3: Write minimal implementation**

Add to `streamlit_app.py` after `detect_text_column`:

```python
def highlight_sentiment(val):
    """Return CSS background color for a sentiment value."""
    if val == "positive":
        return "background-color: #d4edda"
    if val == "negative":
        return "background-color: #f8d7da"
    return ""
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_streamlit_app.py::TestHighlightSentiment -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add streamlit_app.py tests/test_streamlit_app.py
git commit -m "feat: add highlight_sentiment helper"
```

---

### Task 5: Add page config and sidebar

**Files:**
- Modify: `streamlit_app.py:79-88` (replace current title/spinner section)

**Step 1: Add page config**

Replace lines 79-88 of `streamlit_app.py` with:

```python
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
    st.caption(f"Powered by SiEBERT (RoBERTa-large)")
    st.caption(f"Running on {device.upper()}")

st.title("Text Classification Pipeline")
st.write("Classify the sentiment of text in your CSV as positive or negative.")
```

Note: `st.set_page_config` must be the first Streamlit command. Move it before any other `st.*` calls.

**Step 2: Run existing tests to verify nothing broke**

Run: `uv run pytest tests/test_streamlit_app.py -v`
Expected: All existing tests PASS

**Step 3: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: add page config and sidebar"
```

---

### Task 6: Add landing state with sample data button

**Files:**
- Modify: `streamlit_app.py` (replace file uploader section)

**Step 1: Rewrite the upload section**

Replace the file uploader and the `if uploaded_file is not None:` block opening with:

```python
col_upload, col_sample = st.columns(2)

with col_upload:
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

with col_sample:
    st.write("")  # vertical spacer to align with uploader
    st.write("")
    use_sample = st.button("Try with sample data")

st.caption("Supports CSV files. Your data is processed locally and never stored.")

df = None

if use_sample:
    df = pd.read_csv(SAMPLE_DATA_PATH)
    source_name = "mixed_sample"
elif uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        source_name = uploaded_file.name.rsplit(".", 1)[0]
    except Exception:
        st.error("Could not read this file. Please check it's a valid CSV.")
```

**Step 2: Run existing tests to verify nothing broke**

Run: `uv run pytest tests/test_streamlit_app.py -v`
Expected: All existing tests PASS

**Step 3: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: add landing state with sample data and CSV error handling"
```

---

### Task 7: Add column selection with auto-detect and preview

**Files:**
- Modify: `streamlit_app.py` (replace column selection section)

**Step 1: Rewrite column selection**

Replace the old column selection and classify button logic with:

```python
if df is not None:
    if df.empty:
        st.warning("This CSV has no rows. Please upload a file with data.")
    else:
        string_columns = [c for c in df.columns if df[c].dtype == "object"]
        if not string_columns:
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
```

**Step 2: Run existing tests to verify nothing broke**

Run: `uv run pytest tests/test_streamlit_app.py -v`
Expected: All existing tests PASS

**Step 3: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: add column selection with auto-detect and preview"
```

---

### Task 8: Add results display with metrics and color-coded table

**Files:**
- Modify: `streamlit_app.py` (replace classify/results section)

**Step 1: Rewrite the classify and results section**

Add inside the `else` block after column selection (continuing from Task 7):

```python
            if st.button("Classify", type="primary"):
                with st.spinner("Classifying..."):
                    result_df = process_dataframe(
                        df, text_column, model, tokenizer, device
                    )

                st.success("Classification complete!")

                # Summary metrics
                total = len(result_df)
                classified = result_df[result_df["Sentiment"] != ""]
                pos_count = (classified["Sentiment"] == "positive").sum()
                neg_count = (classified["Sentiment"] == "negative").sum()
                avg_conf = classified["Confidence"].mean() if len(classified) > 0 else 0.0

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total rows", total)
                m2.metric("Positive", f"{pos_count} ({pos_count / total * 100:.0f}%)")
                m3.metric("Negative", f"{neg_count} ({neg_count / total * 100:.0f}%)")
                m4.metric("Avg confidence", f"{avg_conf:.1%}")

                # Color-coded results table
                styled = result_df.style.map(
                    highlight_sentiment, subset=["Sentiment"]
                )
                st.dataframe(styled, use_container_width=True)

                # Download
                csv_data = result_df.to_csv(index=False)
                st.download_button(
                    label="Download results as CSV",
                    data=csv_data,
                    file_name=f"{source_name}_sentiment.csv",
                    mime="text/csv",
                )
```

**Step 2: Run existing tests to verify nothing broke**

Run: `uv run pytest tests/test_streamlit_app.py -v`
Expected: All existing tests PASS

**Step 3: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: add results display with metrics and color-coded table"
```

---

### Task 9: Add all-blank-texts info message

**Files:**
- Modify: `streamlit_app.py` (add check after classification)

**Step 1: Add the check**

Inside the classify button block, after `result_df = process_dataframe(...)` and before `st.success(...)`, add:

```python
                    all_blank = result_df["Sentiment"].eq("").all()

                if all_blank:
                    st.info(
                        "All values in this column are empty. "
                        "No classification was performed."
                    )
                else:
```

Then indent the success message, metrics, styled table, and download button inside the `else` block. The all-blank case still gets the download button (returning the original data with empty Sentiment/Confidence).

After the `else` block, add the download for the all-blank case too:

```python
                    # Still offer download for blank case
                    csv_data = result_df.to_csv(index=False)
                    st.download_button(
                        label="Download results as CSV",
                        data=csv_data,
                        file_name=f"{source_name}_sentiment.csv",
                        mime="text/csv",
                    )
```

**Step 2: Run existing tests to verify nothing broke**

Run: `uv run pytest tests/test_streamlit_app.py -v`
Expected: All existing tests PASS

**Step 3: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: add info message for all-blank text columns"
```

---

### Task 10: Run full test suite and lint

**Files:**
- No new files

**Step 1: Run all tests**

Run: `uv run pytest tests/test_streamlit_app.py -v`
Expected: All tests PASS (existing + new)

**Step 2: Run linter**

Run: `uv run ruff check .`
Expected: No errors (or fix with `uv run ruff check --fix .`)

**Step 3: Run formatter**

Run: `uv run ruff format .`
Expected: Files formatted

**Step 4: Run type checker**

Run: `uv run ty check .`
Expected: No blocking errors

**Step 5: Commit any fixes**

```bash
git add -A
git commit -m "chore: fix lint and formatting"
```

---

### Task 11: Manual smoke test

**Files:**
- No changes

**Step 1: Run the app**

Run: `uv run streamlit run streamlit_app.py`

**Step 2: Verify the following**

- Page loads with title, subtitle, sidebar with "How it works"
- Sidebar shows model info and device
- Two-column layout: file uploader on left, "Try with sample data" on right
- Caption about local processing appears below
- Clicking "Try with sample data" loads data and shows column selector
- Column selector auto-detects the `text` column
- Preview shows first 5 values from selected column
- Clicking "Classify" shows spinner, progress bar, then results
- Results show: success message, 4 metrics, color-coded table, download button
- Uploading a malformed file shows error message
- Theme colors match the config
