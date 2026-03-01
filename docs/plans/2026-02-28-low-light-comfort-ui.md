# Low-Light Comfort UI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable light/dark theme switching with muted sentiment colors and improved font readability for extended use in low-light environments.

**Architecture:** Remove forced light theme from `config.toml` so Streamlit's built-in theme toggle works. Update `highlight_sentiment` to accept a `dark` boolean and return theme-appropriate muted colors. Inject custom CSS for larger font size and line height. Add a sidebar hint about the theme toggle.

**Tech Stack:** Streamlit (theme config, `st.get_option`, `st.markdown`), Pandas Styler, CSS

---

### Task 1: Update theme config to allow dark mode

**Files:**
- Modify: `.streamlit/config.toml:1-7`

**Step 1: Update config.toml**

Replace the full contents of `.streamlit/config.toml` with:

```toml
[theme]
primaryColor = "#5B9BD5"
font = "sans serif"
```

This removes `backgroundColor`, `secondaryBackgroundColor`, and `textColor`, allowing Streamlit to supply its own values for light and dark modes. The muted blue `#5B9BD5` is the accent color for both modes.

**Step 2: Verify the app starts**

Run: `uv run streamlit run streamlit_app.py`
Expected: App starts. Default theme is Streamlit's light. User can switch to dark via Settings (hamburger menu > Settings > Theme > Dark).

**Step 3: Commit**

```bash
git add .streamlit/config.toml
git commit -m "Remove forced light theme to enable dark mode switching"
```

---

### Task 2: Update highlight_sentiment for theme-aware colors

**Files:**
- Test: `tests/test_streamlit_app.py`
- Modify: `streamlit_app.py:39-45`

**Step 1: Write failing tests for the new signature**

Add these tests to `tests/test_streamlit_app.py`, replacing the existing `TestHighlightSentiment` class (lines 83-106):

```python
class TestHighlightSentiment:
    def test_positive_light_returns_muted_green(self):
        from streamlit_app import highlight_sentiment

        result = highlight_sentiment("positive", dark=False)
        assert "background-color: #d6ecd2" in result

    def test_negative_light_returns_muted_red(self):
        from streamlit_app import highlight_sentiment

        result = highlight_sentiment("negative", dark=False)
        assert "background-color: #f5d0d0" in result

    def test_positive_dark_returns_deep_green(self):
        from streamlit_app import highlight_sentiment

        result = highlight_sentiment("positive", dark=True)
        assert "background-color: #2d4a2d" in result
        assert "color: #e0e0e0" in result

    def test_negative_dark_returns_deep_red(self):
        from streamlit_app import highlight_sentiment

        result = highlight_sentiment("negative", dark=True)
        assert "background-color: #4a2d2d" in result
        assert "color: #e0e0e0" in result

    def test_empty_returns_empty_string(self):
        from streamlit_app import highlight_sentiment

        assert highlight_sentiment("", dark=False) == ""
        assert highlight_sentiment("", dark=True) == ""

    def test_unknown_returns_empty_string(self):
        from streamlit_app import highlight_sentiment

        assert highlight_sentiment("neutral", dark=False) == ""
        assert highlight_sentiment("neutral", dark=True) == ""
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_streamlit_app.py::TestHighlightSentiment -v`
Expected: FAIL — `highlight_sentiment()` does not accept `dark` parameter.

**Step 3: Update highlight_sentiment implementation**

Replace `highlight_sentiment` in `streamlit_app.py` (lines 39-45) with:

```python
def highlight_sentiment(val: str, *, dark: bool = False) -> str:
    """Return CSS for a sentiment value, adapted to light or dark theme."""
    if val == "positive":
        if dark:
            return "background-color: #2d4a2d; color: #e0e0e0"
        return "background-color: #d6ecd2"
    if val == "negative":
        if dark:
            return "background-color: #4a2d2d; color: #e0e0e0"
        return "background-color: #f5d0d0"
    return ""
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_streamlit_app.py::TestHighlightSentiment -v`
Expected: All 6 tests PASS.

**Step 5: Commit**

```bash
git add streamlit_app.py tests/test_streamlit_app.py
git commit -m "Add theme-aware sentiment colors with dark mode support"
```

---

### Task 3: Update UI to detect theme and pass it to styling

**Files:**
- Modify: `streamlit_app.py:223-224` (the `styled = ...` block)

**Step 1: Add theme detection and update the style.map call**

In `streamlit_app.py`, add theme detection near the top of the UI section (after line 104, `device = get_device()`), add:

```python
is_dark = st.get_option("theme.base") == "dark"
```

Then update the `styled = ...` line (line 223-224) to use `functools.partial`:

Add `from functools import partial` to the imports at the top of the file (line 1 area).

Update the styling call from:

```python
                    styled = result_df.style.map(
                        highlight_sentiment, subset=["Sentiment"]
                    )
```

to:

```python
                    styled = result_df.style.map(
                        partial(highlight_sentiment, dark=is_dark),
                        subset=["Sentiment"],
                    )
```

**Step 2: Verify the app runs in both themes**

Run: `uv run streamlit run streamlit_app.py`
Expected: Classify some text. In light mode, sentiment cells show muted pastel green/red. Switch to dark mode (Settings > Theme > Dark), classify again — cells show deep green/red with light text.

**Step 3: Run all tests to verify nothing broke**

Run: `uv run pytest tests/test_streamlit_app.py -v`
Expected: All tests PASS.

**Step 4: Commit**

```bash
git add streamlit_app.py
git commit -m "Detect theme and apply theme-aware sentiment styling"
```

---

### Task 4: Inject comfort CSS for font size and line height

**Files:**
- Modify: `streamlit_app.py` (after `st.set_page_config`, before `device = get_device()`)

**Step 1: Add CSS injection**

After the `st.set_page_config(...)` block (line 102) and before `device = get_device()` (line 104), add:

```python
st.markdown(
    """
    <style>
    .stMainBlockContainer, .stSidebar {
        font-size: 16px;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
```

**Step 2: Verify the app renders with larger text**

Run: `uv run streamlit run streamlit_app.py`
Expected: Text throughout the app (sidebar, main content) is slightly larger and more spaced than before. Works in both light and dark modes.

**Step 3: Run all tests**

Run: `uv run pytest tests/test_streamlit_app.py -v`
Expected: All tests PASS (CSS injection doesn't affect unit tests).

**Step 4: Commit**

```bash
git add streamlit_app.py
git commit -m "Add comfort CSS for larger font size and line height"
```

---

### Task 5: Add sidebar theme hint

**Files:**
- Modify: `streamlit_app.py:116-118` (sidebar section)

**Step 1: Add theme tip to sidebar**

After the existing `st.caption(f"Running on {device.upper()}")` line (line 118), add:

```python
    st.caption("Tip: Change theme in Settings (\u22ee menu)")
```

This goes inside the `with st.sidebar:` block, after the last caption.

**Step 2: Verify it appears**

Run: `uv run streamlit run streamlit_app.py`
Expected: Sidebar shows "Tip: Change theme in Settings (⋮ menu)" at the bottom.

**Step 3: Run all tests**

Run: `uv run pytest tests/test_streamlit_app.py -v`
Expected: All tests PASS.

**Step 4: Commit**

```bash
git add streamlit_app.py
git commit -m "Add sidebar hint about theme toggle"
```

---

### Task 6: Run linting and final verification

**Step 1: Run ruff check and format**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: No lint errors, formatting OK.

**Step 2: Run type checker**

Run: `uv run ty check .`
Expected: No type errors (or only pre-existing ones).

**Step 3: Run full test suite**

Run: `uv run pytest tests/test_streamlit_app.py -v`
Expected: All tests PASS.

**Step 4: Fix any issues found, then commit**

If ruff or ty found issues, fix and commit:

```bash
git add -A
git commit -m "Fix lint and formatting for low-light comfort UI"
```
