# Low-Light Comfort UI Design

## Goal

Make the app comfortable for extended use and low-light environments by enabling dark mode, using muted colors, and improving font readability.

## Approach

Use Streamlit's built-in light/dark theme switching (Approach A). Minimal code, leverages native infrastructure, maintainable.

## Changes

### 1. Theme Configuration (`config.toml`)

Remove forced light theme colors. Keep only `primaryColor` (muted blue) and `font`. Streamlit supplies appropriate background/text colors for each mode automatically.

```toml
[theme]
primaryColor = "#5B9BD5"
font = "sans serif"
```

### 2. Sentiment Colors (`highlight_sentiment`)

Two palettes — muted versions for both modes. Detect theme via `st.get_option("theme.base")`.

- **Light mode**: `#d6ecd2` (green), `#f5d0d0` (red) with dark text
- **Dark mode**: `#2d4a2d` (green), `#4a2d2d` (red) with light text

Pass theme mode into `highlight_sentiment` or use a wrapper. Update `process_dataframe` to detect theme.

### 3. Font & Spacing (CSS injection)

Inject CSS via `st.markdown` at app startup:

- Base font size: 14px -> 16px
- Line height: ~1.4 -> 1.6
- Works in both light and dark modes

### 4. Sidebar Hint

Add caption in sidebar: `"Tip: Change theme in Settings (hamburger menu)"` so users know the toggle exists.
