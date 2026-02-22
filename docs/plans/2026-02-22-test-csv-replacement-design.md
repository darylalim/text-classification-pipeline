# Test CSV Replacement Design

## Goal

Replace the two existing winter-gear-only test CSV files with multiple domain-specific files covering the variety of English text that SiEBERT handles well.

## File Structure

All files live in `tests/data/csv/`. Single column: `text`.

| File | Rows | Content |
|---|---|---|
| `product_reviews.csv` | 40 | E-commerce reviews (electronics, clothing, home goods) |
| `movie_reviews.csv` | 40 | Film and TV opinions |
| `social_media.csv` | 40 | Tweets, social posts, short-form opinions |
| `restaurant_reviews.csv` | 40 | Dining and food service feedback |
| `app_reviews.csv` | 40 | Mobile/web app store reviews |
| `mixed_sample.csv` | 20 | 4 rows from each domain for quick testing |

Total: 200 domain rows + 20 sample rows.

## Content Guidelines

- ~50/50 positive/negative split per file
- Mix of short (1 sentence) and longer (2-4 sentences) texts
- Include edge cases: mixed sentiment, sarcasm, mild/strong opinions
- Realistic, natural-sounding text
- No blank/empty rows (covered by unit tests)

## Files to Delete

- `tests/data/csv/customer_reviews.csv`
- `tests/data/csv/customer_reviews_sample.csv`

## Changes to CLAUDE.md

Update test data references from old filenames to new filenames.

## No Test Changes

Unit tests in `test_streamlit_app.py` use inline DataFrames and don't reference CSV files.
