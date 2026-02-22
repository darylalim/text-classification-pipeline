# Test CSV Replacement Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace winter-gear-only test CSVs with domain-specific files covering text types SiEBERT excels at (product reviews, movie reviews, social media, restaurant reviews, app reviews).

**Architecture:** Delete 2 old CSV files, create 6 new ones (5 domain files at 40 rows each + 1 mixed sample at 20 rows). All files have a single `text` column. Update CLAUDE.md references.

**Tech Stack:** CSV files, pandas (for verification)

**Design doc:** `docs/plans/2026-02-22-test-csv-replacement-design.md`

---

### Content Guidelines (applies to all CSV tasks)

Every domain file must:
- Have exactly 40 rows (not counting the header)
- Have a single column: `text`
- Contain ~20 positive and ~20 negative texts
- Mix short (1 sentence) and longer (2-4 sentences) texts
- Include 2-3 edge cases per file: sarcasm, mixed sentiment, mild opinions
- Use realistic, natural-sounding language — not obviously AI-generated
- Properly quote any text containing commas with double quotes
- No blank or empty rows

---

### Task 1: Delete old CSV files

**Files:**
- Delete: `tests/data/csv/customer_reviews.csv`
- Delete: `tests/data/csv/customer_reviews_sample.csv`

**Step 1: Delete the files**

```bash
rm tests/data/csv/customer_reviews.csv tests/data/csv/customer_reviews_sample.csv
```

**Step 2: Verify deletion**

```bash
ls tests/data/csv/
```
Expected: empty directory (or only new files if later tasks ran first)

**Step 3: Commit**

```bash
git add -u tests/data/csv/
git commit -m "chore: remove old winter-gear test CSV files"
```

---

### Task 2: Create product_reviews.csv

**Files:**
- Create: `tests/data/csv/product_reviews.csv`

**Step 1: Write the CSV file**

Create `tests/data/csv/product_reviews.csv` with 40 rows of e-commerce product reviews. Single column `text`. Cover categories: electronics, clothing, home goods, kitchen appliances, personal care, fitness equipment, office supplies.

Example rows showing the expected tone/length variety:

```
text
"Absolutely love these headphones. The noise cancellation is incredible and battery life lasts all week."
"The blender broke after two uses. Cheap plastic gears inside, total waste of money."
"It's fine. Does what it says but nothing special for the price."
"Bought this jacket for hiking and it kept me completely dry through a full day of rain. The zippers are solid and pockets are well-placed. My only gripe is the hood is a bit tight over a beanie."
"DO NOT BUY. The seams started coming apart on the first wash and customer service ghosted me."
"The standing desk is sturdy and the motor is whisper quiet, but the cable management tray feels like an afterthought."
```

**Step 2: Verify the CSV is well-formed**

```bash
python -c "import pandas as pd; df = pd.read_csv('tests/data/csv/product_reviews.csv'); print(f'Rows: {len(df)}, Columns: {list(df.columns)}'); assert len(df) == 40; assert list(df.columns) == ['text']"
```
Expected: `Rows: 40, Columns: ['text']`

**Step 3: Commit**

```bash
git add tests/data/csv/product_reviews.csv
git commit -m "chore: add product reviews test CSV (40 rows)"
```

---

### Task 3: Create movie_reviews.csv

**Files:**
- Create: `tests/data/csv/movie_reviews.csv`

**Step 1: Write the CSV file**

Create `tests/data/csv/movie_reviews.csv` with 40 rows of film and TV opinions. Cover: blockbusters, indie films, documentaries, TV series, streaming originals, animated films, classic films.

Example rows:

```
text
"One of the best thrillers I've seen in years. The pacing keeps you on edge from start to finish."
"Terrible screenplay. The dialogue felt like it was written by someone who has never had a real conversation."
"The cinematography was gorgeous but the plot made absolutely no sense by the third act."
"Finally a superhero movie that takes itself seriously without being pretentious about it. Great casting, great score."
"I walked out halfway through. Life is too short for movies this boring."
"Decent enough for a Sunday afternoon. Not the best rom-com but the leads have good chemistry."
```

**Step 2: Verify the CSV is well-formed**

```bash
python -c "import pandas as pd; df = pd.read_csv('tests/data/csv/movie_reviews.csv'); print(f'Rows: {len(df)}, Columns: {list(df.columns)}'); assert len(df) == 40; assert list(df.columns) == ['text']"
```
Expected: `Rows: 40, Columns: ['text']`

**Step 3: Commit**

```bash
git add tests/data/csv/movie_reviews.csv
git commit -m "chore: add movie reviews test CSV (40 rows)"
```

---

### Task 4: Create social_media.csv

**Files:**
- Create: `tests/data/csv/social_media.csv`

**Step 1: Write the CSV file**

Create `tests/data/csv/social_media.csv` with 40 rows of tweets and social media posts. These should be shorter and more informal than reviews. Cover: brand mentions, event reactions, customer complaints, product praise, general opinions, news reactions.

Example rows:

```
text
just got the new pixel and wow the camera is insane
"worst customer service experience of my life, been on hold for 2 hours @BigTelco"
"ngl this coffee shop on 5th street might be the best kept secret in the city"
"Tried the new burger place everyone's been talking about. Massively overrated."
"so happy with my new apartment!! finally have a dishwasher lol"
"another update, another round of bugs. thanks for nothing @AppDev"
```

**Step 2: Verify the CSV is well-formed**

```bash
python -c "import pandas as pd; df = pd.read_csv('tests/data/csv/social_media.csv'); print(f'Rows: {len(df)}, Columns: {list(df.columns)}'); assert len(df) == 40; assert list(df.columns) == ['text']"
```
Expected: `Rows: 40, Columns: ['text']`

**Step 3: Commit**

```bash
git add tests/data/csv/social_media.csv
git commit -m "chore: add social media test CSV (40 rows)"
```

---

### Task 5: Create restaurant_reviews.csv

**Files:**
- Create: `tests/data/csv/restaurant_reviews.csv`

**Step 1: Write the CSV file**

Create `tests/data/csv/restaurant_reviews.csv` with 40 rows of dining and food service feedback. Cover: fine dining, casual restaurants, fast food, cafes, takeout/delivery, bars, bakeries.

Example rows:

```
text
"The tasting menu was extraordinary. Every course was a surprise and the wine pairings were spot on."
"Cold food, rude waitstaff, and a 45-minute wait for a table we had reserved. Never going back."
"Solid neighborhood pizza joint. Nothing fancy but the crust is perfect and prices are fair."
"We celebrated our anniversary here and it was magical. The candlelit patio, the fresh pasta, the tiramisu. Worth every penny."
"The sushi looked beautiful but tasted like it had been sitting out. Style over substance."
"Good brunch spot if you don't mind waiting. The french toast is legit but service is painfully slow on weekends."
```

**Step 2: Verify the CSV is well-formed**

```bash
python -c "import pandas as pd; df = pd.read_csv('tests/data/csv/restaurant_reviews.csv'); print(f'Rows: {len(df)}, Columns: {list(df.columns)}'); assert len(df) == 40; assert list(df.columns) == ['text']"
```
Expected: `Rows: 40, Columns: ['text']`

**Step 3: Commit**

```bash
git add tests/data/csv/restaurant_reviews.csv
git commit -m "chore: add restaurant reviews test CSV (40 rows)"
```

---

### Task 6: Create app_reviews.csv

**Files:**
- Create: `tests/data/csv/app_reviews.csv`

**Step 1: Write the CSV file**

Create `tests/data/csv/app_reviews.csv` with 40 rows of mobile/web app store reviews. Cover: productivity apps, social media apps, fitness/health, banking/finance, games, streaming, navigation, food delivery.

Example rows:

```
text
"Best task manager I've ever used. Syncs perfectly across all my devices and the UI is clean."
"App crashes every time I try to upload a photo. Three updates and the bug is still there."
"It's okay for basic budgeting but the premium tier is overpriced for what you get."
"This meditation app genuinely changed my sleep habits. The guided sessions are calming and well-paced. Been using it daily for six months."
"Used to love this app but the latest redesign is awful. Everything takes twice as many taps now."
"Great GPS accuracy and offline maps are a lifesaver on road trips."
```

**Step 2: Verify the CSV is well-formed**

```bash
python -c "import pandas as pd; df = pd.read_csv('tests/data/csv/app_reviews.csv'); print(f'Rows: {len(df)}, Columns: {list(df.columns)}'); assert len(df) == 40; assert list(df.columns) == ['text']"
```
Expected: `Rows: 40, Columns: ['text']`

**Step 3: Commit**

```bash
git add tests/data/csv/app_reviews.csv
git commit -m "chore: add app reviews test CSV (40 rows)"
```

---

### Task 7: Create mixed_sample.csv

**Files:**
- Create: `tests/data/csv/mixed_sample.csv`

**Step 1: Write the CSV file**

Create `tests/data/csv/mixed_sample.csv` with exactly 20 rows — 4 rows copied from each of the 5 domain files (2 positive, 2 negative from each). Pick rows that represent the variety within each domain (mix of short and long).

**Step 2: Verify the CSV is well-formed**

```bash
python -c "import pandas as pd; df = pd.read_csv('tests/data/csv/mixed_sample.csv'); print(f'Rows: {len(df)}, Columns: {list(df.columns)}'); assert len(df) == 20; assert list(df.columns) == ['text']"
```
Expected: `Rows: 20, Columns: ['text']`

**Step 3: Commit**

```bash
git add tests/data/csv/mixed_sample.csv
git commit -m "chore: add mixed sample test CSV (20 rows)"
```

---

### Task 8: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md:56-58` (the Tests section, lines referencing old CSV files)

**Step 1: Update test data references**

Replace the two bullet points referencing old files:

```markdown
- `tests/data/csv/customer_reviews.csv` — 100 reviews (PRODUCT, DATE, SUMMARY, SENTIMENT_SCORE, Order ID)
- `tests/data/csv/customer_reviews_sample.csv` — 11-row subset for quick testing
```

With:

```markdown
- `tests/data/csv/product_reviews.csv` — 40 e-commerce product reviews
- `tests/data/csv/movie_reviews.csv` — 40 film and TV opinions
- `tests/data/csv/social_media.csv` — 40 tweets and social media posts
- `tests/data/csv/restaurant_reviews.csv` — 40 dining and food service reviews
- `tests/data/csv/app_reviews.csv` — 40 mobile/web app store reviews
- `tests/data/csv/mixed_sample.csv` — 20-row sample (4 from each domain) for quick testing
```

**Step 2: Verify no stale references remain**

```bash
grep -r "customer_reviews" CLAUDE.md tests/
```
Expected: no output (no remaining references to old filenames)

**Step 3: Run existing tests to verify nothing broke**

```bash
uv run pytest tests/test_streamlit_app.py -v
```
Expected: all tests pass (they don't reference CSV files)

**Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md test data references for new CSV files"
```

---

### Task 9: Final verification

**Step 1: Verify all 6 CSV files exist and are well-formed**

```bash
python -c "
import pandas as pd
import os

csv_dir = 'tests/data/csv'
expected = {
    'product_reviews.csv': 40,
    'movie_reviews.csv': 40,
    'social_media.csv': 40,
    'restaurant_reviews.csv': 40,
    'app_reviews.csv': 40,
    'mixed_sample.csv': 20,
}

for fname, expected_rows in expected.items():
    path = os.path.join(csv_dir, fname)
    df = pd.read_csv(path)
    assert list(df.columns) == ['text'], f'{fname}: wrong columns {list(df.columns)}'
    assert len(df) == expected_rows, f'{fname}: expected {expected_rows} rows, got {len(df)}'
    assert df['text'].notna().all(), f'{fname}: has NaN values'
    assert (df['text'].str.strip() != '').all(), f'{fname}: has blank rows'
    print(f'{fname}: OK ({len(df)} rows)')

print('All files verified.')
"
```

**Step 2: Verify old files are gone**

```bash
ls tests/data/csv/
```
Expected: only the 6 new files, no `customer_reviews*` files

**Step 3: Run full test suite**

```bash
uv run pytest -v
```
Expected: all tests pass

**Step 4: Run linter**

```bash
uv run ruff check .
```
Expected: no issues
