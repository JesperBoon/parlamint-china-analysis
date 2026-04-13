# ParlaMint China Analysis Tool

**HCSS Datalab Assignment** — An interactive tool for analysing how China is discussed in Dutch parliamentary debates (2014–2022).

Built on the [ParlaMint-NL](https://www.clarin.eu/parlamint) corpus: 593,961 speeches, linguistically annotated at sentence level.

---

## What it does

- **Trend analysis** — how China mentions have grown over time, by year/quarter/month and per party
- **Party comparison** — which parties discuss China most, normalised for speaking frequency, visualised on a parliamentary seat chart
- **Sentiment analysis** — animated year-by-year view of each party's tone toward China (China-specific sentiment, not overall speech mood)
- **Great power context** — in what combinations is China mentioned with the US, Russia, EU, and NATO — and how does that shift the tone?
- **Top speakers** — bubble chart of individual speakers by China-focus and sentiment, with per-speaker speech drill-down
- **Explore speeches** — full-text search across all China-mentioning speeches

---

## Setup

**Requirements:** Python 3.10+

```bash
pip install pandas pyarrow streamlit plotly
```

**Data** (not included in repo — 13 GB):

1. Download `ParlaMint-NL-en.ana.tgz` from the [ParlaMint repository](https://www.clarin.eu/parlamint) and place it in `data/raw/`
2. Extract China-related debate files:
   ```bash
   python3 scripts/01_extract_subset.py
   ```
3. Parse into a single DataFrame:
   ```bash
   python3 scripts/02_parse_to_df.py
   ```
   This produces `data/processed/speeches.parquet` (~656 MB, 593,961 rows).

**Run the app:**
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Project structure

```
parlamint-china/
├── app.py                    # Streamlit application (7 pages)
├── src/
│   └── analysis.py           # All analytical computations
├── scripts/
│   ├── 01_extract_subset.py  # Stream-extract China-related files from .tgz
│   ├── 02_parse_to_df.py     # TEI XML → parquet ETL pipeline
│   ├── dry_run_lemmas.py     # Lemma hit-count preview (pre-parse check)
│   └── test_parser.py        # Unit tests for parser logic
├── assets/
│   └── hcss_logo.png
├── .streamlit/
│   └── config.toml           # HCSS house style theme
└── data/                     # Gitignored — generate locally
    ├── raw/                  # Place source .tgz here
    ├── subset/               # Extracted China-related XML files
    └── processed/            # speeches.parquet + speeches.csv
```

---

## Methodology notes

**China detection** uses lemma-level matching (not substring) on 31 single-token terms and 8 multi-word phrases, including city names (Shanghai, Wuhan), political terms (Belt and Road, One China), human rights terms (Uyghur, Falun Gong), and tech companies (Alibaba, TikTok, Huawei).

**China-specific sentiment** (`china_sentiment_avg`) is computed from sentence-level ParlaSent scores (XLM-R, 0–5 scale) filtered to only sentences that contain a China-related lemma. This is methodologically distinct from `sentiment_avg`, which reflects the overall speech tone. Coverage: ~2,600 speeches.

**Speaker role resolution** matches each speech to the speaker's active affiliation at the speech date (from `listPerson.xml`), correctly attributing speeches to party membership vs. ministerial roles.

**Parliament seat chart** uses election-period compositions (2017 TK, 2021 TK, 2019 EK) dynamically selected based on the active date filter.

---

## Tech stack

| Layer | Technology |
|---|---|
| Data processing | Python, pandas, pyarrow |
| XML parsing | xml.etree (standard library) |
| Visualisation | Plotly Express + Graph Objects |
| Interface | Streamlit |
| Storage | Apache Parquet |

---

## Running tests

```bash
python3 scripts/test_parser.py
```

5 unit tests covering China-sentiment isolation and great power co-occurrence counting.
