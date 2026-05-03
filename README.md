# ParlaMint China Analysis Tool

A dashboard that maps how China became one of the most contested topics in Dutch politics — from a barely-mentioned trading partner in 2014 to a recurring subject of debate on human rights, semiconductors, and geopolitical positioning by 2022. The data is the full [ParlaMint-NL](https://www.clarin.eu/parlamint) corpus: 593,961 parliamentary speeches, annotated at sentence level.

---

## What it does

- **How fast did China grow as a topic?** Trend analysis by year, quarter, and party
- **Who drives the debate?** Party comparison normalised for speaking frequency, shown on a parliamentary seat chart
- **What's the tone?** China-specific sentiment per party over time — filtered to sentences that actually mention China, not just overall speech mood
- **Is China discussed alone or in context?** Co-occurrence with the US, Russia, EU, and NATO — and how that shifts the tone
- **Who are the key voices?** Bubble chart of individual speakers by China-focus and institutional influence, with per-speaker speech drill-down
- **What did they actually say?** Full-text search across all China-mentioning speeches

---

## Setup

**Requirements:** Python 3.10+

```bash
pip install pandas pyarrow streamlit plotly anthropic huggingface_hub
```

**Run the app:**
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser. The dataset (~186 MB) is downloaded automatically from HuggingFace on first run.

**Optional — build the dataset yourself** from the raw ParlaMint corpus (13 GB):

1. Download `ParlaMint-NL-en.ana.tgz` from the [ParlaMint repository](https://www.clarin.eu/parlamint) and place it in `data/raw/`
2. Extract China-related debate files:
   ```bash
   python3 scripts/01_extract_subset.py
   ```
3. Parse into a single DataFrame:
   ```bash
   python3 scripts/02_parse_to_df.py
   ```

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
├── .streamlit/
│   └── config.toml           # Theme configuration
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
| AI narrative analysis | Claude (Anthropic) |
| Dataset hosting | HuggingFace Datasets |

---

## Running tests

```bash
python3 scripts/test_parser.py
```

5 unit tests covering China-sentiment isolation and great power co-occurrence counting.
