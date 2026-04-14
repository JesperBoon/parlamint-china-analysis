# Process Log — ParlaMint China Analysis Tool

**HCSS Datalab Assignment | Jesper Boon**
**Corpus:** ParlaMint-NL 2014–2022 | 593,961 speeches

---

## Chronological decision log

A step-by-step record of major decisions made during development, in the order they were taken.

---

**1. Start from a working baseline**
The assignment started with a functioning 7-page Streamlit app and a fully parsed parquet dataset already in place. Decision: extend the existing tool rather than rebuild it, so effort could go into analytical depth instead of infrastructure.

---

**2. Merge Trend and Party pages into one**
The original app had "Trend over time" and "Party comparison" as separate pages. Decision: consolidate them into a single "Party & Sentiment Trends" page with a shared party multiselect and four tabs (Sentiment over time, Mention volume, Party rates, Parliament seats). *Reason:* these analyses share a common filter and users naturally want to move between them without losing their selection.

---

**3. Fix the parliament seat chart — block layout**
The original `horseshoe_coords()` scattered seats across all radial rows globally, mixing parties visually. Decision: rewrite as `horseshoe_coords_grouped()` that assigns each party a proportional angular arc and fills it column-by-column. *Reason:* the real Dutch parliament arranges parties as solid blocks; the scattered layout was misleading.

---

**4. Add animation to the seat chart (political order ↔ sentiment order)**
Rather than a static chart, decision was made to bake two `go.Frame` objects into the figure — one coloured by political spectrum, one re-ordered by descending China sentiment — with toggle buttons. *Reason:* a Streamlit re-render would reset user state; Plotly frames keep the transition smooth and self-contained.

---

**5. Add a policy event timeline overlay**
Decision: define 20 Dutch–China policy milestones (2015–2022) and overlay them as vertical marker lines on trend charts. *Reason:* raw time-series without context make it hard to tell whether parliament is reacting to real events or not.

**Sub-decision — two rendering paths for vlines:** `add_vline(x="2015")` crashes on categorical string x-axes (Plotly bug: internally calls `sum(["2015", "2015"])`). Decision: detect axis type and use `add_shape` + `add_annotation` for string axes, `add_vline` for numeric axes.

**Sub-decision — three x representations per event:** Each milestone stores `year_frac`, `period_Y`, and `period_Q` so the shared `add_policy_lines()` helper works cleanly across all chart types without per-chart conversion logic.

---

**6. Add AI narrative analysis via Anthropic API**
Decision: add a "Policy & Geopolitics → AI" tab that calls Claude Haiku to generate structured narrative summaries. *Reason:* the quantitative charts show *that* China discourse shifts; the AI tab lets users ask *why* in natural language, grounded in the actual speech corpus.

**Sub-decision — four analysis modes:** Topic × Year, Policy event deep-dive (±6 months window), Single party stance, Party comparison. The deep-dive uses a date window rather than topic+year because policy events cut across topic categories.

**Sub-decision — cap token usage:** Single party analysis capped at 60 speeches; deep-dive pool capped at ~400 speeches (~20K tokens). *Reason:* Claude Haiku charges per token; uncapped pooling on popular parties could become expensive in a demo context.

---

**7. Add great power sentiment proxy (no re-parse)**
Adding Germany sentiment required `mentions_germany` — a column that doesn't exist in the parquet and would require a ~2h full XML re-parse. Decision: defer Germany, but add a "Reference comparison" tab using the *existing* boolean columns (`mentions_us`, `mentions_russia`, `mentions_eu`, `mentions_nato`) as filters on `china_sentiment_avg`. *Reason:* delivers the analytical insight (how co-occurrence with other powers shifts China tone) without re-parse cost.

---

## What was already there at the start

The tool launched with a working 7-page Streamlit app and a fully parsed parquet dataset. The baseline included:

- `scripts/01_extract_subset.py` — stream-extracts China-related files from the raw 13 GB `.tgz`
- `scripts/02_parse_to_df.py` — TEI XML → flat parquet ETL pipeline
- `src/analysis.py` — initial set of analytical functions
- `app.py` — 7-page Streamlit dashboard with sidebar filters

Existing analytical functions: `china_trend`, `china_by_party`, `sentiment_heatmap`, `top_china_speakers`, `speaker_speeches`, `great_power_cooccurrence`, `china_power_combinations`, `seat_chart_data`, `china_topic_distribution`.

Existing pages: Overview (static), Trend over time, Party comparison, Sentiment analysis, Great power context, Top speakers, Explore speeches.

---

## What was built this session

### 1. Anthropic API integration (Policy & Geopolitics → AI tab)

**What:** The AI narrative analysis tab allows generating structured analyses using Claude Haiku (`claude-haiku-4-5-20251001`). API key is stored in `.streamlit/secrets.toml` (not committed).

**Four AI modes:**
1. **Topic × Year** — pool all China speeches in a given topic-year, generate a narrative summary
2. **Policy event deep-dive** — pool speeches ±6 months around a specific Dutch–China policy milestone, ask whether parliament *reacts to* or *precedes* events
3. **Single party** — generate a party-specific stance profile on China across all their speeches
4. **Compare parties** — compare two or more parties' China discourse side-by-side

**Design choice:** deep-dive mode uses a date-window pool (±6 months) rather than topic+year, because policy events cut across topic categories.

---

### 2. Party & Sentiment Trends page (consolidated)

**Before:** "Trend over time" and "Party comparison" were separate pages.

**After:** Merged into one page with a shared party multiselect and four tabs:

| Tab | Content |
|-----|---------|
| Sentiment over time | Per-party `china_sentiment_avg` trend lines, year resolution, policy milestone overlay |
| Mention volume | Per-party China speech count / %, Year or Quarter toggle, count/% toggle, policy milestone overlay |
| Party rates | Normalised China mention rates for all parties (% of each party's own speeches) |
| Parliament seats | Horseshoe chart of seat composition, Year selector, Political order ↔ Sentiment order animation |

**New analysis function added:** `party_sentiment_trend(df, parties, top_n)` in `analysis.py` — long-format per-party per-year china sentiment averages.

`china_trend_by_party()` updated to accept an optional `parties` parameter and to include a `pct` column.

---

### 3. Parliament seat chart — clustered party blocks

**Problem:** The original `horseshoe_coords()` distributed seats row-by-row globally. This scattered parties across all radial rows, so PVV seats ended up mixed with VVD seats.

**Solution:** New `horseshoe_coords_grouped(party_sizes)` in `app.py`:
- Allocates an angular arc segment proportional to each party's seat count
- Fills that segment column-by-column (not row-by-row globally)
- Result: each party appears as a solid block, matching the real Dutch parliament layout

**Year selector:** `st.select_slider` for 2015–2022 filters the underlying `df` before calling `an.seat_chart_data()`, so the correct election-period seat map is selected dynamically.

**Animation:** Two `go.Frame` objects baked into the figure:
- Frame 1 (default): seats coloured by political spectrum (Left/Center/Right)
- Frame 2: seats re-ordered by descending `mean_china_sentiment`, coloured by sentiment score
- "Re-seat by China stance →" and "← Political order" toggle buttons

---

### 4. Great power comparison — sentiment proxy tab

**Problem:** No `mentions_germany` column in the parquet — adding Germany would require re-parsing all XML. Deferred.

**Added:** 5th tab "Reference comparison" on the Sentiment analysis page.

**Two new analysis functions:**

`power_sentiment_proxy(df)` — answers "How negative is parliament when discussing China *in the context of* [US / Russia / EU / NATO]?" Returns `[power, avg_sentiment, n_speeches]`. No re-parse needed — uses existing `mentions_us/russia/eu/nato` boolean columns as a filter on `china_sentiment_avg`.

`power_sentiment_trend(df)` — same split per year, returns `[year, power, avg_sentiment, n_speeches]`.

**Findings (as of the full corpus):**
- China alone: 1.500 avg sentiment
- China + US: 1.609
- China + EU: 1.623
- China + Russia: 1.389 (most negative context)

---

### 5. Policy event timeline

**What:** 20 Dutch–China policy milestones (2015–2022) overlaid as vertical marker lines on relevant charts.

**Data:** Each event has `date`, `year_frac` (float for numeric axes), `period_Y` (string for yearly period axes), `period_Q` (string for quarterly axes), `label`, `category`, `short`.

**Five categories with distinct colours:**
| Category | Colour |
|----------|--------|
| Human Rights | `#C0392B` red |
| Trade & Economy | `#E67E22` orange |
| Technology | `#2980B9` blue |
| Security | `#8E44AD` purple |
| Diplomacy | `#27AE60` green |

**`add_policy_lines(fig, x_type, events, selected)`** — shared helper that injects `add_vline()` (numeric axes) or `add_shape()` + `add_annotation()` (string/categorical axes) onto any Plotly figure.

**Bug fixed during implementation:** `add_vline(x="2015")` crashes on categorical string x-axes because Plotly internally calls `sum(["2015", "2015"])` → `TypeError: unsupported operand type(s) for +: 'int' and 'str'`. Fixed by using `add_shape` + `add_annotation` when `x_val` is a string.

**Milestone overlays appear on:** Chapter 1 trend bar (Overview), Sentiment over time tab (Party & Sentiment), Mention volume tab (Party & Sentiment), Topic over time chart (Policy & Geopolitics), Sentiment per topic chart (Policy & Geopolitics).

---

## Known limitations

| Limitation | Detail |
|-----------|--------|
| No Germany column | `mentions_germany` doesn't exist — would require full XML re-parse (~2h runtime). Excluded from great power comparison. |
| Sentiment coverage ~0.4% | `china_sentiment_avg` is non-null for only ~2,600 speeches out of 593,961. Most speeches don't have a China-specific sentence with a ParlaSent score. Use `n_speeches` metric to assess reliability per analysis. |
| Scrollytelling without CSS | True scroll-triggered animation requires JavaScript. Implemented as Plotly `go.Frame` animation with toggle buttons instead. |
| ParlaSent scale interpretation | Scores are 0–5 (not -1 to +1). Scale midpoint 2.5 = neutral. Values below 2.0 are unambiguously negative. The corpus average is ~1.4–1.6 across all China speeches. |
| AI token usage | Claude Haiku charges per token. Deep-dive pool (±6 months, up to 400 speeches) can approach 20K tokens per call. Single party analysis caps at 60 speeches to control costs. |
| Party label normalisation | `party.PVV` → `PVV`, `ministry.EZK` → `min.EZK`. Ministers and chairs are filtered out in most analyses via `parties_only()`. Some edge cases (party leaders serving as minister) may briefly show minister label. |

---

## Architecture decisions

**Why parquet?** 656 MB flat parquet loads in ~3 seconds with `@st.cache_data`. CSV equivalent is 1.2 GB and loads in 15s. Columnar format also makes party/year groupby significantly faster.

**Why sentence-level China filtering for sentiment?** Speech-level `sentiment_avg` reflects the general mood of a debate contribution, not attitude toward China specifically. A speech criticising China's belt-and-road policy might have an overall neutral tone because the speaker also discusses EU alternatives positively. Filtering to China-sentences gives a more targeted signal at the cost of coverage.

**Why `year_frac` + `period_Y` + `period_Q` on each policy event?** Charts use different x-axis types depending on their granularity. A unified event object with three x representations avoids repeated conversions and lets `add_policy_lines` work cleanly across all chart types.

---

## File structure (post-session)

```
parlamint-china/
├── app.py                    # ~1850 lines — Streamlit app (7 pages)
├── src/
│   └── analysis.py           # ~560 lines — all analytical computations
├── scripts/
│   ├── 01_extract_subset.py  # stream-extract China files from .tgz
│   ├── 02_parse_to_df.py     # TEI XML → parquet ETL
│   ├── dry_run_lemmas.py     # lemma hit-count preview (pre-parse check)
│   └── test_parser.py        # 5 unit tests for parser logic
├── assets/
│   └── hcss_logo.png
├── .streamlit/
│   ├── config.toml           # HCSS house style (colours, font)
│   └── secrets.toml          # ANTHROPIC_API_KEY (gitignored)
├── process_log.md            # this file
├── requirements.txt
└── data/                     # gitignored — generate locally
    ├── raw/                  # source .tgz (13 GB)
    ├── subset/               # extracted China-related XML files
    └── processed/            # speeches.parquet + speeches.csv
```
