"""
app.py — ParlaMint China Analysis Tool
HCSS Datalab | Dutch Parliamentary Debates on China (2015–2022)

Run with: streamlit run app.py
"""

# ── App-wide language: English ────────────────────────────────────────────────

import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import analysis as an

HCSS_PRIMARY = "#003082"
HCSS_ACCENT = "#0066CC"
HCSS_PALETTE = ["#003082", "#0066CC", "#5A8FD6", "#1A1A1A", "#7F8C8D", "#A6BDDB"]

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="China in Dutch Parliament",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "processed", "speeches.parquet")

# ── Data loading (cached) ──────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    return an.load(DATA_PATH)

# ── Sidebar ────────────────────────────────────────────────────────────────────
LOGO_PATH = os.path.join(os.path.dirname(__file__), "assets", "hcss_logo.png")

with st.sidebar:
    if os.path.exists(LOGO_PATH):
        # narrow column wrapper → browser-native scaling = crisp on retina
        col_a, col_b, col_c = st.columns([1, 2, 1])
        with col_b:
            st.image(LOGO_PATH, use_container_width=True)
    st.title("HCSS Tool")
    st.caption("How China is discussed in the Dutch Parliament")
    st.caption("ParlaMint-NL | 2015–2022")
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Trend over time",
            "Party comparison",
            "Sentiment analysis",
            "Great power context",
            "Top speakers",
            "Explore speeches",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption(
        "**Sentiment note:** The sentiment analyses on this tool use "
        "*China-specific sentiment* — computed only from sentences that "
        "explicitly mention China. Overall speech tone (`sentiment_label`) "
        "appears only in the distribution chart as a reference."
    )

# ── Load data ──────────────────────────────────────────────────────────────────
if not os.path.exists(DATA_PATH):
    st.error(
        f"Dataset not found at `{DATA_PATH}`. "
        "Run `python3 scripts/02_parse_to_df.py` first."
    )
    st.stop()

df = load_data()

# ── Shared filters (shown on all pages except Overview) ───────────────────────
if page != "Overview":
    with st.sidebar:
        st.subheader("Filters")
        year_min, year_max = int(df["year"].min()), int(df["year"].max())
        year_range = st.slider("Year range", year_min, year_max, (year_min, year_max))
        df = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])]

        chamber_options = ["All"] + sorted(df["chamber"].dropna().unique().tolist())
        chamber = st.selectbox("Chamber", chamber_options)
        if chamber != "All":
            df = df[df["chamber"] == chamber]

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Overview
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    col_logo, col_title = st.columns([1, 8])
    with col_logo:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, use_container_width=True)
    with col_title:
        st.title("HCSS Tool — How China is discussed in the Dutch Parliament")
    st.markdown(
        "This tool analyses **ParlaMint-NL** — a corpus of Dutch parliamentary "
        "debates from 2015 to 2022 — through the lens of *China in a Changing "
        "World Order*, an HCSS research pillar."
    )
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    china_df = df[df["china_mentions"] > 0]
    c1.metric("Total speeches", f"{len(df):,}")
    c2.metric("China-mentioning speeches", f"{len(china_df):,}")
    c3.metric("Unique speakers (China)", china_df["speaker_name"].nunique())
    c4.metric("Years covered", f"{df['year'].min()}–{df['year'].max()}")

    st.divider()
    st.subheader("China mentions per year")
    trend = an.china_trend(df, freq="Y")
    fig = px.bar(
        trend, x="period", y="china_speeches",
        labels={"period": "Year", "china_speeches": "Speeches mentioning China"},
        color_discrete_sequence=[HCSS_PRIMARY],
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Trend over time
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Trend over time":
    st.title("China mentions over time")

    freq = st.radio("Granularity", ["Year", "Quarter", "Month"], horizontal=True)
    freq_map = {"Year": "Y", "Quarter": "Q", "Month": "M"}
    trend = an.china_trend(df, freq=freq_map[freq])

    tab1, tab2 = st.tabs(["Absolute count", "As % of all speeches"])

    with tab1:
        fig = px.line(
            trend, x="period", y="china_speeches",
            markers=True,
            labels={"period": "", "china_speeches": "Speeches mentioning China"},
            color_discrete_sequence=[HCSS_PRIMARY],
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = px.line(
            trend, x="period", y="pct",
            markers=True,
            labels={"period": "", "pct": "% of all speeches"},
            color_discrete_sequence=[HCSS_ACCENT],
        )
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Party comparison
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Party comparison":
    st.title("Which parties talk about China?")

    by_party = an.china_by_party(df, top_n=50)

    tab1, tab2 = st.tabs(["Total speeches", "Normalised rate (%)"])
    st.caption(
        "**Normalised rate** = % of that party's *own* speeches that mention China. "
        "It corrects for the fact that some parties simply speak more often than others. "
        "It does **not** weight by parliamentary seats or speaking time."
    )

    with tab1:
        fig = px.bar(
            by_party.sort_values("china_speeches"),
            x="china_speeches", y="party", orientation="h",
            labels={"china_speeches": "Speeches mentioning China", "party": ""},
            color_discrete_sequence=[HCSS_PRIMARY],
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = px.bar(
            by_party.sort_values("rate"),
            x="rate", y="party", orientation="h",
            labels={"rate": "% of party's speeches mentioning China", "party": ""},
            color_discrete_sequence=[HCSS_ACCENT],
        )
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Sentiment analysis
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Sentiment analysis":
    st.title("Tone toward China")
    st.info(
        "**China-specific sentiment** is computed from sentences that explicitly "
        "mention China — so it captures attitude *toward China*, not just the general "
        "mood of the speech. Coverage: ~2,400 speeches (0.4% of all, 50% of China-mentioning "
        "speeches with sentiment data). The distribution tab shows overall `sentiment_label` for context.",
        icon="ℹ️",
    )

    tab1, tab2, tab3 = st.tabs(["Party heatmap", "Left vs Right over time", "Distribution"])

    with tab1:
        st.subheader("Mean China-specific sentiment per party per year")
        st.caption("Scale: 0 = very negative, 5 = very positive | Blank = insufficient data")
        heatmap = an.sentiment_heatmap(df)
        if not heatmap.empty:
            fig = px.imshow(
                heatmap,
                color_continuous_scale="RdYlGn",
                zmin=0, zmax=5,
                aspect="auto",
                labels={"color": "Sentiment score"},
            )
            fig.update_layout(xaxis_title="Year", yaxis_title="Party")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough data with current filters.")

    with tab2:
        st.subheader("Left vs Right China sentiment over time")
        bloc_df = an.sentiment_by_bloc(df)
        if not bloc_df.empty:
            fig = px.line(
                bloc_df, x="year", y="mean_china_sentiment",
                color="bloc",
                markers=True,
                color_discrete_map={"Left": "#C0392B", "Center": "#7F8C8D", "Right": HCSS_PRIMARY},
                labels={"mean_china_sentiment": "Mean China sentiment", "year": "Year"},
            )
            fig.add_hline(y=2.5, line_dash="dot", line_color="grey",
                          annotation_text="Neutral threshold")
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Distribution of sentiment labels")
        china_df = df[df["china_mentions"] > 0]
        dist = china_df["sentiment_label"].value_counts().reset_index()
        dist.columns = ["label", "count"]
        order = ["negneg", "mixneg", "neuneg", "neupos", "mixpos", "pospos"]
        dist["label"] = pd.Categorical(dist["label"], categories=order, ordered=True)
        dist = dist.sort_values("label")
        fig = px.bar(
            dist, x="label", y="count",
            color="label",
            color_discrete_sequence=px.colors.diverging.RdYlGn,
            labels={"label": "Sentiment", "count": "Number of speeches"},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Great power context
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Great power context":
    st.title("China alongside other great powers")
    st.markdown(
        "In speeches that mention China — how often are other great powers "
        "mentioned in the same speech? This reveals whether China is discussed "
        "in isolation or framed against geopolitical alternatives."
    )

    cooc = an.great_power_cooccurrence(df)
    if not cooc.empty:
        fig = px.line(
            cooc, x="year", y="cooccurrence_pct",
            color="power",
            markers=True,
            labels={"cooccurrence_pct": "% of China speeches also mentioning", "year": "Year"},
            color_discrete_map={"US": HCSS_PRIMARY, "RUSSIA": "#C0392B",
                                 "EU": HCSS_ACCENT, "NATO": "#1A1A1A"},
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Raw co-occurrence table")
        st.dataframe(cooc, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Top speakers
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Top speakers":
    st.title("Most active speakers on China")

    speakers = an.top_china_speakers(df, top_n=20)

    fig = px.bar(
        speakers.sort_values("total_china_mentions"),
        x="total_china_mentions", y="speaker_name",
        color="party", orientation="h",
        hover_data=["avg_china_sentiment", "china_speeches"],
        labels={"total_china_mentions": "Total China mentions", "speaker_name": ""},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Speaker table")
    st.dataframe(speakers, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Explore speeches
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Explore speeches":
    st.title("Explore speeches")

    col1, col2 = st.columns(2)
    with col1:
        party_opts = ["All"] + sorted(an.parties_only(df)["party"].dropna().unique().tolist())
        sel_party = st.selectbox("Party", party_opts)
    with col2:
        topic_opts = ["All"] + sorted(df["topic"].dropna().unique().tolist())
        sel_topic = st.selectbox("Topic", topic_opts)

    search = st.text_input("Search in speech text", placeholder="e.g. Huawei, Belt and Road, Taiwan")

    result = df[df["china_mentions"] > 0].copy()
    if sel_party != "All":
        result = result[result["party"] == sel_party]
    if sel_topic != "All":
        result = result[result["topic"] == sel_topic]
    if search:
        result = result[result["text"].str.contains(search, case=False, na=False)]

    st.caption(f"{len(result):,} speeches match your filters")

    display_cols = ["date", "speaker_name", "party", "topic",
                    "sentiment_label", "china_sentiment_avg", "china_mentions", "text"]
    st.dataframe(
        result[display_cols].sort_values("date", ascending=False).head(200),
        use_container_width=True,
    )
