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
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import analysis as an


def horseshoe_coords(n_seats: int, n_rows: int = 5):
    """Compute (x, y) for n_seats arranged in a parliamentary horseshoe."""
    radii = [1.0 + i * 0.18 for i in range(n_rows)]
    total_w = sum(radii)
    seats_per_row = [max(1, round(n_seats * r / total_w)) for r in radii]
    diff = n_seats - sum(seats_per_row)
    seats_per_row[-1] += diff
    coords = []
    for r, n in zip(radii, seats_per_row):
        for i in range(n):
            angle = math.pi - (i + 0.5) * math.pi / n
            coords.append((r * math.cos(angle), r * math.sin(angle)))
    return coords

SPECTRUM_COLORS = {
    "Left": "#D62828",    # Red
    "Center": "#F77F00",  # Orange
    "Right": "#023E8A",   # Blue
}
SPECTRUM_PARTY_COLORS = {
    # Left
    "SP": "#C91A1A", "PvdA": "#E03A3A", "GroenLinks": "#67A32E",
    "PvdD": "#3A7D44", "DENK": "#8B6914", "BIJ1": "#5A3D8A",
    "Volt": "#7B2FBE", "OSF": "#F4A460",
    # Center
    "D66": "#FCBF49", "ChristenUnie": "#A07840", "NSC": "#2D4A6E",
    # Right
    "CDA": "#D4920D", "VVD": "#0B5EA8", "PVV": "#E8721A",
    "FvD": "#BF3D11", "SGP": "#5B2D8E", "JA21": "#D05A14",
    "50PLUS": "#C8A200", "BBB": "#228B22",
}
SPEC_DEFAULT = "#AAAAAA"

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

        # Speaker context filter: distinguishes role at time of speaking
        # (party member vs minister vs chamber chair, etc.)
        ctx_label = {
            "": "Party member (MP)",
            "head": "Chamber chair",
            "minister": "Minister",
            "secretaryOfState": "Secretary of State",
            "deputyHead": "Deputy chair",
        }
        ctx_present = sorted(df["speaker_context"].dropna().unique().tolist()) if "speaker_context" in df.columns else []
        ctx_options = [ctx_label.get(c, c or "Party member (MP)") for c in ctx_present]
        if ctx_options:
            ctx_chosen_labels = st.multiselect(
                "Speaker role",
                ctx_options,
                default=ctx_options,
                help="Filter by the role the speaker held at the time of the speech.",
            )
            reverse_label = {v: k for k, v in ctx_label.items()}
            ctx_chosen_keys = [reverse_label.get(l, l) for l in ctx_chosen_labels]
            if "speaker_context" in df.columns:
                df = df[df["speaker_context"].isin(ctx_chosen_keys)]

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

    tab1, tab2, tab3 = st.tabs(["Absolute count", "As % of all speeches", "By party"])

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

    with tab3:
        st.caption(
            "When did China become a topic *for each party*? "
            "Showing the 8 parties with the most China-mentioning speeches."
        )
        by_party_trend = an.china_trend_by_party(df, freq=freq_map[freq])
        if by_party_trend.empty:
            st.warning("No China-mentioning speeches under current filters.")
        else:
            fig = px.line(
                by_party_trend, x="period", y="china_speeches",
                color="party", markers=True,
                labels={"period": "", "china_speeches": "Speeches mentioning China",
                        "party": "Party"},
                color_discrete_sequence=HCSS_PALETTE + ["#5DADE2", "#229954"],
            )
            st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Party comparison
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Party comparison":
    st.title("Which parties talk about China?")

    by_party = an.china_by_party(df, top_n=50)

    tab1, tab2, tab3 = st.tabs(["Total speeches", "Normalised rate (%)", "Parliament seats"])
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

    with tab3:
        seat_chamber = st.radio(
            "Chamber",
            ["Tweede Kamer", "Eerste Kamer"],
            horizontal=True,
        )
        chamber_key = "tweedekamer" if seat_chamber == "Tweede Kamer" else "eerstekamer"

        seats = an.seat_chart_data(df, chamber=chamber_key)

        if seats.empty:
            st.warning("No seat data available for this filter.")
        else:
            # Election period label
            period = seats["period_label"].iloc[0]
            st.caption(
                f"**Composition**: {period} election period — seats coloured by political spectrum. "
                f"Click a party's dot to see its China sentiment profile."
            )

            # Party selector for click interaction
            all_parties = sorted(seats["party"].unique().tolist())
            selected_party = st.selectbox(
                "Select a party to inspect",
                options=["— select a party —"] + all_parties,
                index=0,
            )

            # Build color map: spectrum colour for all, highlighted for selected
            n_rows = 6 if chamber_key == "tweedekamer" else 4
            coords = horseshoe_coords(len(seats), n_rows=n_rows)
            seats = seats.assign(x=[c[0] for c in coords], y=[c[1] for c in coords])

            # Color: selected party gets its party colour; others use spectrum+dimmed
            colors = []
            for _, row in seats.iterrows():
                if selected_party != "— select a party —" and row["party"] == selected_party:
                    colors.append(SPECTRUM_PARTY_COLORS.get(row["party"], SPEC_DEFAULT))
                else:
                    colors.append(SPECTRUM_COLORS.get(row["spectrum"], SPEC_DEFAULT))

            seats["dot_color"] = colors
            seats["highlighted"] = (seats["party"] == selected_party) if selected_party != "— select a party —" else False

            fig = px.scatter(
                seats,
                x="x", y="y",
                color="spectrum",
                color_discrete_map=SPECTRUM_COLORS,
                hover_data={
                    "party": True,
                    "rate": ":.1f",
                    "china_speeches": True,
                    "total_speeches": True,
                    "mean_china_sentiment": ":.2f",
                    "sentiment_n": True,
                    "x": False, "y": False, "spectrum": False, "dot_color": False, "highlighted": False,
                },
                labels={
                    "rate": "China mention %",
                    "china_speeches": "China speeches",
                    "total_speeches": "Total speeches",
                    "mean_china_sentiment": "Avg sentiment",
                    "sentiment_n": "Sentiment speeches",
                },
            )

            # Apply per-point colours (spectrum colour, selected party overridden in hover)
            for trace in fig.data:
                trace.marker = dict(size=16, line=dict(width=0.8, color="white"))
            for i, color in enumerate(colors):
                fig.data[0].marker.color = None  # disable legend colour

            # Full per-point colour override via scatter with custom colors
            fig2 = go.Figure()
            # Draw by spectrum group so legend is clean
            for spectrum, group in seats.groupby("spectrum"):
                group_df = group
                highlight = group_df["highlighted"]
                # Non-highlighted dots first
                non_hl = group_df[~highlight]
                fig2.add_trace(go.Scatter(
                    x=non_hl["x"], y=non_hl["y"],
                    mode="markers",
                    name=spectrum,
                    marker=dict(
                        size=16,
                        color=SPECTRUM_COLORS.get(spectrum, SPEC_DEFAULT),
                        line=dict(width=0.8, color="white"),
                        opacity=0.85,
                    ),
                    customdata=non_hl[["party", "rate", "china_speeches", "total_speeches",
                                       "mean_china_sentiment", "sentiment_n"]].values,
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "China mention rate: %{customdata[1]:.1f}%<br>"
                        "China speeches: %{customdata[2]}<br>"
                        "Total speeches: %{customdata[3]}<br>"
                        "Avg China sentiment: %{customdata[4]:.2f}<br>"
                        "<extra></extra>"
                    ),
                    showlegend=True,
                ))
                # Highlighted dots on top
                hl = group_df[highlight]
                if len(hl) > 0:
                    fig2.add_trace(go.Scatter(
                        x=hl["x"], y=hl["y"],
                        mode="markers",
                        name=f"{spectrum} (selected)" if selected_party == "— select a party —" else hl["party"].iloc[0],
                        marker=dict(
                            size=20,
                            color=SPECTRUM_PARTY_COLORS.get(hl["party"].iloc[0], SPEC_DEFAULT),
                            line=dict(width=2, color="#222"),
                            symbol="diamond",
                        ),
                        customdata=hl[["party", "rate", "china_speeches", "total_speeches",
                                       "mean_china_sentiment", "sentiment_n"]].values,
                        hovertemplate=(
                            "<b>%{customdata[0]}</b><br>"
                            "China mention rate: %{customdata[1]:.1f}%<br>"
                            "China speeches: %{customdata[2]}<br>"
                            "Total speeches: %{customdata[3]}<br>"
                            "Avg China sentiment: %{customdata[4]:.2f}<br>"
                            "<extra></extra>"
                        ),
                        showlegend=selected_party != "— select a party —",
                        legendgroup=spectrum,
                    ))

            fig2.update_layout(
                xaxis=dict(visible=False, scaleanchor="y", scaleratio=1),
                yaxis=dict(visible=False),
                plot_bgcolor="white", height=500,
                legend=dict(
                    title="Political spectrum",
                    orientation="h", yanchor="bottom", y=1.02,
                    xanchor="center", x=0.5,
                ),
            )
            st.plotly_chart(fig2, use_container_width=True)

            # Spectrum colour legend
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("🔴 **Left** — SP, PvdA, GL, PvdD, DENK, BIJ1, Volt, OSF")
            with col2:
                st.markdown("🟠 **Center** — D66, ChristenUnie, NSC")
            with col3:
                st.markdown("🔵 **Right** — VVD, CDA, PVV, FvD, SGP, JA21, 50PLUS, BBB")

            # Sentiment panel for selected party
            if selected_party != "— select a party —":
                st.divider()
                summary = an.party_sentiment_summary(df, selected_party)
                party_seats = seats[seats["party"] == selected_party]
                n_seats_val = len(party_seats)

                col_party, col_stats = st.columns([1, 3])

                with col_party:
                    party_color = SPECTRUM_PARTY_COLORS.get(selected_party, SPEC_DEFAULT)
                    spectrum_val = party_seats["spectrum"].iloc[0] if len(party_seats) else "Unknown"
                    st.markdown(
                        f"### {selected_party}  "
                        f"<span style='background:{party_color};color:white;padding:2px 8px;border-radius:4px;font-size:0.8em'>{spectrum_val}</span>",
                        unsafe_allow_html=True,
                    )
                    rate_val = party_seats["rate"].iloc[0]
                    st.metric("Seats in parliament", n_seats_val)
                    st.metric("China mention rate", f"{rate_val:.1f}%")
                    if summary["mean_china_sentiment"] is not None:
                        sentiment_val = summary["mean_china_sentiment"]
                        # 0=very negative, 5=very positive
                        label = "🟢 Negative" if sentiment_val < 2.5 else "🟡 Neutral" if sentiment_val < 3.3 else "🟠 Positive"
                        st.metric("Avg China sentiment", f"{sentiment_val:.2f} ({label})")
                    else:
                        st.caption("No sentiment data for this party.")

                with col_stats:
                    st.markdown("**Year-by-year sentiment on China** (0=negative, 5=positive)")
                    if summary["yearly_sentiment"]:
                        yearly_df = pd.DataFrame([
                            {"Year": k, "China Sentiment": v}
                            for k, v in sorted(summary["yearly_sentiment"].items())
                        ])
                        # Sentiment bar: colour by value
                        yearly_df["bar_color"] = yearly_df["China Sentiment"].apply(
                            lambda x: "#D62828" if x < 2.3 else "#F77F00" if x < 3.0 else "#3A7D44"
                        )
                        yearly_fig = px.bar(
                            yearly_df, x="Year", y="China Sentiment",
                            color="China Sentiment",
                            color_continuous_scale=[[0, "#D62828"], [0.5, "#F77F00"], [1, "#3A7D44"]],
                            range_color=[1.5, 4.0],
                        )
                        yearly_fig.add_hline(y=2.5, line_dash="dash", annotation_text="Negative threshold", line_color="gray")
                        yearly_fig.update_layout(
                            height=220, margin=dict(t=10, b=10),
                            showlegend=False,
                            yaxis=dict(title=None),
                        )
                        st.plotly_chart(yearly_fig, use_container_width=True)
                    else:
                        st.caption("No yearly sentiment data.")

                    if summary["top_topics"]:
                        st.markdown("**Top topics when speaking about China**")
                        topics_df = pd.DataFrame([
                            {"Topic": t, "Speeches": c}
                            for t, c in summary["top_topics"].items()
                        ])
                        st.dataframe(topics_df, use_container_width=True, hide_index=True)


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

    tab1, tab_h, tab2, tab3 = st.tabs([
        "Animated by year", "Party heatmap", "Left vs Right over time", "Distribution"
    ])

    with tab1:
        st.subheader("Watch each party's stance on China shift over time")
        st.caption(
            "Press play (▶) or drag the slider. Bars below 2.5 = negative tone "
            "toward China that year, above = positive."
        )
        heatmap = an.sentiment_heatmap(df)
        if not heatmap.empty:
            anim_df = (
                heatmap.reset_index()
                .melt(id_vars="party", var_name="year", value_name="sentiment")
                .dropna()
            )
            anim_df["year"] = anim_df["year"].astype(int).astype(str)
            anim_df = anim_df.sort_values(["year", "party"])
            fig = px.bar(
                anim_df, x="party", y="sentiment",
                animation_frame="year",
                range_y=[0, 5],
                color="sentiment",
                color_continuous_scale="RdYlGn",
                range_color=[0, 5],
                labels={"sentiment": "China sentiment", "party": ""},
            )
            fig.add_hline(y=2.5, line_dash="dot", line_color="grey",
                          annotation_text="Neutral")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough data with current filters.")

    with tab_h:
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
        "When Dutch politicians mention China, **who else is in the room**? "
        "Is China discussed as an isolated actor, framed against the US, "
        "grouped with Russia, or set off against the EU? And does the company "
        "China keeps shape the *tone* of the conversation?"
    )

    tab_combo, tab_trend = st.tabs([
        "Combinations & sentiment", "Co-occurrence over time"
    ])

    with tab_combo:
        st.subheader("Which combinations dominate — and how negative are they?")
        st.caption(
            "Each bar = a unique combination of great powers mentioned together "
            "with China in the same speech. Colour = mean China-specific sentiment "
            "(red = negative, green = positive). Combinations with <10 speeches hidden."
        )
        combos = an.china_power_combinations(df)
        if combos.empty:
            st.warning("No combinations meet the threshold under current filters.")
        else:
            fig = px.bar(
                combos.sort_values("n_speeches"),
                x="n_speeches", y="combination", orientation="h",
                color="mean_china_sentiment",
                color_continuous_scale="RdYlGn",
                range_color=[0, 5],
                labels={"n_speeches": "Number of speeches",
                        "combination": "",
                        "mean_china_sentiment": "China sentiment"},
                hover_data={"mean_china_sentiment": ":.2f",
                            "mean_sentiment_avg": ":.2f"},
            )
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("Underlying numbers"):
                st.dataframe(combos, use_container_width=True)

    with tab_trend:
        st.caption(
            "How often is each great power mentioned alongside China, "
            "as a percentage of China-mentioning speeches that year?"
        )
        cooc = an.great_power_cooccurrence(df)
        if not cooc.empty:
            fig = px.line(
                cooc, x="year", y="cooccurrence_pct",
                color="power", markers=True,
                labels={"cooccurrence_pct": "% of China speeches also mentioning",
                        "year": "Year"},
                color_discrete_map={"US": HCSS_PRIMARY, "RUSSIA": "#C0392B",
                                     "EU": HCSS_ACCENT, "NATO": "#1A1A1A"},
            )
            st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Top speakers
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Top speakers":
    st.title("Most active speakers on China")

    speakers = an.top_china_speakers(df, top_n=25)

    tab_bubble, tab_drill = st.tabs(["Overview", "Speaker drill-down"])

    with tab_bubble:
        st.caption(
            "**X axis** = share of this person's *total* speeches that mention China. "
            "**Y axis** = average China-specific sentiment (below 2.5 = negative). "
            "**Bubble size** = total number of China mentions. "
            "**Colour** = party."
        )
        bubble_df = speakers.dropna(subset=["avg_china_sentiment", "china_pct"])
        if not bubble_df.empty:
            # Label format: name + context badge if not plain MP
            ctx_badge = {"minister": " [Min.]", "head": " [Chair]",
                         "secretaryOfState": " [Sec.]", "deputyHead": " [Dep.]"}
            bubble_df = bubble_df.copy()
            bubble_df["label"] = bubble_df.apply(
                lambda r: r["speaker_name"] + ctx_badge.get(r.get("speaker_context", ""), ""),
                axis=1,
            )
            fig = px.scatter(
                bubble_df,
                x="china_pct", y="avg_china_sentiment",
                size="total_china_mentions", color="party",
                text="label",
                hover_data={"china_speeches": True, "total_speeches": True,
                            "china_pct": ":.1f", "avg_china_sentiment": ":.2f",
                            "label": False},
                labels={"china_pct": "% of speeches mentioning China",
                        "avg_china_sentiment": "Avg. China sentiment",
                        "party": "Party"},
                size_max=45,
            )
            fig.add_hline(y=2.5, line_dash="dot", line_color="grey",
                          annotation_text="Neutral sentiment")
            fig.update_traces(textposition="top center",
                              textfont=dict(size=10))
            fig.update_layout(height=560)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No sentiment data available for current filters.")

        with st.expander("Full speaker table"):
            st.dataframe(speakers, use_container_width=True)

    with tab_drill:
        st.subheader("Explore one speaker's China speeches")
        all_china_speakers = sorted(
            df[df["china_mentions"] > 0]["speaker_name"].dropna().unique().tolist()
        )
        sel_speaker = st.selectbox("Select speaker", all_china_speakers)

        if sel_speaker:
            row = speakers[speakers["speaker_name"] == sel_speaker]
            if not row.empty:
                r = row.iloc[0]
                ctx_map = {"minister": "Minister", "head": "Chamber chair",
                           "secretaryOfState": "Secretary of State",
                           "deputyHead": "Deputy chair", "": "Party member"}
                ctx_str = ctx_map.get(r.get("speaker_context", ""), "")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Party", r["party"].replace("party.", ""))
                c2.metric("Role", ctx_str)
                c3.metric("China speeches", int(r["china_speeches"]))
                c4.metric("China % of total", f"{r['china_pct']:.1f}%")

                sent_val = r["avg_china_sentiment"]
                if pd.notna(sent_val):
                    sentiment_str = (
                        "Negative" if sent_val < 1.8 else
                        "Mildly negative" if sent_val < 2.5 else
                        "Neutral" if sent_val < 3.2 else "Positive"
                    )
                    st.info(
                        f"Average China sentiment: **{sent_val:.2f} / 5** — *{sentiment_str}*"
                    )

            st.divider()
            speeches = an.speaker_speeches(df, sel_speaker)
            st.caption(f"{len(speeches)} China-mentioning speeches — most recent first")
            for _, s in speeches.head(15).iterrows():
                with st.expander(
                    f"{str(s['date'])[:10]}  ·  {s.get('topic','') or '—'}  ·  "
                    f"sentiment: {s['sentiment_label'] or '—'}"
                ):
                    # Pull first sentence mentioning China as quote
                    text = s["text"] or ""
                    sentences = [t.strip() for t in text.split(".") if t.strip()]
                    china_terms = {"china", "chinese", "beijing", "taiwan", "uyghur",
                                   "uighur", "xi", "huawei", "hong kong", "xinjiang"}
                    quote = next(
                        (t for t in sentences
                         if any(term in t.lower() for term in china_terms)),
                        sentences[0] if sentences else text[:200],
                    )
                    st.markdown(f"> *{quote}*")
                    if pd.notna(s.get("china_sentiment_avg")):
                        st.caption(f"China sentiment score: {s['china_sentiment_avg']:.2f} / 5")

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
