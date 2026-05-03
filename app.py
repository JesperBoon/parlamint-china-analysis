"""
app.py — ParlaMint China Analysis Tool
Dutch Parliamentary Debates on China (2015–2022)

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


def horseshoe_coords_grouped(party_sizes: list, n_rows: int = 6):
    """
    Compute horseshoe (x, y) where each party occupies a contiguous arc segment.
    Seats fill column-by-column within each party's angular block, so the party
    appears as a solid cluster — like the real Dutch parliament.
    party_sizes: seat counts in left-to-right display order.
    Returns flat list of (x, y), grouped by party.
    """
    radii = [1.0 + i * 0.20 for i in range(n_rows)]
    total = sum(party_sizes)
    coords = []
    angle_cursor = 0.0
    for n_p in party_sizes:
        if n_p == 0:
            continue
        width = (n_p / total) * math.pi
        n_cols = max(1, math.ceil(n_p / n_rows))
        for i in range(n_p):
            col = i // n_rows
            row = i % n_rows
            angle = angle_cursor + (col + 0.5) / n_cols * width
            actual_angle = math.pi - angle   # 0=left arc, π=right arc
            r = radii[min(row, len(radii) - 1)]
            coords.append((r * math.cos(actual_angle), r * math.sin(actual_angle)))
        angle_cursor += width
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

PRIMARY_COLOR = "#003082"
ACCENT_COLOR = "#0066CC"
COLOR_PALETTE = ["#003082", "#0066CC", "#5A8FD6", "#1A1A1A", "#7F8C8D", "#A6BDDB"]

# ── Policy event timeline ──────────────────────────────────────────────────────
# 20 key Dutch–China policy moments, 2015–2022.
# year_frac: fractional year used as x-position on integer-year axes.
# period_Y / period_Q: string labels for yearly / quarterly period axes.
POLICY_EVENTS = [
    {"date": "2015-10-01", "year_frac": 2015.75, "period_Y": "2015", "period_Q": "2015Q4",
     "label": "COSCO acquires Rotterdam Euromax terminal stake",
     "category": "Trade & Economy", "short": "COSCO Rotterdam",
     "description": "Chinese state shipping company COSCO acquired a minority stake in the Rotterdam Euromax terminal, raising early concerns about Chinese investment in Dutch strategic infrastructure."},
    {"date": "2016-06-01", "year_frac": 2016.42, "period_Y": "2016", "period_Q": "2016Q2",
     "label": "First Dutch parliamentary motion on Uyghurs",
     "category": "Human Rights", "short": "1st Uyghur motion",
     "description": "The first Dutch parliamentary motion specifically addressing the treatment of Uyghurs in Xinjiang — a precursor to the more forceful resolutions that would follow from 2019 onward."},
    {"date": "2017-05-01", "year_frac": 2017.33, "period_Y": "2017", "period_Q": "2017Q2",
     "label": "Belt and Road Forum — NL participation debate",
     "category": "Diplomacy", "short": "BRI Forum",
     "description": "China's Belt and Road Initiative global summit in Beijing. Dutch participation triggered parliamentary debate on whether engagement with BRI constituted endorsement of Chinese geopolitical expansion."},
    {"date": "2018-07-01", "year_frac": 2018.50, "period_Y": "2018", "period_Q": "2018Q3",
     "label": "US–China trade war escalates (tariff round 2)",
     "category": "Trade & Economy", "short": "Trade war",
     "description": "The US escalated its trade conflict with China with a second round of tariffs. Dutch parliament debated the implications for Netherlands–China trade and the risk of being caught between the two economic superpowers."},
    {"date": "2018-11-01", "year_frac": 2018.83, "period_Y": "2018", "period_Q": "2018Q4",
     "label": "Dutch government starts Huawei 5G security review",
     "category": "Technology", "short": "Huawei review",
     "description": "Following US warnings, the Dutch government opened a formal security review into whether Huawei equipment posed an espionage risk if deployed in the Netherlands' 5G network infrastructure."},
    {"date": "2019-05-01", "year_frac": 2019.33, "period_Y": "2019", "period_Q": "2019Q2",
     "label": "US blacklists Huawei — NL suppliers affected",
     "category": "Technology", "short": "Huawei blacklist",
     "description": "The US placed Huawei on its entity list, barring American firms from supplying components. Dutch chip equipment firm ASML and other suppliers were immediately affected, forcing a Dutch policy response."},
    {"date": "2019-10-01", "year_frac": 2019.75, "period_Y": "2019", "period_Q": "2019Q4",
     "label": "Dutch parliament passes Xinjiang motion",
     "category": "Human Rights", "short": "Xinjiang motion",
     "description": "Dutch parliament passed a formal motion condemning the mass detention of Uyghurs in Xinjiang — one of the strongest official statements by any EU member state at that point."},
    {"date": "2020-01-23", "year_frac": 2020.06, "period_Y": "2020", "period_Q": "2020Q1",
     "label": "Wuhan lockdown — COVID-19 enters Dutch debate",
     "category": "Security", "short": "COVID / Wuhan",
     "description": "China's lockdown of Wuhan brought COVID-19 into Dutch parliamentary debate, initially framing China as the origin of the outbreak and raising questions about Chinese transparency and WHO cooperation."},
    {"date": "2020-03-28", "year_frac": 2020.24, "period_Y": "2020", "period_Q": "2020Q1",
     "label": "NL recalls defective Chinese face masks",
     "category": "Trade & Economy", "short": "Mask recall",
     "description": "The Netherlands recalled a shipment of 600,000 Chinese-supplied face masks after they failed safety standards — a flashpoint in the broader debate about Dutch dependency on Chinese supply chains."},
    {"date": "2020-07-09", "year_frac": 2020.52, "period_Y": "2020", "period_Q": "2020Q3",
     "label": "NL ends extradition treaty with Hong Kong",
     "category": "Diplomacy", "short": "HK extradition ends",
     "description": "Following China's imposition of the National Security Law on Hong Kong, the Netherlands suspended its extradition treaty with Hong Kong — one of the most concrete Dutch diplomatic responses to Chinese policy."},
    {"date": "2020-10-15", "year_frac": 2020.79, "period_Y": "2020", "period_Q": "2020Q4",
     "label": "ASML export licence to China restricted (US pressure)",
     "category": "Technology", "short": "ASML export limit",
     "description": "Under US pressure, the Dutch government declined to renew ASML's export licence for its most advanced EUV lithography machines to Chinese customers — restricting Chinese access to the world's most critical chip-making equipment."},
    {"date": "2021-02-25", "year_frac": 2021.15, "period_Y": "2021", "period_Q": "2021Q1",
     "label": "Dutch parliament adopts Uyghur 'genocide' motion",
     "category": "Human Rights", "short": "Genocide motion",
     "description": "Dutch parliament passed a motion labelling Chinese policies in Xinjiang as genocide — the first EU parliament to do so, and a direct contradiction of the government's own initial position."},
    {"date": "2021-03-22", "year_frac": 2021.22, "period_Y": "2021", "period_Q": "2021Q1",
     "label": "EU–China mutual sanctions (MEPs, scholars)",
     "category": "Diplomacy", "short": "EU–China sanctions",
     "description": "The EU imposed targeted sanctions on Chinese officials over Xinjiang abuses. China retaliated with counter-sanctions on European MEPs and scholars, effectively blocking ratification of the EU–China investment deal."},
    {"date": "2021-05-01", "year_frac": 2021.33, "period_Y": "2021", "period_Q": "2021Q2",
     "label": "KPN chooses Ericsson over Huawei for 5G core",
     "category": "Technology", "short": "KPN Huawei out",
     "description": "Dutch telecom giant KPN selected Ericsson over Huawei for its 5G core network — a decision made under government pressure, seen as the definitive Dutch exit from Huawei's 5G ecosystem."},
    {"date": "2021-09-15", "year_frac": 2021.70, "period_Y": "2021", "period_Q": "2021Q3",
     "label": "Dutch parliament Taiwan solidarity motion",
     "category": "Security", "short": "Taiwan motion",
     "description": "Dutch parliament passed a motion expressing solidarity with Taiwan and calling for EU support for Taiwanese participation in international organisations, drawing a sharp diplomatic protest from Beijing."},
    {"date": "2021-11-01", "year_frac": 2021.83, "period_Y": "2021", "period_Q": "2021Q4",
     "label": "ASML banned from shipping EUV machines to China",
     "category": "Technology", "short": "ASML EUV ban",
     "description": "The Dutch government permanently refused to renew ASML's EUV export licence to China following sustained US pressure, placing the world's most advanced semiconductor equipment effectively beyond Chinese reach."},
    {"date": "2022-01-01", "year_frac": 2022.00, "period_Y": "2022", "period_Q": "2022Q1",
     "label": "NL tightens FDI screening — China acquisitions scrutinised",
     "category": "Trade & Economy", "short": "FDI screening",
     "description": "The Netherlands introduced tightened foreign direct investment screening rules, allowing the government to block or review Chinese acquisitions in sensitive sectors including semiconductors, ports, and telecoms."},
    {"date": "2022-02-24", "year_frac": 2022.15, "period_Y": "2022", "period_Q": "2022Q1",
     "label": "Russia invades Ukraine — China neutrality debated",
     "category": "Security", "short": "Russia–Ukraine / China",
     "description": "Russia's invasion of Ukraine brought China's position into Dutch parliamentary debate. China's refusal to condemn the invasion and its 'no-limits partnership' with Russia raised new concerns about Chinese complicity in European instability."},
    {"date": "2022-06-01", "year_frac": 2022.42, "period_Y": "2022", "period_Q": "2022Q2",
     "label": "Dutch parliament debates semiconductor export controls",
     "category": "Technology", "short": "Chip export debate",
     "description": "Dutch parliament debated further restrictions on semiconductor equipment exports to China, reflecting growing awareness that ASML's deep-UV machines — still exportable — remained a critical chokepoint in China's chip ambitions."},
    {"date": "2022-09-01", "year_frac": 2022.67, "period_Y": "2022", "period_Q": "2022Q3",
     "label": "Cabinet announces further ASML export restrictions",
     "category": "Technology", "short": "ASML further limits",
     "description": "The Dutch cabinet announced additional export restrictions on ASML deep-UV lithography machines, extending the EUV precedent to a broader range of chip equipment — a major escalation in technology decoupling from China."},
]

POLICY_EVENT_COLORS = {
    "Human Rights":  "#C0392B",
    "Trade & Economy": "#E67E22",
    "Technology":    "#2980B9",
    "Security":      "#8E44AD",
    "Diplomacy":     "#27AE60",
}


def add_policy_lines(fig, x_type: str = "year", events=None, selected=None):
    """
    Inject vertical marker lines for policy events onto a Plotly figure.
    x_type: 'year' (int/float axis), 'period_Y' (string yearly), 'period_Q' (string quarterly).
    selected: list of event short labels to show; None = show all.

    Note: add_vline() crashes on categorical/string x-axes (Plotly tries to do
    arithmetic on string x-values internally). For string x-types we use
    add_shape() + add_annotation() instead, which handles category labels correctly.
    """
    if events is None:
        events = POLICY_EVENTS
    for ev in events:
        if selected and ev["short"] not in selected:
            continue
        x_val = ev.get(x_type, ev["year_frac"])
        color = POLICY_EVENT_COLORS.get(ev["category"], "#888888")

        if isinstance(x_val, str):
            # Categorical (string period) axis — add_vline fails here.
            # Use add_shape + add_annotation which handle category labels correctly.
            fig.add_shape(
                type="line",
                x0=x_val, x1=x_val,
                y0=0, y1=1,
                xref="x", yref="paper",
                line=dict(dash="dot", width=1.2, color=color),
            )
            fig.add_annotation(
                x=x_val,
                y=1,
                xref="x", yref="paper",
                text=ev["short"],
                showarrow=False,
                textangle=-90,
                font=dict(size=8, color=color),
                yanchor="top",
                xanchor="right",
            )
        else:
            # Numeric axis — add_vline works fine.
            fig.add_vline(
                x=x_val,
                line_dash="dot", line_width=1.2, line_color=color,
                annotation_text=ev["short"],
                annotation_position="top left",
                annotation_font=dict(size=8, color=color),
                annotation_textangle=-90,
            )
    return fig

# ── Sentiment label mapping (shorthand → plain English) ───────────────────────
SENTIMENT_LABELS = {
    "negneg": "Very Negative",
    "mixneg": "Mixed Negative",
    "neuneg": "Neutral–Negative",
    "neupos": "Neutral–Positive",
    "mixpos": "Mixed Positive",
    "pospos": "Very Positive",
}
SENTIMENT_ORDER = ["Very Negative", "Mixed Negative", "Neutral–Negative",
                   "Neutral–Positive", "Mixed Positive", "Very Positive"]

def score_to_label(score) -> str:
    """Convert numeric sentiment score (0–5) to plain English label."""
    if pd.isna(score):
        return "No data"
    if score < 1.0:   return "Very Negative"
    if score < 1.8:   return "Negative"
    if score < 2.5:   return "Neutral–Negative"
    if score < 3.2:   return "Neutral–Positive"
    if score < 4.2:   return "Positive"
    return "Very Positive"

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="China in Dutch Parliament",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "processed", "speeches.parquet")

# ── Data loading (cached) ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading dataset...")
def load_data():
    if not os.path.exists(DATA_PATH):
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id="ELFrijol/parlamint-china-nl",
            filename="speeches.parquet",
            repo_type="dataset",
            local_dir=os.path.dirname(DATA_PATH),
        )
    return an.load(DATA_PATH)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("China in Dutch Parliament")
    st.caption("How China is discussed in the Dutch Parliament")
    st.caption("ParlaMint-NL | 2015–2022")
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Party & Sentiment Trends",
            "Sentiment analysis",
            "Great power context",
            "Top speakers",
            "Policy & Geopolitics",
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
    st.title("How China is discussed in the Dutch Parliament")

    st.markdown(
        "Between 2014 and 2022, China went from a barely-mentioned trading partner "
        "to a contentious topic in Dutch politics. This dashboard maps the dynamics "
        "within the Eerste Kamer and Tweede Kamer that drive decisions and stances toward China."
    )
    st.caption("Based on 593,961 speeches from ParlaMint-NL · Use the sidebar to explore each dimension in depth.")

    # ── KPIs ──────────────────────────────────────────────────────────────────
    china_df = df[df["china_mentions"] > 0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total speeches analysed", f"{len(df):,}")
    c2.metric("Mention China", f"{len(china_df):,}", f"{len(china_df)/len(df)*100:.1f}% of all")
    c3.metric("Unique speakers", china_df["speaker_name"].nunique())
    c4.metric("Peak year", "2020", f"{china_df[china_df['year']==2020].shape[0]} speeches")

    st.divider()

    # ── Chapter 1: The rise ────────────────────────────────────────────────────
    st.subheader("1 · A topic that wouldn't stop growing")
    st.markdown("China becomes more and more of a topic in Dutch parliament — with an obvious peak during COVID.")
    st.info(
        "In 2014, China appeared in just **13 speeches** across the entire parliament. "
        "By 2020, that number had risen to **569** — a 44× increase in six years. "
        "The jump wasn't gradual: it accelerated sharply after 2018, driven by Huawei's "
        "5G expansion, the Uyghur detention camps, and growing concern about economic dependency. "
        "Its peak was around the COVID crisis."
    )

    trend = an.china_trend(df, freq="Y")
    fig = px.bar(
        trend, x="period", y="china_speeches",
        labels={"period": "", "china_speeches": "Speeches mentioning China"},
        color_discrete_sequence=[PRIMARY_COLOR],
    )
    fig.update_layout(showlegend=False, margin=dict(t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Chapter 2: The tone ────────────────────────────────────────────────────
    st.subheader("2 · The tone is overwhelmingly negative — and getting more so")
    st.markdown(
        "Sentiment scores are computed only from sentences that explicitly mention China "
        "(scale 0–5, where 2.5 is neutral). Across all parties, the mean sits well below "
        "neutral — and drops further after 2019. The quotes below are representative."
    )

    col_q1, col_q2 = st.columns(2)
    with col_q1:
        st.markdown(
            "> *\"There are even pictures of the abuses in China taken from satellites. "
            "There's no lack of evidence.\"*"
        )
        st.caption("Sjoerd Sjoerdsma (D66) · February 2021 · on Xinjiang")
    with col_q2:
        st.markdown(
            "> *\"The entire population of the Netherlands has to do with arbitrary "
            "detention, abuse and indoctrination in political re-education camps.\"*"
        )
        st.caption("Tunahan Kuzu (DENK) · September 2018 · on Uyghurs")

    bloc_df = an.sentiment_by_bloc(df)
    if not bloc_df.empty:
        fig = px.line(
            bloc_df, x="year", y="mean_china_sentiment",
            color="bloc", markers=True,
            color_discrete_map={"Left": "#C0392B", "Center": "#7F8C8D", "Right": PRIMARY_COLOR},
            labels={"mean_china_sentiment": "Mean China sentiment (0–5)", "year": ""},
        )
        fig.add_hline(y=2.5, line_dash="dot", line_color="grey",
                      annotation_text="Neutral")
        fig.update_layout(margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Chapter 3: Is China uniquely controversial? ───────────────────────────
    st.subheader("3 · Is China uniquely controversial — or is this just geopolitics?")
    st.markdown(
        "Dutch parliament is negative about China. But are they equally negative about "
        "Russia and the US? The bars below show average China-specific sentiment depending "
        "on which other power is mentioned in the same speech. "
        "**Speeches combining China with Russia are the most negative of all.**"
    )

    proxy_df = an.power_sentiment_proxy(df)
    if not proxy_df.empty:
        POWER_COLORS = {
            "China (overall)": PRIMARY_COLOR,
            "China + US":      "#5A8FD6",
            "China + Russia":  "#C0392B",
            "China + EU":      "#F39C12",
            "China + NATO":    "#1A1A1A",
        }
        proxy_df = proxy_df.sort_values("avg_sentiment")
        proxy_df["color"] = proxy_df["power"].map(POWER_COLORS)
        proxy_df["label"] = proxy_df["avg_sentiment"].apply(score_to_label)

        col_bar, col_numbers = st.columns([3, 1])
        with col_bar:
            fig = px.bar(
                proxy_df,
                x="avg_sentiment", y="power", orientation="h",
                color="power",
                color_discrete_map=POWER_COLORS,
                labels={"avg_sentiment": "Avg. China sentiment (0–5)", "power": ""},
                hover_data={"n_speeches": True, "label": True, "avg_sentiment": ":.2f"},
                text="label",
            )
            fig.add_vline(x=2.5, line_dash="dot", line_color="grey",
                          annotation_text="Neutral", annotation_position="top")
            fig.update_traces(textposition="outside", textfont=dict(size=10))
            fig.update_layout(
                showlegend=False,
                xaxis=dict(range=[0, 4]),
                margin=dict(t=10, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)
        with col_numbers:
            st.markdown("**Avg. sentiment**")
            for _, row in proxy_df.sort_values("avg_sentiment").iterrows():
                st.markdown(
                    f"**{row['power']}**  \n"
                    f"`{row['avg_sentiment']:.2f}` · {row['n_speeches']:,} speeches"
                )

    st.divider()

    # ── Chapter 4: Geopolitical framing ───────────────────────────────────────
    st.subheader("4 · In what geopolitical contexts is China mostly discussed?")
    st.markdown(
        "When Dutch politicians mention China, they most often also mention the **EU** — "
        "reflecting debates about European trade policy and strategic autonomy. "
        "Speeches that combine China with **Russia** are the most negative of all, "
        "suggesting a distinct geopolitical threat frame."
    )

    combos = an.china_power_combinations(df)
    if not combos.empty:
        fig = px.bar(
            combos.sort_values("n_speeches").tail(8),
            x="n_speeches", y="combination", orientation="h",
            color="mean_china_sentiment",
            color_continuous_scale="RdYlGn",
            range_color=[0, 5],
            labels={"n_speeches": "Number of speeches",
                    "combination": "",
                    "mean_china_sentiment": "Avg China sentiment"},
        )
        fig.update_layout(margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Parliament seat chart: hawkishness vs political order ─────────────────
    st.subheader("What if we ordered Dutch parties from most negative to most positive attitude toward China?")
    st.markdown(
        "Each dot = one parliamentary seat, coloured by party — parties clustered as in the real Tweede Kamer. "
        "Press **Re-seat by China stance** to reorder parties by their average sentiment toward China."
    )

    hook_year = st.select_slider(
        "Year",
        options=sorted(df["year"].dropna().unique().tolist()),
        value=int(df["year"].max()),
        key="hook_year",
    )
    df_hook = df[df["year"] == hook_year]
    seats_hook = an.seat_chart_data(df_hook, chamber="tweedekamer")

    if not seats_hook.empty:
        SPEC_ORDER_MAP = {
            "SP": 0, "PvdA": 1, "GroenLinks": 2, "PvdD": 3, "DENK": 4,
            "BIJ1": 5, "Volt": 6, "OSF": 7,
            "D66": 8, "ChristenUnie": 9, "NSC": 10,
            "CDA": 11, "VVD": 12, "PVV": 13, "FvD": 14,
            "SGP": 15, "JA21": 16, "50PLUS": 17, "BBB": 18,
        }

        party_info = (
            seats_hook.drop_duplicates("party")
            .set_index("party")[["rate", "mean_china_sentiment"]]
        )

        def _build_frame(party_order):
            """Given an ordered list of party names, return (x, y, colors, hovertexts)."""
            sizes = [(seats_hook["party"] == p).sum() for p in party_order]
            coords = horseshoe_coords_grouped(sizes, n_rows=6)
            xs, ys, cs, hts = [], [], [], []
            i = 0
            for p, n in zip(party_order, sizes):
                color = SPECTRUM_PARTY_COLORS.get(p, SPEC_DEFAULT)
                rate = party_info["rate"].get(p, 0.0)
                snt = party_info["mean_china_sentiment"].get(p, float("nan"))
                snt_str = f"{snt:.2f}" if not pd.isna(snt) else "no data"
                for _ in range(n):
                    xs.append(coords[i][0])
                    ys.append(coords[i][1])
                    cs.append(color)
                    hts.append(
                        f"<b>{p}</b><br>"
                        f"Mention rate: {rate:.1f}%<br>"
                        f"Avg China sentiment: {snt_str}"
                    )
                    i += 1
            return xs, ys, cs, hts

        parties_pol = sorted(
            seats_hook["party"].unique(),
            key=lambda p: SPEC_ORDER_MAP.get(p, 50),
        )
        x1h, y1h, c1h, ht1h = _build_frame(parties_pol)

        parties_snt = sorted(
            seats_hook["party"].unique(),
            key=lambda p: (
                party_info["mean_china_sentiment"].get(p, 2.5)
                if not pd.isna(party_info["mean_china_sentiment"].get(p, float("nan")))
                else 2.5
            ),
        )
        x2h, y2h, c2h, ht2h = _build_frame(parties_snt)

        ann_pol = [dict(
            text="← Left wing  ·  Political spectrum  ·  Right wing →",
            x=0.5, y=0.01, xref="paper", yref="paper",
            showarrow=False, font=dict(size=11, color="#888"), xanchor="center",
        )]
        ann_snt = [dict(
            text="← Most negative on China  ·  Stance toward China  ·  Most positive →",
            x=0.5, y=0.01, xref="paper", yref="paper",
            showarrow=False, font=dict(size=11, color="#888"), xanchor="center",
        )]

        fig_hook = go.Figure(
            data=[go.Scatter(
                x=x1h, y=y1h, mode="markers",
                marker=dict(size=12, color=c1h, line=dict(width=0.8, color="white")),
                hovertext=ht1h,
                hovertemplate="%{hovertext}<extra></extra>",
                showlegend=False,
            )],
            frames=[
                go.Frame(
                    name="sentiment",
                    data=[go.Scatter(
                        x=x2h, y=y2h, mode="markers",
                        marker=dict(size=12, color=c2h, line=dict(width=0.8, color="white")),
                        hovertext=ht2h,
                        hovertemplate="%{hovertext}<extra></extra>",
                        showlegend=False,
                    )],
                    layout=go.Layout(annotations=ann_snt),
                ),
                go.Frame(
                    name="political",
                    data=[go.Scatter(
                        x=x1h, y=y1h, mode="markers",
                        marker=dict(size=12, color=c1h, line=dict(width=0.8, color="white")),
                        hovertext=ht1h,
                        hovertemplate="%{hovertext}<extra></extra>",
                        showlegend=False,
                    )],
                    layout=go.Layout(annotations=ann_pol),
                ),
            ],
            layout=go.Layout(
                xaxis=dict(visible=False, range=[-1.75, 1.75]),
                yaxis=dict(visible=False, scaleanchor="x", scaleratio=1,
                           range=[-0.12, 1.72]),
                plot_bgcolor="white",
                height=480,
                margin=dict(t=10, b=90, l=10, r=10),
                annotations=ann_pol,
                updatemenus=[dict(
                    type="buttons",
                    showactive=True,
                    y=-0.02, x=0.5, xanchor="center", yanchor="top",
                    direction="left",
                    pad=dict(t=8),
                    buttons=[
                        dict(
                            label="Re-seat by China stance →",
                            method="animate",
                            args=[["sentiment"], dict(
                                frame=dict(duration=900, redraw=True),
                                transition=dict(duration=700, easing="cubic-in-out"),
                            )],
                        ),
                        dict(
                            label="← Political order",
                            method="animate",
                            args=[["political"], dict(
                                frame=dict(duration=900, redraw=True),
                                transition=dict(duration=700, easing="cubic-in-out"),
                            )],
                        ),
                    ],
                )],
            ),
        )
        st.caption(f"Tweede Kamer composition · {hook_year} · parties in political order left→right")
        st.plotly_chart(fig_hook, use_container_width=True)
        st.caption(
            "Hawkishness on China cuts across the left/right divide — "
            "PVV (far right) and SP/GroenLinks (far left) cluster together on the critical end."
        )

    st.divider()
    st.markdown(
        "**Use the sidebar** to explore any of these dimensions in depth — "
        "filter by year, chamber, or speaker role, and drill down into individual speeches."
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Party & Sentiment Trends (replaces Trend over time + Party comparison)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Party & Sentiment Trends":
    st.title("Party & Sentiment Trends")
    st.markdown(
        "How has each party's *tone* toward China shifted over time — and who drives the debate? "
        "Select parties to compare across all views."
    )

    # ── Party selector (shared across all tabs) ────────────────────────────────
    all_party_opts = sorted(an.parties_only(df)["party"].dropna().unique().tolist())
    top8 = an.china_by_party(df, top_n=8)["party"].tolist()
    default_sel = [p for p in top8 if p in all_party_opts]

    selected_parties = st.multiselect(
        "Select parties",
        all_party_opts,
        default=default_sel,
        help="Applies to all tabs. Select 2–10 parties for best legibility.",
    )
    if not selected_parties:
        st.info("Select at least one party above.")
        st.stop()

    color_map = {p: SPECTRUM_PARTY_COLORS.get(p, SPEC_DEFAULT) for p in selected_parties}

    tab_sent, tab_vol, tab_rates, tab_seats = st.tabs([
        "Sentiment over time", "Mention volume", "Party rates", "Parliament seats",
    ])

    # ── Tab 1: Sentiment over time ─────────────────────────────────────────────
    with tab_sent:
        st.subheader("Average China sentiment per party per year")
        st.caption(
            "Computed only from sentences that explicitly mention China (scale 0–5, below 2.5 = negative). "
            "Only party-years with at least 1 scored speech are shown."
        )
        show_events_sent = st.checkbox("Show policy milestones", value=True, key="ev_sent")
        sent_trend = an.party_sentiment_trend(df, parties=selected_parties)
        if sent_trend.empty:
            st.warning("No sentiment data for the selected parties under current filters.")
        else:
            fig = px.line(
                sent_trend, x="year", y="avg_china_sentiment",
                color="party", markers=True,
                color_discrete_map=color_map,
                labels={
                    "avg_china_sentiment": "Avg. China sentiment (0–5)",
                    "year": "", "party": "Party", "n_speeches": "Scored speeches",
                },
                hover_data={"n_speeches": True, "avg_china_sentiment": ":.2f"},
            )
            fig.add_hline(y=2.5, line_dash="dot", line_color="lightgrey",
                          annotation_text="Neutral")
            if show_events_sent:
                add_policy_lines(fig, x_type="year_frac")
            fig.update_layout(
                yaxis=dict(range=[0, 5]),
                height=500,
                margin=dict(t=50),
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="center", x=0.5),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "A downward trend = increasingly negative framing of China. "
                "Dotted vertical lines = key Dutch–China policy moments (toggle above)."
            )

    # ── Tab 2: Mention volume ──────────────────────────────────────────────────
    with tab_vol:
        st.subheader("How often does each party mention China?")
        col_mode, col_gran = st.columns(2)
        with col_mode:
            y_mode = st.radio(
                "Y-axis",
                ["Mention count", "Share of party speeches (%)"],
                horizontal=True, key="vol_ymode",
            )
        with col_gran:
            freq_label = st.radio("Granularity", ["Year", "Quarter"], horizontal=True, key="vol_gran")
        freq_map_vol = {"Year": "Y", "Quarter": "Q"}
        vol_trend = an.china_trend_by_party(
            df, freq=freq_map_vol[freq_label], parties=selected_parties
        )
        show_events_vol = st.checkbox("Show policy milestones", value=True, key="ev_vol")
        if vol_trend.empty:
            st.warning("No data for selected parties.")
        else:
            y_col = "china_speeches" if y_mode == "Mention count" else "pct"
            y_label = (
                "Speeches mentioning China"
                if y_mode == "Mention count"
                else "% of party's speeches mentioning China"
            )
            fig = px.line(
                vol_trend, x="period", y=y_col,
                color="party", markers=True,
                color_discrete_map=color_map,
                labels={"period": "", "party": "Party", y_col: y_label},
            )
            if show_events_vol:
                x_type_vol = "period_Y" if freq_label == "Year" else "period_Q"
                add_policy_lines(fig, x_type=x_type_vol)
            fig.update_layout(
                height=480,
                margin=dict(t=50),
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="center", x=0.5),
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Tab 3: Party rates ─────────────────────────────────────────────────────
    with tab_rates:
        st.subheader("Normalised China mention rates — all parties")
        st.caption(
            "% of each party's own speeches that mention China — corrects for parties that simply speak more."
        )
        st.markdown(
            "Normalisation means we divide each party's China mentions by their total number of speeches. "
            "This way, a small party with 10 China mentions out of 50 speeches (20%) ranks higher than a "
            "large party with 50 China mentions out of 1,000 speeches (5%) — even though the large party "
            "mentioned China more in absolute terms."
        )
        by_party = an.china_by_party(df, top_n=50)
        sent_per_party = (
            df[df["china_sentiment_avg"].notna() & (df["china_mentions"] > 0)]
            .groupby("party")["china_sentiment_avg"].mean().round(3)
            .rename("avg_china_sentiment")
        )
        by_party = by_party.merge(sent_per_party, on="party", how="left")
        by_party["sentiment_label"] = by_party["avg_china_sentiment"].apply(score_to_label)

        sent_view = st.radio(
            "Sort by",
            ["China mention rate", "Sentiment (most negative first)"],
            horizontal=True, key="party_sort_unified",
        )
        if sent_view == "Sentiment (most negative first)":
            plot_df = by_party.dropna(subset=["avg_china_sentiment"]).sort_values("avg_china_sentiment")
            st.caption(
                "⚠️ **Left = most negative sentiment toward China**, right = most positive. "
                "This is *not* a political left/right ordering."
            )
            fig = px.bar(
                plot_df, x="party", y="avg_china_sentiment",
                color="avg_china_sentiment",
                color_continuous_scale="RdYlGn", range_color=[0, 5],
                labels={"avg_china_sentiment": "Avg. China sentiment (0–5)", "party": ""},
                hover_data={"rate": ":.1f", "china_speeches": True, "sentiment_label": True},
                text="sentiment_label",
            )
            fig.add_hline(y=2.5, line_dash="dot", line_color="grey", annotation_text="Neutral")
            fig.update_traces(textposition="outside", textfont=dict(size=9))
            fig.update_layout(showlegend=False, coloraxis_showscale=False)
        else:
            plot_df = by_party.sort_values("rate")
            fig = px.bar(
                plot_df, x="rate", y="party", orientation="h",
                color="avg_china_sentiment",
                color_continuous_scale="RdYlGn", range_color=[0, 5],
                labels={"rate": "% of party's speeches mentioning China",
                        "party": "", "avg_china_sentiment": "Avg. sentiment"},
                hover_data={"china_speeches": True, "total_speeches": True,
                            "sentiment_label": True, "avg_china_sentiment": ":.2f"},
            )
            fig.update_layout(coloraxis_colorbar=dict(title="Sentiment"))
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 4: Parliament seats ────────────────────────────────────────────────
    with tab_seats:
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
            period = seats["period_label"].iloc[0]
            st.caption(
                f"**Composition**: {period} election period — each dot is one seat, "
                f"coloured by party. Select a party below to highlight it."
            )

            all_parties = sorted(seats["party"].unique().tolist())
            selected_party = st.selectbox(
                "Select a party to inspect",
                options=["— select a party —"] + all_parties,
                index=0,
                key="seat_party_sel",
            )

            n_rows = 6 if chamber_key == "tweedekamer" else 4
            coords = horseshoe_coords(len(seats), n_rows=n_rows)
            seats = seats.assign(x=[c[0] for c in coords], y=[c[1] for c in coords])
            seats["highlighted"] = (
                (seats["party"] == selected_party)
                if selected_party != "— select a party —" else False
            )

            fig2 = go.Figure()
            for party, group in seats.groupby("party"):
                highlight = group["highlighted"]
                non_hl = group[~highlight]
                hl = group[highlight]
                party_color = SPECTRUM_PARTY_COLORS.get(party, SPEC_DEFAULT)

                if len(non_hl):
                    opacity = 0.3 if selected_party != "— select a party —" else 0.85
                    fig2.add_trace(go.Scatter(
                        x=non_hl["x"], y=non_hl["y"],
                        mode="markers", name=party,
                        marker=dict(size=14, color=party_color,
                                    line=dict(width=0.5, color="white"),
                                    opacity=opacity),
                        customdata=non_hl[["party", "rate", "china_speeches",
                                           "total_speeches", "mean_china_sentiment"]].values,
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
                if len(hl):
                    fig2.add_trace(go.Scatter(
                        x=hl["x"], y=hl["y"],
                        mode="markers", name=f"{party} ★",
                        marker=dict(size=18, color=party_color,
                                    line=dict(width=2, color="#222"),
                                    symbol="diamond", opacity=1.0),
                        customdata=hl[["party", "rate", "china_speeches",
                                       "total_speeches", "mean_china_sentiment"]].values,
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

            fig2.update_layout(
                xaxis=dict(visible=False, scaleanchor="y", scaleratio=1),
                yaxis=dict(visible=False),
                plot_bgcolor="white", height=500,
                legend=dict(
                    title="Party",
                    orientation="h", yanchor="bottom", y=1.02,
                    xanchor="center", x=0.5,
                    font=dict(size=10),
                ),
            )
            st.plotly_chart(fig2, use_container_width=True)

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
                        label = "Negative" if sentiment_val < 2.5 else "Neutral" if sentiment_val < 3.3 else "Positive"
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

    tab1, tab_h, tab2, tab3, tab_ref = st.tabs([
        "Animated by year", "Sorted by sentiment", "Single party over time",
        "Distribution", "Reference comparison",
    ])

    heatmap = an.sentiment_heatmap(df)

    with tab1:
        st.subheader("Watch each party's stance on China shift over time")
        st.caption(
            "Press play (▶) or drag the year slider. "
            "Grey bars = no data for that party that year. "
            "Below 2.5 = negative tone toward China."
        )
        if not heatmap.empty:
            all_parties_h = sorted(heatmap.index.tolist())
            all_years_h = sorted(heatmap.columns.tolist())
            # Build complete grid including NaN → grey bar at 0
            rows_anim = []
            for yr in all_years_h:
                for p in all_parties_h:
                    val = heatmap.loc[p, yr] if p in heatmap.index and yr in heatmap.columns else float("nan")
                    rows_anim.append({"year": str(int(yr)), "party": p,
                                      "sentiment": val,
                                      "has_data": not pd.isna(val)})
            anim_df = pd.DataFrame(rows_anim)
            anim_df["bar_val"] = anim_df["sentiment"].fillna(0)
            anim_df["color_val"] = anim_df["sentiment"].fillna(-1)
            anim_df = anim_df.sort_values(["year", "party"])

            fig = px.bar(
                anim_df, x="party", y="bar_val",
                animation_frame="year",
                range_y=[0, 5],
                color="color_val",
                color_continuous_scale=[[0, "#CCCCCC"], [0.001, "#CC0000"],
                                         [0.5, "#FFFF00"], [1.0, "#00AA00"]],
                range_color=[-1, 5],
                labels={"bar_val": "China sentiment (0–5)", "party": ""},
                hover_data={"has_data": True, "sentiment": ":.2f",
                            "bar_val": False, "color_val": False},
            )
            fig.add_hline(y=2.5, line_dash="dot", line_color="grey",
                          annotation_text="Neutral")
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough data with current filters.")

    with tab_h:
        st.subheader("Parties ranked by average China sentiment")
        st.caption("Most negative on the left. Only parties with sufficient data shown.")
        if not heatmap.empty:
            party_avg = heatmap.mean(axis=1).sort_values()
            sorted_df = pd.DataFrame({
                "party": party_avg.index,
                "avg_sentiment": party_avg.values,
            })
            sorted_df["label"] = sorted_df["avg_sentiment"].apply(score_to_label)
            fig = px.bar(
                sorted_df,
                x="party", y="avg_sentiment",
                color="avg_sentiment",
                color_continuous_scale="RdYlGn",
                range_color=[0, 5],
                labels={"avg_sentiment": "Avg. China sentiment (0–5)", "party": ""},
                hover_data={"label": True},
                text="label",
            )
            fig.add_hline(y=2.5, line_dash="dot", line_color="grey",
                          annotation_text="Neutral")
            fig.update_traces(textposition="outside", textfont=dict(size=10))
            fig.update_layout(showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough data with current filters.")

    with tab2:
        st.subheader("One party's China sentiment over time")
        party_opts_sent = sorted(heatmap.index.tolist()) if not heatmap.empty else []
        if party_opts_sent:
            sel_party_sent = st.selectbox("Select party", party_opts_sent, key="sent_party")
            party_row = heatmap.loc[sel_party_sent].dropna()
            if not party_row.empty:
                line_df = pd.DataFrame({
                    "year": party_row.index.astype(int),
                    "sentiment": party_row.values,
                })
                line_df["label"] = line_df["sentiment"].apply(score_to_label)
                fig = px.line(
                    line_df, x="year", y="sentiment",
                    markers=True,
                    labels={"sentiment": "Avg. China sentiment (0–5)", "year": "Year"},
                    color_discrete_sequence=[PRIMARY_COLOR],
                    hover_data={"label": True},
                )
                fig.add_hline(y=2.5, line_dash="dot", line_color="grey",
                              annotation_text="Neutral")
                fig.update_layout(yaxis=dict(range=[0, 5]))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No sentiment data for this party.")
        else:
            st.warning("Not enough data with current filters.")

    with tab3:
        st.subheader("Distribution of sentiment labels")
        china_df = df[df["china_mentions"] > 0]
        dist = china_df["sentiment_label"].value_counts().reset_index()
        dist.columns = ["label", "count"]
        dist["label_full"] = dist["label"].map(SENTIMENT_LABELS).fillna(dist["label"])
        dist["label_full"] = pd.Categorical(
            dist["label_full"], categories=SENTIMENT_ORDER, ordered=True
        )
        dist = dist.sort_values("label_full")
        fig = px.bar(
            dist, x="label_full", y="count",
            color="label_full",
            color_discrete_sequence=px.colors.diverging.RdYlGn,
            labels={"label_full": "Sentiment", "count": "Number of speeches"},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab_ref:
        st.subheader("Is parliament more negative about China than about other powers?")
        st.markdown(
            "Each bar = avg. China-specific sentiment from speeches that mention China **and** "
            "that power in the same speech. Lower = parliament sounds more negative when "
            "discussing China in that context. The 'China (overall)' bar is the baseline."
        )
        proxy_ref = an.power_sentiment_proxy(df)
        if proxy_ref.empty:
            st.warning("Not enough data under current filters.")
        else:
            REF_COLORS = {
                "China (overall)": PRIMARY_COLOR,
                "China + US":      "#5A8FD6",
                "China + Russia":  "#C0392B",
                "China + EU":      "#F39C12",
                "China + NATO":    "#1A1A1A",
            }
            proxy_ref = proxy_ref.sort_values("avg_sentiment")
            proxy_ref["label"] = proxy_ref["avg_sentiment"].apply(score_to_label)
            fig = px.bar(
                proxy_ref,
                x="avg_sentiment", y="power", orientation="h",
                color="power",
                color_discrete_map=REF_COLORS,
                labels={"avg_sentiment": "Avg. China sentiment (0–5)", "power": ""},
                hover_data={"n_speeches": True, "label": True, "avg_sentiment": ":.2f"},
                text="label",
            )
            fig.add_vline(x=2.5, line_dash="dot", line_color="grey",
                          annotation_text="Neutral", annotation_position="top")
            fig.update_traces(textposition="outside", textfont=dict(size=10))
            fig.update_layout(
                showlegend=False,
                xaxis=dict(range=[0, 4]),
                margin=dict(t=10),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("How has that comparison shifted over time?")
        ptrend_ref = an.power_sentiment_trend(df)
        if not ptrend_ref.empty:
            TREND_COLORS = {
                "China (overall)": PRIMARY_COLOR,
                "China + US":      "#5A8FD6",
                "China + Russia":  "#C0392B",
                "China + EU":      "#F39C12",
            }
            fig2 = px.line(
                ptrend_ref, x="year", y="avg_sentiment",
                color="power", markers=True,
                color_discrete_map=TREND_COLORS,
                labels={"avg_sentiment": "Avg. China sentiment (0–5)",
                        "year": "Year", "power": "Context"},
                hover_data={"n_speeches": True},
            )
            fig2.add_hline(y=2.5, line_dash="dot", line_color="grey",
                           annotation_text="Neutral")
            fig2.update_layout(yaxis=dict(range=[0, 5]))
            st.plotly_chart(fig2, use_container_width=True)
            st.caption(
                "Note: sentiment here is China-specific — computed only from sentences "
                "mentioning China. A lower score when Russia is co-mentioned reflects "
                "a more hawkish framing, not a negative assessment of Russia itself."
            )

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
            combos_plot = combos.copy()
            combos_plot["sentiment_label"] = combos_plot["mean_china_sentiment"].apply(score_to_label)
            fig = px.bar(
                combos_plot.sort_values("n_speeches"),
                x="n_speeches", y="combination", orientation="h",
                color="mean_china_sentiment",
                color_continuous_scale="RdYlGn",
                range_color=[0, 5],
                labels={"n_speeches": "Number of speeches",
                        "combination": "",
                        "mean_china_sentiment": "China sentiment"},
                hover_data={"mean_china_sentiment": ":.2f",
                            "sentiment_label": True,
                            "mean_sentiment_avg": ":.2f"},
            )
            fig.update_layout(coloraxis_colorbar=dict(
                title="Sentiment",
                tickvals=[0, 1.25, 2.5, 3.75, 5],
                ticktext=["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"],
            ))
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("Underlying numbers"):
                st.dataframe(combos, use_container_width=True)

    with tab_trend:
        gp_ymode = st.radio(
            "Y-axis",
            ["Co-occurrence frequency (%)", "Avg. China sentiment"],
            horizontal=True,
            key="gp_ymode",
            help=(
                "**Co-occurrence frequency** = % of China speeches that year which also "
                "mention this power.  "
                "**Avg. China sentiment** = tone of speeches that mention China *and* this power."
            ),
        )
        POWER_COLOR_MAP = {
            "US": PRIMARY_COLOR, "RUSSIA": "#C0392B",
            "EU": ACCENT_COLOR, "NATO": "#1A1A1A",
        }
        show_events_gp = st.checkbox("Show policy milestones", value=True, key="ev_gp")
        if gp_ymode == "Co-occurrence frequency (%)":
            st.caption(
                "% of China-mentioning speeches that year which also mention each power."
            )
            cooc = an.great_power_cooccurrence(df)
            if not cooc.empty:
                fig = px.line(
                    cooc, x="year", y="cooccurrence_pct",
                    color="power", markers=True,
                    labels={"cooccurrence_pct": "% of China speeches also mentioning",
                            "year": "Year", "power": "Power"},
                    color_discrete_map=POWER_COLOR_MAP,
                )
                if show_events_gp:
                    add_policy_lines(fig, x_type="year_frac")
                fig.update_layout(height=460, margin=dict(t=50))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption(
                "Average China-specific sentiment in speeches that mention "
                "China *and* each power. Below 2.5 = negative tone toward China."
            )
            ptrend = an.power_sentiment_trend(df)
            ptrend_lines = ptrend[ptrend["power"] != "China (overall)"].copy()
            ptrend_lines["power_short"] = ptrend_lines["power"].str.replace("China + ", "", regex=False)
            if not ptrend_lines.empty:
                fig = px.line(
                    ptrend_lines, x="year", y="avg_sentiment",
                    color="power_short", markers=True,
                    labels={"avg_sentiment": "Avg. China sentiment (0–5)",
                            "year": "Year", "power_short": "Also mentions"},
                    color_discrete_map={
                        "US": PRIMARY_COLOR, "Russia": "#C0392B",
                        "EU": ACCENT_COLOR, "NATO": "#1A1A1A",
                    },
                    hover_data={"n_speeches": True},
                )
                fig.add_hline(y=2.5, line_dash="dot", line_color="grey",
                              annotation_text="Neutral")
                if show_events_gp:
                    add_policy_lines(fig, x_type="year_frac")
                fig.update_layout(yaxis=dict(range=[0, 5]), height=460, margin=dict(t=50))
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
            "**X axis** = share of this person's total speeches that mention China. "
            "**Y axis** = average China-specific sentiment (below 2.5 = negative). "
            "**Bubble size** = influence score (China mentions × role weight: "
            "Minister/Secretary of State = ×3, Chair = ×1.5, others = ×1). "
            "Labels shown for above-median influence only. **Colour** = party."
        )
        st.markdown(
            "Bubble size reflects institutional importance: it combines how often a speaker mentions China "
            "with the decision-making power of their role — giving a sense of who is both close to the topic "
            "and positioned to act on it."
        )
        bubble_df = speakers.dropna(subset=["avg_china_sentiment", "china_pct"])
        if not bubble_df.empty:
            ctx_badge = {"minister": " [Min.]", "head": " [Chair]",
                         "secretaryOfState": " [Sec.]", "deputyHead": " [Dep.]"}
            ctx_weight = {"minister": 3.0, "head": 1.5,
                          "secretaryOfState": 3.0, "deputyHead": 1.5}
            bubble_df = bubble_df.copy()
            bubble_df["role_weight"] = bubble_df["speaker_context"].map(ctx_weight).fillna(1.0)
            bubble_df["influence_score"] = (
                bubble_df["total_china_mentions"] * bubble_df["role_weight"]
            ).round(1)
            bubble_df["label"] = bubble_df.apply(
                lambda r: r["speaker_name"] + ctx_badge.get(r.get("speaker_context", ""), ""),
                axis=1,
            )
            # Only label top-influence speakers to avoid clutter
            median_infl = bubble_df["influence_score"].median()
            bubble_df["show_label"] = bubble_df.apply(
                lambda r: r["label"] if r["influence_score"] >= median_infl else "",
                axis=1,
            )
            avg_china_pct = bubble_df["china_pct"].mean()

            fig = px.scatter(
                bubble_df,
                x="china_pct", y="avg_china_sentiment",
                size="influence_score", color="party",
                text="show_label",
                hover_data={"china_speeches": True, "total_speeches": True,
                            "china_pct": ":.1f", "avg_china_sentiment": ":.2f",
                            "influence_score": ":.1f", "role_weight": ":.1f",
                            "show_label": False},
                labels={"china_pct": "% of speeches mentioning China",
                        "avg_china_sentiment": "Avg. China sentiment (0–5)",
                        "party": "Party", "influence_score": "Influence score"},
                size_max=50,
            )
            fig.add_hline(y=2.5, line_dash="dot", line_color="grey",
                          annotation_text="Neutral sentiment")
            fig.add_vline(x=avg_china_pct, line_dash="dot", line_color="lightgrey",
                          annotation_text="Avg. China focus")
            fig.update_traces(textposition="top center", textfont=dict(size=9))
            fig.update_layout(height=580)
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
                label_full = SENTIMENT_LABELS.get(s.get("sentiment_label", ""), s.get("sentiment_label", "—") or "—")
                with st.expander(
                    f"{str(s['date'])[:10]}  ·  {s.get('topic','') or '—'}  ·  "
                    f"{label_full}"
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

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Policy & Geopolitics
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Policy & Geopolitics":
    st.title("Policy & Geopolitical Topics")
    st.markdown(
        "This is where the story lives. The most powerful tool here is the AI narrative — "
        "it lets you zoom in on the institutional context within any timeframe of your choice. "
        "Below are the broader policy domains and geopolitical themes that co-occur with China "
        "in Dutch parliamentary debate."
    )

    # ── Topic definitions ──────────────────────────────────────────────────────
    POLICY_TOPICS = {
        "Human Rights": ["uyghur", "uighur", "xinjiang", "tibet", "falun gong",
                         "human rights", "detention", "forced labour"],
        "Trade & Economy": ["trade", "belt and road", "made in china", "export",
                            "import", "economic", "investment", "renminbi"],
        "Technology": ["huawei", "5g", "tiktok", "zte", "alibaba", "tencent",
                       "bytedance", "semiconductor", "tech"],
        "Security & Military": ["military", "security", "threat", "defence",
                                 "south china sea", "taiwan", "pla", "nato"],
        "Diplomacy": ["diplomat", "ambassador", "summit", "relations", "bilateral",
                      "one china", "xi jinping", "beijing"],
    }

    @st.cache_data(show_spinner=False)
    def build_topic_df(_df):
        china = _df[_df["china_mentions"] > 0].copy()
        rows = []
        for topic, keywords in POLICY_TOPICS.items():
            mask = china["text"].str.lower().str.contains(
                "|".join(keywords), regex=True, na=False
            )
            subset = china[mask]
            if subset.empty:
                continue
            for year, grp in subset.groupby("year"):
                rows.append({
                    "topic": topic,
                    "year": year,
                    "n_speeches": len(grp),
                    "avg_sentiment": round(grp["china_sentiment_avg"].mean(), 3),
                })
        return pd.DataFrame(rows)

    topic_df = build_topic_df(df)

    tab_overview, tab_time, tab_ai = st.tabs([
        "Topic breakdown", "Over time", "AI narrative"
    ])

    with tab_overview:
        st.subheader("How much does each policy domain appear — and how negative?")
        agg = topic_df.groupby("topic").agg(
            total_speeches=("n_speeches", "sum"),
            avg_sentiment=("avg_sentiment", "mean"),
        ).reset_index()
        agg["sentiment_label"] = agg["avg_sentiment"].apply(score_to_label)
        fig = px.bar(
            agg.sort_values("total_speeches"),
            x="total_speeches", y="topic", orientation="h",
            color="avg_sentiment",
            color_continuous_scale="RdYlGn", range_color=[0, 5],
            labels={"total_speeches": "Speeches", "topic": "",
                    "avg_sentiment": "Avg sentiment"},
            hover_data={"sentiment_label": True, "avg_sentiment": ":.2f"},
            text="sentiment_label",
        )
        fig.update_traces(textposition="outside", textfont=dict(size=10))
        fig.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab_time:
        st.subheader("How has each topic's prominence shifted over time?")
        show_events_pol = st.checkbox("Show policy milestones", value=True, key="ev_pol")
        if not topic_df.empty:
            fig = px.line(
                topic_df, x="year", y="n_speeches", color="topic",
                markers=True,
                labels={"n_speeches": "Speeches", "year": "Year", "topic": "Topic"},
                color_discrete_sequence=COLOR_PALETTE + ["#C0392B", "#27AE60"],
            )
            if show_events_pol:
                add_policy_lines(fig, x_type="year_frac")
            fig.update_layout(height=460, margin=dict(t=50))
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Sentiment per topic over time")
            topic_sel = st.selectbox("Select topic", list(POLICY_TOPICS.keys()))
            t_df = topic_df[topic_df["topic"] == topic_sel].dropna(subset=["avg_sentiment"])
            if not t_df.empty:
                fig2 = px.bar(
                    t_df, x="year", y="avg_sentiment",
                    color="avg_sentiment",
                    color_continuous_scale="RdYlGn", range_color=[0, 5],
                    labels={"avg_sentiment": "Avg. China sentiment (0–5)", "year": ""},
                )
                fig2.add_hline(y=2.5, line_dash="dot", line_color="grey",
                               annotation_text="Neutral")
                if show_events_pol:
                    add_policy_lines(fig2, x_type="year_frac")
                fig2.update_layout(
                    showlegend=False, coloraxis_showscale=False,
                    height=380, margin=dict(t=50),
                )
                st.plotly_chart(fig2, use_container_width=True)

    with tab_ai:
        st.subheader("AI narrative analysis")

        # ── Try to load Anthropic ──────────────────────────────────────────────
        try:
            import anthropic
            _api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
        except Exception:
            _api_key = ""

        @st.cache_data(show_spinner="Generating narrative...", ttl=3600)
        def call_claude(prompt_text, key):
            client = anthropic.Anthropic(api_key=key)
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=700,
                messages=[{"role": "user", "content": prompt_text}],
            )
            return msg.content[0].text

        if not _api_key:
            st.warning(
                "No API key configured. To enable AI narratives:\n\n"
                "1. Create `.streamlit/secrets.toml`\n"
                "2. Add: `ANTHROPIC_API_KEY = \"sk-ant-...\"`\n"
                "3. Restart Streamlit."
            )
            st.info(
                "**What this panel does:** selects the most relevant speeches for a "
                "chosen topic and year, sends them to Claude, and returns a plain-language "
                "summary of the dominant political narrative — bridging sentiment data "
                "with actionable geopolitical insight."
            )
        else:
            # ── Analysis mode ──────────────────────────────────────────────────
            analysis_mode = st.radio(
                "Analysis mode",
                ["Topic × Year", "Policy event deep-dive",
                 "Single party", "Compare parties"],
                horizontal=True,
                key="ai_mode",
                help=(
                    "**Topic × Year** — dominant framing across all parties for a topic+year.  "
                    "**Policy event deep-dive** — pick a milestone; Claude analyses what "
                    "parliament was debating in the ±6 months around that event.  "
                    "**Single party** — one party's stance on a topic.  "
                    "**Compare parties** — Claude contrasts 2–4 parties."
                ),
            )

            # ── Controls (conditional on mode) ────────────────────────────────
            col_t, col_y = st.columns(2)
            if analysis_mode == "Policy event deep-dive":
                event_labels = [f"{e['short']} ({e['date'][:4]})" for e in POLICY_EVENTS]
                with col_t:
                    sel_event_label = st.selectbox("Policy event", event_labels, key="ai_event")
                sel_event = POLICY_EVENTS[event_labels.index(sel_event_label)]
                ai_topic = None
                ai_year = int(sel_event["date"][:4])
                with col_y:
                    st.markdown(
                        f"**{sel_event['label']}**  \n"
                        f"*{sel_event['date']} · {sel_event['category']}*"
                    )
                    st.caption(sel_event.get("description", ""))
            else:
                with col_t:
                    ai_topic = st.selectbox("Topic", list(POLICY_TOPICS.keys()), key="ai_topic")
                with col_y:
                    ai_year = st.selectbox(
                        "Year", sorted(df["year"].unique().tolist(), reverse=True), key="ai_year"
                    )

            # ── Build pool + party selector (mode-dependent) ──────────────────
            china_filtered = df[df["china_mentions"] > 0].copy()
            selected_parties = []

            if analysis_mode == "Policy event deep-dive":
                # Window: ±6 months around the event date
                ev_date = pd.to_datetime(sel_event["date"])
                pool = china_filtered[
                    (china_filtered["date"] >= ev_date - pd.DateOffset(months=6)) &
                    (china_filtered["date"] <= ev_date + pd.DateOffset(months=6))
                ]
                can_generate = not pool.empty
                if pool.empty:
                    st.warning("No China speeches found in the ±6 month window around this event.")
            else:
                keywords = POLICY_TOPICS[ai_topic]
                topic_mask = china_filtered["text"].str.lower().str.contains(
                    "|".join(keywords), regex=True, na=False
                )
                pool = china_filtered[topic_mask & (china_filtered["year"] == ai_year)]
                available_parties = sorted(pool["party"].dropna().unique().tolist())

                if analysis_mode == "Single party":
                    if available_parties:
                        sel_party_ai = st.selectbox(
                            "Select party", available_parties, key="ai_party_single"
                        )
                        selected_parties = [sel_party_ai]
                    else:
                        st.warning("No speeches found for this topic and year.")
                elif analysis_mode == "Compare parties":
                    if len(available_parties) >= 2:
                        selected_parties = st.multiselect(
                            "Select parties to compare (2–4 recommended)",
                            available_parties,
                            default=available_parties[:3] if len(available_parties) >= 3 else available_parties[:2],
                            max_selections=4,
                            key="ai_party_multi",
                        )
                        if len(selected_parties) < 2:
                            st.caption("Select at least 2 parties to enable comparison.")
                    else:
                        st.warning("Not enough parties with speeches for this topic and year.")

                can_generate = (
                    analysis_mode == "Topic × Year"
                    or (analysis_mode == "Single party" and len(selected_parties) == 1)
                    or (analysis_mode == "Compare parties" and len(selected_parties) >= 2)
                )

            if st.button("Generate narrative", type="primary", disabled=not can_generate):
                # Build sample based on mode
                if analysis_mode == "Policy event deep-dive":
                    sample = pool.sort_values("date").head(30)
                elif analysis_mode == "Topic × Year":
                    sample = pool.sort_values("china_sentiment_avg").head(25)
                elif analysis_mode == "Single party":
                    sample = pool[pool["party"] == selected_parties[0]].head(25)
                else:  # Compare parties
                    sample = pd.concat([
                        pool[pool["party"] == p].head(10) for p in selected_parties
                    ])

                if sample.empty:
                    st.warning("No speeches match this combination.")
                else:
                    context_lines = []
                    for _, row in sample.iterrows():
                        snippet = row["text"][:300].replace("\n", " ")
                        context_lines.append(
                            f"[{str(row['date'])[:10]} | {row['speaker_name']} | {row['party']}] {snippet}"
                        )
                    context = "\n\n".join(context_lines)

                    if analysis_mode == "Policy event deep-dive":
                        prompt = (
                            f"You are a political analyst studying Dutch parliamentary debate on China.\n\n"
                            f"The policy event you are analysing: **{sel_event['label']}** "
                            f"(date: {sel_event['date']}, category: {sel_event['category']}).\n\n"
                            f"Below are {len(sample)} speech excerpts from the Dutch parliament in the "
                            f"6 months before and after this event. Each starts with [date | speaker | party].\n\n"
                            f"{context}\n\n"
                            f"Write a policy briefing (4 paragraphs) that:\n"
                            f"1. Describes the parliamentary debate leading up to this event — "
                            f"what were the key concerns and which parties were most vocal?\n"
                            f"2. Analyses how the event appears to have shifted the debate — "
                            f"did it confirm fears, trigger new arguments, or pass unnoticed?\n"
                            f"3. Identifies the most notable individual contributions — cite speakers by name\n"
                            f"4. Draws one strategic insight: does Dutch parliament *react* to events like this, "
                            f"or do the debates *precede* them?\n\n"
                            f"Write for a senior policy audience. Be specific, analytical, avoid vague generalities."
                        )
                    elif analysis_mode == "Topic × Year":
                        prompt = (
                            f"You are analysing Dutch parliamentary speeches about China "
                            f"from {ai_year}, focused on the topic of '{ai_topic}'.\n\n"
                            f"Below are {len(sample)} speech excerpts. Each starts with "
                            f"[date | speaker | party].\n\n"
                            f"{context}\n\n"
                            f"Write a concise analytical narrative (3–4 paragraphs) that:\n"
                            f"1. Describes the dominant political framing of China on {ai_topic}\n"
                            f"2. Notes which parties drive the tone and why\n"
                            f"3. Highlights any notable shift, tension, or consensus\n"
                            f"4. Ends with one sentence on the geopolitical implication for the Netherlands\n\n"
                            f"Write for a policy audience. Be specific, cite parties, avoid vague generalities."
                        )
                    elif analysis_mode == "Single party":
                        party_name = selected_parties[0]
                        prompt = (
                            f"You are analysing Dutch parliamentary speeches about China "
                            f"from {ai_year}, focused on the topic of '{ai_topic}', "
                            f"as debated by {party_name}.\n\n"
                            f"Below are {len(sample)} speech excerpts from {party_name} members. "
                            f"Each starts with [date | speaker | party].\n\n"
                            f"{context}\n\n"
                            f"Write a focused analytical profile (3 paragraphs) that:\n"
                            f"1. Describes {party_name}'s specific framing of China on {ai_topic}\n"
                            f"2. Identifies the key speakers and their argumentative angle\n"
                            f"3. Assesses whether {party_name}'s position is ideologically consistent, "
                            f"pragmatic, or reactive to events — and what this implies strategically\n\n"
                            f"Write for a policy audience. Be specific, cite speakers by name."
                        )
                    else:  # Compare parties
                        party_list = ", ".join(selected_parties)
                        prompt = (
                            f"You are analysing Dutch parliamentary speeches about China "
                            f"from {ai_year}, focused on the topic of '{ai_topic}'. "
                            f"Compare the positions of: {party_list}.\n\n"
                            f"Below are speech excerpts from these parties (up to 10 per party). "
                            f"Each starts with [date | speaker | party].\n\n"
                            f"{context}\n\n"
                            f"Write a comparative analytical narrative (4 paragraphs) that:\n"
                            f"1. Summarises how each party frames China on {ai_topic} — similarities and differences\n"
                            f"2. Identifies the sharpest point of disagreement between the parties\n"
                            f"3. Notes any surprising agreement or unexpected alignment\n"
                            f"4. Ends with what these differences reveal about Dutch political fault lines "
                            f"on China policy in {ai_year}\n\n"
                            f"Write for a policy audience. Cite parties and individual speakers by name."
                        )

                    narrative = call_claude(prompt, _api_key)
                    st.markdown("---")
                    st.markdown(narrative)
                    if analysis_mode == "Policy event deep-dive":
                        mode_label = sel_event["short"]
                        topic_label = f"±6 months around {sel_event['date']}"
                    elif analysis_mode == "Single party":
                        mode_label = selected_parties[0]
                        topic_label = f"{ai_topic} · {ai_year}"
                    elif analysis_mode == "Compare parties":
                        mode_label = " vs ".join(selected_parties)
                        topic_label = f"{ai_topic} · {ai_year}"
                    else:
                        mode_label = "Full parliament"
                        topic_label = f"{ai_topic} · {ai_year}"
                    st.caption(
                        f"Generated by Claude Haiku · {len(sample)} speeches · "
                        f"{topic_label} · {mode_label} · Cached 1 hour"
                    )
