"""
analysis.py

All analytical computations on speeches.parquet.
Each function takes a DataFrame and returns a DataFrame ready for plotting.

Sentiment interpretation note:
    sentiment_avg     = overall tone of the speech (not China-specific)
    china_sentiment_avg = tone of sentences that explicitly mention China.
    Always prefer china_sentiment_avg for claims about attitudes toward China.
    Use sentiment_avg only for general speaker/party tone profiles.
"""

import pandas as pd
import numpy as np


# ── Data loading & cleaning ────────────────────────────────────────────────────

def load(path: str) -> pd.DataFrame:
    """Load parquet, parse dates, clean party labels."""
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Strip prefixes: "party.PVV" → "PVV", "ministry.EZK" → "min.EZK"
    def clean_party(p: str) -> str:
        if p.startswith("party."):
            return p[len("party."):]
        if p.startswith("ministry."):
            return "min." + p[len("ministry."):]
        return p

    df["party"] = df["party"].apply(clean_party)

    # 3-class sentiment derived from 6-class label
    label_to_3 = {
        "negneg": "Negative", "mixneg": "Negative",
        "neuneg": "Neutral",  "neupos": "Neutral",
        "mixpos": "Positive", "pospos": "Positive",
    }
    df["sentiment_3"] = df["sentiment_label"].map(label_to_3).fillna("Unknown")

    return df


def china_only(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to speeches that explicitly mention China (china_mentions > 0)."""
    return df[df["china_mentions"] > 0].copy()


def parties_only(df: pd.DataFrame) -> pd.DataFrame:
    """Remove chairs, ministers and unknown speakers — keep party members only."""
    mask = (
        ~df["party"].str.startswith("min.", na=False) &
        (df["party"] != "TK") &
        (df["party"] != "EK") &
        (df["party"] != "")
    )
    return df[mask].copy()


# ── 1. China mention trend over time ──────────────────────────────────────────

def china_trend(df: pd.DataFrame, freq: str = "Q") -> pd.DataFrame:
    """
    Count of China-mentioning speeches per time period.
    freq: 'Q' = quarter, 'Y' = year, 'M' = month
    Returns: DataFrame with columns [period, china_speeches, total_speeches, pct]
    """
    df = df.copy()
    df["period"] = df["date"].dt.to_period(freq)

    total = df.groupby("period").size().rename("total_speeches")
    china = df[df["china_mentions"] > 0].groupby("period").size().rename("china_speeches")

    result = pd.concat([total, china], axis=1).fillna(0).reset_index()
    result["pct"] = (result["china_speeches"] / result["total_speeches"] * 100).round(2)
    result["period"] = result["period"].astype(str)
    return result


# ── 1b. China trend per party over time ───────────────────────────────────────

def china_trend_by_party(
    df: pd.DataFrame, freq: str = "Y", top_n: int = 8, parties=None
) -> pd.DataFrame:
    """
    China-mention speeches per party per period.
    Returns long-format [period, party, china_speeches, total_speeches, pct].
    If parties is None, auto-selects top_n by total China-speech count.
    """
    df = parties_only(df).copy()
    df["period"] = df["date"].dt.to_period(freq)

    # Total speeches per party per period (needed for pct)
    total_pp = (
        df.groupby(["period", "party"])
        .size()
        .rename("total_speeches")
        .reset_index()
    )

    china = df[df["china_mentions"] > 0].copy()
    if china.empty:
        return china

    if parties is not None:
        top_parties = parties
    else:
        top_parties = china.groupby("party").size().nlargest(top_n).index.tolist()
    china = china[china["party"].isin(top_parties)]

    result = (
        china.groupby(["period", "party"])
        .size()
        .rename("china_speeches")
        .reset_index()
    )
    result = result.merge(total_pp, on=["period", "party"], how="left")
    result["pct"] = (result["china_speeches"] / result["total_speeches"] * 100).round(2)
    result["period"] = result["period"].astype(str)
    return result


def party_sentiment_trend(
    df: pd.DataFrame, parties=None, top_n: int = 8
) -> pd.DataFrame:
    """
    Mean china_sentiment_avg per party per year, long format for trend line charts.
    If parties is None, uses top_n parties by scored China speeches.
    Returns: [year, party, avg_china_sentiment, n_speeches]
    """
    df = parties_only(df)
    df = df[df["china_sentiment_avg"].notna() & (df["china_mentions"] > 0)].copy()
    if df.empty:
        return df

    if parties is not None:
        df = df[df["party"].isin(parties)]
    else:
        top = df.groupby("party")["speech_id"].count().nlargest(top_n).index
        df = df[df["party"].isin(top)]

    return (
        df.groupby(["year", "party"])
        .agg(
            avg_china_sentiment=("china_sentiment_avg", "mean"),
            n_speeches=("speech_id", "count"),
        )
        .round(3)
        .reset_index()
    )


# ── 2. Party frequency — who talks about China most? ──────────────────────────

def china_by_party(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """
    China-mention speeches per party, normalised by total speeches per party.
    Returns top_n parties sorted by normalised rate.
    """
    df = parties_only(df)
    total = df.groupby("party").size().rename("total_speeches")
    china = df[df["china_mentions"] > 0].groupby("party").size().rename("china_speeches")

    result = pd.concat([total, china], axis=1).fillna(0).reset_index()
    result["rate"] = (result["china_speeches"] / result["total_speeches"] * 100).round(2)
    return result.nlargest(top_n, "china_speeches").reset_index(drop=True)


# ── 3. Sentiment shift per party over time ────────────────────────────────────

def sentiment_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean china_sentiment_avg per party per year.
    Only includes speeches that have a china_sentiment_avg score.
    Returns pivot table: rows = party, columns = year, values = mean score.
    Interpretation: lower score = more negative tone toward China.
    """
    df = parties_only(df)
    df = df[df["china_sentiment_avg"].notna()]

    pivot = (
        df.groupby(["party", "year"])["china_sentiment_avg"]
        .mean()
        .round(3)
        .unstack("year")
    )
    # Keep parties with at least 3 data points across all years
    pivot = pivot[pivot.notna().sum(axis=1) >= 3]
    return pivot


# ── 4. Top speakers on China ──────────────────────────────────────────────────

def top_china_speakers(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Speakers ranked by total China mentions.
    Includes china_pct: % of each speaker's total speeches that mention China.
    Includes dominant speaker_context (minister / head / etc).
    """
    total_per_speaker = df.groupby("speaker_name").size().rename("total_speeches")

    china_df = df[df["china_mentions"] > 0].copy()
    result = (
        china_df.groupby(["speaker_name", "party"])
        .agg(
            total_china_mentions=("china_mentions", "sum"),
            china_speeches=("speech_id", "count"),
            avg_china_sentiment=("china_sentiment_avg", "mean"),
            speaker_context=("speaker_context", lambda x: x.mode()[0] if len(x) else ""),
        )
        .reset_index()
        .nlargest(top_n, "total_china_mentions")
    )
    result = result.merge(total_per_speaker, on="speaker_name", how="left")
    result["china_pct"] = (
        result["china_speeches"] / result["total_speeches"] * 100
    ).round(1)
    result["avg_china_sentiment"] = result["avg_china_sentiment"].round(3)
    return result.reset_index(drop=True)


def speaker_speeches(df: pd.DataFrame, speaker_name: str, top_n: int = 50) -> pd.DataFrame:
    """All China-mentioning speeches for one speaker, most recent first."""
    result = (
        df[(df["china_mentions"] > 0) & (df["speaker_name"] == speaker_name)]
        .sort_values("date", ascending=False)
        .head(top_n)
    )
    cols = ["date", "topic", "sentiment_label", "china_sentiment_avg",
            "china_mentions", "text"]
    return result[[c for c in cols if c in result.columns]].copy()


# ── 5. Great power co-occurrence ──────────────────────────────────────────────

def great_power_cooccurrence(df: pd.DataFrame) -> pd.DataFrame:
    """
    In China-mentioning speeches: how often is each great power also mentioned?
    Returns counts and percentages per power, split by year.
    Useful for: 'Is China discussed in relation to Russia/US/EU, and does this change over time?'
    """
    df = df[df["china_mentions"] > 0].copy()
    powers = ["mentions_us", "mentions_russia", "mentions_eu", "mentions_nato"]

    rows = []
    for year, group in df.groupby("year"):
        n = len(group)
        for col in powers:
            cooc = (group[col] > 0).sum()
            rows.append({
                "year": year,
                "power": col.replace("mentions_", "").upper(),
                "cooccurrence_count": int(cooc),
                "cooccurrence_pct": round(cooc / n * 100, 2),
                "china_speeches_that_year": n,
            })

    return pd.DataFrame(rows)


# ── 6. Left-right sentiment split ─────────────────────────────────────────────

# Rough left-right grouping based on Dutch political spectrum
PARTY_BLOC = {
    "VVD": "Right", "CDA": "Right", "PVV": "Right", "FvD": "Right",
    "SGP": "Right", "ChristenUnie": "Center",
    "D66": "Center", "NSC": "Center",
    "PvdA": "Left", "SP": "Left", "GroenLinks": "Left",
    "PvdD": "Left", "DENK": "Left", "Volt": "Left",
}

def sentiment_by_bloc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean china_sentiment_avg per political bloc per year.
    Returns: DataFrame with [year, bloc, mean_china_sentiment, n_speeches]
    """
    df = parties_only(df)
    df = df[df["china_sentiment_avg"].notna()].copy()
    df["bloc"] = df["party"].map(PARTY_BLOC).fillna("Other")
    df = df[df["bloc"] != "Other"]

    return (
        df.groupby(["year", "bloc"])
        .agg(
            mean_china_sentiment=("china_sentiment_avg", "mean"),
            n_speeches=("speech_id", "count"),
        )
        .round(3)
        .reset_index()
    )


# ── 6b. Great power combinations + sentiment ──────────────────────────────────

def china_power_combinations(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each China-mentioning speech, classify which other great powers are
    co-mentioned in the same speech, then aggregate by combination.

    Returns: DataFrame with [combination, n_speeches, mean_china_sentiment, mean_sentiment_avg].
    Combinations are sorted by frequency. Rare ones (<10 speeches) are dropped
    to keep the chart readable.
    """
    df = df[df["china_mentions"] > 0].copy()
    if df.empty:
        return df

    def label(row):
        powers = []
        if row["mentions_us"] > 0: powers.append("US")
        if row["mentions_russia"] > 0: powers.append("Russia")
        if row["mentions_eu"] > 0: powers.append("EU")
        if row["mentions_nato"] > 0: powers.append("NATO")
        if not powers:
            return "China alone"
        return "China + " + " + ".join(powers)

    df["combination"] = df.apply(label, axis=1)

    result = (
        df.groupby("combination")
        .agg(
            n_speeches=("speech_id", "count"),
            mean_china_sentiment=("china_sentiment_avg", "mean"),
            mean_sentiment_avg=("sentiment_avg", "mean"),
        )
        .reset_index()
        .round(3)
    )
    result = result[result["n_speeches"] >= 10]
    return result.sort_values("n_speeches", ascending=False).reset_index(drop=True)


# ── 7. Parliament seat chart data ─────────────────────────────────────────────

# Election-period seat maps.
# TK 2017 (Mar 2017): same composition until Mar 2021
# TK 2021 (Mar 2021): new government composition
# EK 2019 (Jun 2019): FvD surge after European elections

TK_SEATS_2017 = {
    "VVD": 33, "PVV": 20, "CDA": 19, "D66": 19, "GroenLinks": 14,
    "SP": 14, "PvdA": 9, "ChristenUnie": 5, "PvdD": 5, "50PLUS": 4,
    "SGP": 3, "DENK": 3, "FvD": 2,
}
TK_SEATS_2021 = {
    "VVD": 35, "PVV": 17, "D66": 8, "CDA": 14, "GroenLinks": 8,
    "SP": 9, "PvdA": 9, "ChristenUnie": 5, "PvdD": 3, "50PLUS": 1,
    "SGP": 3, "DENK": 3, "FvD": 8, "JA21": 3, "BBB": 1, "Volt": 2,
    "BIJ1": 1,
}
EK_SEATS_2019 = {
    "VVD": 12, "CDA": 9, "D66": 7, "PVV": 5, "GroenLinks": 8,
    "SP": 4, "PvdA": 6, "ChristenUnie": 4, "PvdD": 3, "50PLUS": 2,
    "SGP": 2, "FvD": 12, "OSF": 1,
}

TK_ELECTION_PERIODS = {
    "2017–2021": (None, "2021-03-16", TK_SEATS_2017),
    "2021–2022": ("2021-03-17", None, TK_SEATS_2021),
}
EK_ELECTION_PERIODS = {
    "2019–2022": ("2019-06-12", None, EK_SEATS_2019),
}


def election_composition(df: pd.DataFrame, chamber: str):
    """
    Choose the seat map that best matches the filtered date range.
    Returns (seat_map dict, period_label str).
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    min_date = df["date"].min()
    max_date = df["date"].max()

    if chamber == "tweedekamer":
        for label, (after, before, seats) in TK_ELECTION_PERIODS.items():
            a_ok = after is None or min_date >= pd.to_datetime(after)
            b_ok = before is None or max_date < pd.to_datetime(before)
            if a_ok and b_ok:
                return seats, label
        return TK_SEATS_2017, "2017–2021"
    else:
        return EK_SEATS_2019, "2019–2022"


def seat_chart_data(df: pd.DataFrame, chamber: str = "tweedekamer") -> pd.DataFrame:
    """
    For each parliamentary seat, return China-mention metrics and sentiment.
    Composition is dynamically chosen based on the filtered date range.

    Returns DataFrame with columns:
        [seat_idx, party, rate, china_speeches, total_speeches,
         mean_china_sentiment, sentiment_n, spectrum, period_label]
    """
    seat_map, period_label = election_composition(df, chamber)
    by_party = china_by_party(df, top_n=100).set_index("party")

    sentiment_df = parties_only(df)
    sentiment_df = sentiment_df[sentiment_df["china_sentiment_avg"].notna()]
    sent_agg = (
        sentiment_df.groupby("party")
        .agg(
            mean_china_sentiment=("china_sentiment_avg", "mean"),
            sentiment_n=("speech_id", "count"),
        )
        .round(3)
    )

    LEFT = {"SP", "PvdA", "GroenLinks", "PvdD", "DENK", "BIJ1", "Volt", "OSF"}
    CENTER = {"D66", "ChristenUnie", "NSC"}
    RIGHT = {
        "VVD", "CDA", "PVV", "FvD", "SGP", "JA21",
        "50PLUS", "BBB",
    }

    rows = []
    seat_idx = 0
    for party, n_seats in seat_map.items():
        rate = float(by_party["rate"].get(party, 0.0))
        china = int(by_party["china_speeches"].get(party, 0))
        total = int(by_party["total_speeches"].get(party, 0))
        mcs = float(sent_agg["mean_china_sentiment"].get(party, float("nan")))
        sn = int(sent_agg["sentiment_n"].get(party, 0))
        if party in LEFT:
            spectrum = "Left"
        elif party in CENTER:
            spectrum = "Center"
        else:
            spectrum = "Right"
        for _ in range(n_seats):
            rows.append({
                "seat_idx": seat_idx,
                "party": party,
                "rate": rate,
                "china_speeches": china,
                "total_speeches": total,
                "mean_china_sentiment": mcs,
                "sentiment_n": sn,
                "spectrum": spectrum,
                "period_label": period_label,
            })
            seat_idx += 1
    return pd.DataFrame(rows)


def party_sentiment_summary(df: pd.DataFrame, party: str) -> dict:
    """
    Per-party sentiment summary for the party detail overlay.
    """
    df_p = parties_only(df)
    df_p = df_p[df_p["party"] == party]
    china = df_p[df_p["china_mentions"] > 0]
    yearly = (
        china[china["china_sentiment_avg"].notna()]
        .groupby("year")["china_sentiment_avg"]
        .mean()
        .round(3)
        .to_dict()
    )
    top_topics = (
        china.groupby("topic").size()
        .nlargest(5)
        .to_dict()
    )
    overall = round(china["china_sentiment_avg"].mean(), 3) if not china.empty else None
    return {
        "party": party,
        "mean_china_sentiment": overall,
        "n_china_speeches": int(len(china)),
        "yearly_sentiment": yearly,
        "top_topics": top_topics,
    }


# ── 8. Topic context of China mentions ────────────────────────────────────────

def china_topic_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    In which debate topics does China appear most often?
    Returns topic counts and % among China-mentioning speeches.
    """
    df = df[df["china_mentions"] > 0].copy()
    total = len(df)
    result = (
        df.groupby("topic")
        .agg(count=("speech_id", "count"))
        .reset_index()
        .sort_values("count", ascending=False)
    )
    result["pct"] = (result["count"] / total * 100).round(2)
    return result.reset_index(drop=True)
