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

def china_trend_by_party(df: pd.DataFrame, freq: str = "Y", top_n: int = 8) -> pd.DataFrame:
    """
    China-mention speeches per party per period.
    Returns long-format [period, party, china_speeches], limited to top_n
    parties by total China-speech count for chart legibility.
    """
    df = parties_only(df)
    df = df[df["china_mentions"] > 0].copy()
    if df.empty:
        return df
    df["period"] = df["date"].dt.to_period(freq)

    top_parties = df.groupby("party").size().nlargest(top_n).index.tolist()
    df = df[df["party"].isin(top_parties)]

    result = (
        df.groupby(["period", "party"])
        .size()
        .rename("china_speeches")
        .reset_index()
    )
    result["period"] = result["period"].astype(str)
    return result


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
    Speakers ranked by total China mentions, with their party and mean sentiment.
    """
    df = df[df["china_mentions"] > 0].copy()
    result = (
        df.groupby(["speaker_name", "party"])
        .agg(
            total_china_mentions=("china_mentions", "sum"),
            china_speeches=("speech_id", "count"),
            avg_china_sentiment=("china_sentiment_avg", "mean"),
        )
        .reset_index()
        .nlargest(top_n, "total_china_mentions")
    )
    result["avg_china_sentiment"] = result["avg_china_sentiment"].round(3)
    return result.reset_index(drop=True)


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

# 2017 Tweede Kamer composition (covers most of our 2015–2022 data window)
TK_SEATS_2017 = {
    "VVD": 33, "PVV": 20, "CDA": 19, "D66": 19, "GroenLinks": 14,
    "SP": 14, "PvdA": 9, "ChristenUnie": 5, "PvdD": 5, "50PLUS": 4,
    "SGP": 3, "DENK": 3, "FvD": 2,
}
# 2019 Eerste Kamer composition (75 seats)
EK_SEATS_2019 = {
    "FvD": 12, "VVD": 12, "CDA": 9, "D66": 7, "PVV": 5, "GroenLinks": 8,
    "SP": 4, "PvdA": 6, "ChristenUnie": 4, "PvdD": 3, "50PLUS": 2,
    "SGP": 2, "OSF": 1,
}


def seat_chart_data(df: pd.DataFrame, chamber: str = "tweedekamer") -> pd.DataFrame:
    """
    For each parliamentary seat, return the China-mention rate of its party.
    Returns: DataFrame with [seat_idx, party, rate, china_speeches, total_speeches].
    Uses a fixed reference composition (2017 TK / 2019 EK).
    """
    seat_map = TK_SEATS_2017 if chamber == "tweedekamer" else EK_SEATS_2019

    by_party = china_by_party(df, top_n=100).set_index("party")

    rows = []
    seat_idx = 0
    for party, n_seats in seat_map.items():
        rate = float(by_party["rate"].get(party, 0.0))
        china = int(by_party["china_speeches"].get(party, 0))
        total = int(by_party["total_speeches"].get(party, 0))
        for _ in range(n_seats):
            rows.append({
                "seat_idx": seat_idx,
                "party": party,
                "rate": rate,
                "china_speeches": china,
                "total_speeches": total,
            })
            seat_idx += 1
    return pd.DataFrame(rows)


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
