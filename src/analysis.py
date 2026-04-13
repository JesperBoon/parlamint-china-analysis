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


# ── 7. Topic context of China mentions ────────────────────────────────────────

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
