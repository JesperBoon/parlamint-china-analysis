"""
02_parse_to_df.py

Parses all XML files in data/subset/ into one flat DataFrame.
Each row = one speech (utterance).
Saves to data/processed/speeches.parquet

Columns:
    speech_id            unique ID of the utterance
    file_id              source filename (= one parliamentary session)
    date                 date of the session (YYYY-MM-DD)
    year                 year (int)
    chamber              tweedekamer / eerstekamer
    speaker_id           speaker ID from listPerson
    speaker_name         full name
    party                political party
    gender               M / F
    role                 chair / regular
    topic                pre-annotated topic label (e.g. 'lawcr', 'intrel')
    sentiment_avg        mean sentence-level sentiment score across the full speech
    sentiment_label      dominant sentiment label across the full speech
    china_sentiment_avg  mean sentence-level sentiment ONLY for sentences that
                         contain a China-related lemma. NULL when no China sentence
                         exists in the speech.
                         ⚠ Interpretation note: sentiment_avg reflects the overall
                         tone of the speech, NOT China-specific tone. Use
                         china_sentiment_avg for any claim about attitudes toward China.
    text                 reconstructed speech text (space-joined word tokens)
    word_count           number of word tokens
    china_mentions       count of China-related lemmas in this speech
    mentions_us          count of US-related lemmas (america, trump, biden, ...)
    mentions_russia      count of Russia-related lemmas (russia, putin, kremlin, ...)
    mentions_eu          count of EU-related lemmas (eu, europe, european, ...)
    mentions_nato        count of NATO-related lemmas
"""

import os
import re
import glob
from collections import Counter

import xml.etree.ElementTree as ET
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
SUBSET_DIR = os.path.join(BASE, "data", "subset")
OUT_PATH = os.path.join(BASE, "data", "processed", "speeches.parquet")

TEI = "http://www.tei-c.org/ns/1.0"
T = lambda tag: f"{{{TEI}}}{tag}"

# China-related terms. Split into single-token lemmas (fast set lookup)
# and multi-word phrases (substring match on joined lemma sequence).
CHINA_LEMMAS_SINGLE = {
    # Original
    "china", "chinese", "beijing", "peking",
    "xi", "huawei", "taiwan", "xinjiang", "tibet", "ccp", "bri",
    # Cities
    "shanghai", "shenzhen", "wuhan",
    # Human rights
    "uyghur", "uighur",
    # Leaders / institutions
    "mao", "cctv",
    # Tech / companies
    "alibaba", "tencent", "tiktok", "zte",
}
CHINA_PHRASES = {
    "hong kong",
    "belt and road",
    "one china",
    "south china sea",
    "made in china",
    "xi jinping",
    "communist party",
    "falun gong",
}

# Great power co-occurrence lemmas — single-word only (multi-word phrases
# like "united states" are split across tokens so we match on key words)
GREAT_POWER_LEMMAS = {
    "mentions_us":     {"america", "american", "washington", "biden", "trump", "obama", "usa"},
    "mentions_russia": {"russia", "russian", "moscow", "putin", "kremlin"},
    "mentions_eu":     {"eu", "europe", "european", "brussels"},
    "mentions_nato":   {"nato"},
}

# ── Speaker metadata ───────────────────────────────────────────────────────────

def parse_persons(subset_dir: str) -> dict:
    """Returns {speaker_id: {name, gender, affiliations}} from listPerson.xml.
    Each affiliation keeps its (ref, role, from, to) so we can resolve the
    correct party / context by speech date (Fork B)."""
    path = os.path.join(subset_dir, "ParlaMint-NL-en.TEI.ana", "ParlaMint-NL-listPerson.xml")
    if not os.path.exists(path):
        print(f"  [warn] listPerson.xml not found at {path}")
        return {}

    tree = ET.parse(path)
    root = tree.getroot()
    persons = {}

    for person in root.iter(T("person")):
        pid = person.get("{http://www.w3.org/XML/1998/namespace}id", "")

        forename = ""
        surname = ""
        pn = person.find(T("persName"))
        if pn is not None:
            fn = pn.find(T("forename"))
            sn = pn.find(T("surname"))
            forename = fn.text.strip() if fn is not None and fn.text else ""
            surname = sn.text.strip() if sn is not None and sn.text else ""
        name = f"{forename} {surname}".strip() or pid

        sex_el = person.find(T("sex"))
        gender = sex_el.get("value", "") if sex_el is not None else ""

        affiliations = []
        for aff in person.findall(T("affiliation")):
            affiliations.append({
                "ref": aff.get("ref", "").lstrip("#"),
                "role": aff.get("role", ""),
                "from": aff.get("from", ""),
                "to": aff.get("to", ""),
            })

        persons[pid] = {
            "speaker_name": name,
            "gender": gender,
            "affiliations": affiliations,
        }

    print(f"  Loaded {len(persons)} speakers from listPerson.xml")
    return persons


def _active_at(aff: dict, date: str) -> bool:
    """True if affiliation is active on date (YYYY-MM-DD)."""
    f, t = aff["from"], aff["to"]
    if f and date < f:
        return False
    if t and date > t:
        return False
    return True


def resolve_speaker(persons: dict, speaker_id: str, date: str) -> dict:
    """Pick the correct party and speaker_context for this speech date.

    party           → active 'member' affiliation, preferring party.XXX
                      refs over generic chamber refs (TK/EK).
    speaker_context → active non-member role (minister, head, chair, ...),
                      describing the capacity in which the person spoke.
    """
    info = persons.get(speaker_id)
    if not info:
        return {"speaker_name": "", "gender": "", "party": "", "speaker_context": ""}

    active = [a for a in info["affiliations"] if _active_at(a, date)] if date else []

    party = ""
    for a in active:
        if a["role"] == "member" and a["ref"].startswith("party."):
            party = a["ref"]
            break
    if not party:
        for a in active:
            if a["role"] == "member":
                party = a["ref"]
                break

    context = ""
    for a in active:
        if a["role"] and a["role"] != "member":
            context = a["role"]
            break

    return {
        "speaker_name": info["speaker_name"],
        "gender": info["gender"],
        "party": party,
        "speaker_context": context,
    }


# ── Debate XML parser ──────────────────────────────────────────────────────────

def extract_date_chamber(filename: str):
    """Extract date and chamber from filename like ParlaMint-NL-en_2017-02-09-tweedekamer-7.ana.xml"""
    base = os.path.basename(filename)
    m = re.search(r"_(\d{4}-\d{2}-\d{2})-(tweedekamer|eerstekamer)", base)
    if m:
        return m.group(1), m.group(2)
    return "", ""


def parse_utterance(u_el, date: str, chamber: str, file_id: str, persons: dict) -> dict:
    """Extract all relevant fields from one <u> element."""
    speech_id = u_el.get("{http://www.w3.org/XML/1998/namespace}id", "")
    who = u_el.get("who", "").lstrip("#")
    ana = u_el.get("ana", "")

    # Role
    role = "chair" if "#chair" in ana else "regular"

    # Topic — extract first topic:XXX token
    topic_match = re.search(r"topic:(\S+)", ana)
    topic = topic_match.group(1) if topic_match else ""

    # Process sentence by sentence so we can track China-specific sentiment
    words = []
    sentiments = []       # all sentence scores → sentiment_avg
    sent_labels = []      # all sentence labels → sentiment_label
    china_sentiments = [] # only scores from sentences containing a China lemma

    for s_el in u_el.iter(T("s")):
        # Sentence-level sentiment (pre-computed by ParlaSent classifier)
        sent_score = None
        sent_label = ""
        measure = s_el.find(T("measure"))
        if measure is not None and measure.get("type") == "sentiment":
            try:
                sent_score = float(measure.get("quantity", 0))
                sentiments.append(sent_score)
            except ValueError:
                pass
            sent_label = measure.get("ana", "").replace("senti:", "")
            if sent_label:
                sent_labels.append(sent_label)

        # Word tokens — collect text + lemma, flag if sentence contains China term
        sentence_lemmas = []
        for w_el in s_el.iter(T("w")):
            if w_el.text:
                lemma = w_el.get("lemma", "").lower()
                words.append((w_el.text.strip(), lemma))
                sentence_lemmas.append(lemma)

        # If this sentence mentions China AND has a sentiment score, record it
        if sent_score is not None:
            sent_lemma_str = " ".join(sentence_lemmas)
            if (any(l in CHINA_LEMMAS_SINGLE for l in sentence_lemmas)
                    or any(p in sent_lemma_str for p in CHINA_PHRASES)):
                china_sentiments.append(sent_score)

    text = " ".join(w for w, _ in words)
    word_count = len(words)
    all_lemmas = [lemma for _, lemma in words]
    all_lemmas_str = " ".join(all_lemmas)

    # China mentions: single-token matches + phrase occurrences
    china_mentions = (
        sum(1 for l in all_lemmas if l in CHINA_LEMMAS_SINGLE)
        + sum(all_lemmas_str.count(p) for p in CHINA_PHRASES)
    )

    # China-specific sentiment — None when no China sentence in this speech
    china_sentiment_avg = (
        round(sum(china_sentiments) / len(china_sentiments), 3)
        if china_sentiments else None
    )

    # Overall sentiment aggregates
    sentiment_avg = round(sum(sentiments) / len(sentiments), 3) if sentiments else None
    dominant_label = Counter(sent_labels).most_common(1)[0][0] if sent_labels else ""

    # Great power co-occurrence counts
    power_counts = {
        col: sum(1 for l in all_lemmas if l in lemma_set)
        for col, lemma_set in GREAT_POWER_LEMMAS.items()
    }

    resolved = resolve_speaker(persons, who, date)

    return {
        "speech_id": speech_id,
        "file_id": file_id,
        "date": date,
        "year": int(date[:4]) if date else None,
        "chamber": chamber,
        "speaker_id": who,
        "speaker_name": resolved["speaker_name"],
        "party": resolved["party"],
        "gender": resolved["gender"],
        "speaker_context": resolved["speaker_context"],
        "role": role,
        "topic": topic,
        "sentiment_avg": sentiment_avg,
        "sentiment_label": dominant_label,
        "china_sentiment_avg": china_sentiment_avg,
        "text": text,
        "word_count": word_count,
        "china_mentions": china_mentions,
        **power_counts,
    }


def parse_debate_file(filepath: str, persons: dict) -> list[dict]:
    """Parse one session XML → list of speech dicts."""
    file_id = os.path.basename(filepath).replace(".ana.xml", "")
    date, chamber = extract_date_chamber(filepath)

    try:
        tree = ET.parse(filepath)
    except ET.ParseError as e:
        print(f"  [warn] Parse error in {file_id}: {e}")
        return []

    root = tree.getroot()
    rows = []
    for u_el in root.iter(T("u")):
        rows.append(parse_utterance(u_el, date, chamber, file_id, persons))
    return rows


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    # Load speaker metadata
    print("Loading speaker metadata...")
    persons = parse_persons(SUBSET_DIR)

    # Find all debate XMLs in subset
    pattern = os.path.join(SUBSET_DIR, "**", "*.ana.xml")
    files = [f for f in glob.glob(pattern, recursive=True)
             if "listPerson" not in f and "listOrg" not in f
             and "ParlaMint-NL-en.ana" not in os.path.basename(f)
             and "taxonomy" not in f]

    print(f"Parsing {len(files)} debate files...")
    all_rows = []

    for i, filepath in enumerate(sorted(files), 1):
        rows = parse_debate_file(filepath, persons)
        all_rows.extend(rows)
        if i % 100 == 0:
            print(f"  ... {i}/{len(files)} files, {len(all_rows)} speeches so far")

    print(f"\nBuilding DataFrame ({len(all_rows)} speeches)...")
    df = pd.DataFrame(all_rows)

    # Reorder columns cleanly (speaker metadata already resolved per-row)
    cols = [
        "speech_id", "file_id", "date", "year", "chamber",
        "speaker_id", "speaker_name", "party", "gender",
        "role", "speaker_context",
        "topic", "sentiment_avg", "sentiment_label", "china_sentiment_avg",
        "text", "word_count", "china_mentions",
        "mentions_us", "mentions_russia", "mentions_eu", "mentions_nato",
    ]
    df = df[[c for c in cols if c in df.columns]]

    # Save CSV checkpoint first (no dependencies)
    csv_path = OUT_PATH.replace(".parquet", ".csv")
    df.to_csv(csv_path, index=False)
    print(f"CSV checkpoint saved to: {csv_path}")

    # Save parquet
    df.to_parquet(OUT_PATH, index=False)
    print(f"Parquet saved to: {OUT_PATH}")
    print(f"\nDataFrame shape: {df.shape}")
    print(f"\nColumn overview:")
    print(df.dtypes.to_string())
    print(f"\nSample row (first China mention):")
    china_rows = df[df["china_mentions"] > 0]
    if not china_rows.empty:
        print(china_rows.iloc[0][["date", "speaker_name", "party", "topic",
                                   "sentiment_label", "china_sentiment_avg",
                                   "china_mentions", "text"]].to_string())

    print(f"\nchina_sentiment_avg coverage:")
    has_china_sent = df["china_sentiment_avg"].notna().sum()
    print(f"  Speeches with China sentiment score: {has_china_sent} ({has_china_sent/len(df)*100:.1f}%)")

    print(f"\nGreat power co-occurrence (speeches with at least 1 mention):")
    for col in ["mentions_us", "mentions_russia", "mentions_eu", "mentions_nato"]:
        n = (df[col] > 0).sum()
        print(f"  {col}: {n} ({n/len(df)*100:.1f}%)")


if __name__ == "__main__":
    main()
