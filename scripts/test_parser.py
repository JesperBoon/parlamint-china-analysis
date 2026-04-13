"""
test_parser.py

Quick sanity checks for 02_parse_to_df.py logic.
Tests the two new features: china_sentiment_avg and great power co-occurrence.
Run with: python3 scripts/test_parser.py
"""

import sys
import os
import xml.etree.ElementTree as ET
from collections import Counter

# Constants duplicated here since 02_parse_to_df.py can't be imported
# (Python module names can't start with a digit)
CHINA_LEMMAS = {
    "china", "chinese", "beijing", "peking",
    "xi", "huawei", "taiwan", "hong kong", "xinjiang",
    "tibet", "ccp", "bri",
}
GREAT_POWER_LEMMAS = {
    "mentions_us":     {"america", "american", "washington", "biden", "trump", "obama", "usa"},
    "mentions_russia": {"russia", "russian", "moscow", "putin", "kremlin"},
    "mentions_eu":     {"eu", "europe", "european", "brussels"},
    "mentions_nato":   {"nato"},
}

# We can't import directly by filename with hyphen, so inline the test XML
TEI = "http://www.tei-c.org/ns/1.0"
T = lambda tag: f"{{{TEI}}}{tag}"

# ── Helpers ────────────────────────────────────────────────────────────────────

def make_utterance_xml(sentences: list[dict]) -> ET.Element:
    """
    Build a minimal <u> element for testing.
    Each sentence dict: {words: [(text, lemma)], sentiment: float, label: str}
    """
    u = ET.Element(f"{{{TEI}}}u")
    u.set("{http://www.w3.org/XML/1998/namespace}id", "test.u1")
    u.set("who", "#TestSpeaker")
    u.set("ana", "#regular topic:intrel")

    seg = ET.SubElement(u, f"{{{TEI}}}seg")
    seg.set("{http://www.w3.org/XML/1998/namespace}id", "test.seg1")

    for i, sent in enumerate(sentences):
        s = ET.SubElement(seg, f"{{{TEI}}}s")
        s.set("{http://www.w3.org/XML/1998/namespace}id", f"test.s{i}")

        measure = ET.SubElement(s, f"{{{TEI}}}measure")
        measure.set("type", "sentiment")
        measure.set("quantity", str(sent["sentiment"]))
        measure.set("ana", f"senti:{sent['label']}")

        for text, lemma in sent["words"]:
            w = ET.SubElement(s, f"{{{TEI}}}w")
            w.set("lemma", lemma)
            w.text = text

    return u


# We need to replicate parse_utterance locally since we can't import from
# a file named 02_parse_to_df.py directly (starts with digit)
def _parse_utterance_local(u_el, date, chamber, file_id):
    """Local copy of parse_utterance for testing."""
    import re
    speech_id = u_el.get("{http://www.w3.org/XML/1998/namespace}id", "")
    who = u_el.get("who", "").lstrip("#")
    ana = u_el.get("ana", "")
    role = "chair" if "#chair" in ana else "regular"
    topic_match = re.search(r"topic:(\S+)", ana)
    topic = topic_match.group(1) if topic_match else ""

    words = []
    sentiments = []
    sent_labels = []
    china_sentiments = []

    for s_el in u_el.iter(f"{{{TEI}}}s"):
        sent_score = None
        sent_label = ""
        measure = s_el.find(f"{{{TEI}}}measure")
        if measure is not None and measure.get("type") == "sentiment":
            try:
                sent_score = float(measure.get("quantity", 0))
                sentiments.append(sent_score)
            except ValueError:
                pass
            sent_label = measure.get("ana", "").replace("senti:", "")
            if sent_label:
                sent_labels.append(sent_label)

        sentence_lemmas = []
        for w_el in s_el.iter(f"{{{TEI}}}w"):
            if w_el.text:
                lemma = w_el.get("lemma", "").lower()
                words.append((w_el.text.strip(), lemma))
                sentence_lemmas.append(lemma)

        if sent_score is not None and any(l in CHINA_LEMMAS for l in sentence_lemmas):
            china_sentiments.append(sent_score)

    all_lemmas = [lemma for _, lemma in words]
    china_mentions = sum(1 for l in all_lemmas if l in CHINA_LEMMAS)
    china_sentiment_avg = (
        round(sum(china_sentiments) / len(china_sentiments), 3)
        if china_sentiments else None
    )
    sentiment_avg = round(sum(sentiments) / len(sentiments), 3) if sentiments else None
    dominant_label = Counter(sent_labels).most_common(1)[0][0] if sent_labels else ""
    power_counts = {
        col: sum(1 for l in all_lemmas if l in lemma_set)
        for col, lemma_set in GREAT_POWER_LEMMAS.items()
    }

    return {
        "china_sentiment_avg": china_sentiment_avg,
        "sentiment_avg": sentiment_avg,
        "sentiment_label": dominant_label,
        "china_mentions": china_mentions,
        **power_counts,
    }


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_china_sentiment_is_none_when_no_china():
    """Speech without China terms → china_sentiment_avg must be None."""
    u = make_utterance_xml([
        {"words": [("The", "the"), ("minister", "minister"), ("spoke", "speak")],
         "sentiment": 3.0, "label": "neupos"},
    ])
    result = _parse_utterance_local(u, "2020-01-01", "tweedekamer", "test")
    assert result["china_sentiment_avg"] is None, \
        f"Expected None, got {result['china_sentiment_avg']}"
    print("  PASS  test_china_sentiment_is_none_when_no_china")


def test_china_sentiment_only_uses_china_sentences():
    """
    Speech has two sentences:
      - Sentence 1: negative (score 0.3), mentions China
      - Sentence 2: positive (score 4.8), no China mention
    china_sentiment_avg should be 0.3, NOT average of both.
    sentiment_avg should be average of both (2.55).
    """
    u = make_utterance_xml([
        {"words": [("China", "china"), ("is", "be"), ("rising", "rise")],
         "sentiment": 0.3, "label": "negneg"},
        {"words": [("The", "the"), ("economy", "economy"), ("grew", "grow")],
         "sentiment": 4.8, "label": "pospos"},
    ])
    result = _parse_utterance_local(u, "2020-01-01", "tweedekamer", "test")
    assert result["china_sentiment_avg"] == 0.3, \
        f"Expected 0.3, got {result['china_sentiment_avg']}"
    assert result["sentiment_avg"] == round((0.3 + 4.8) / 2, 3), \
        f"Expected {round((0.3+4.8)/2,3)}, got {result['sentiment_avg']}"
    print("  PASS  test_china_sentiment_only_uses_china_sentences")


def test_china_sentiment_averages_multiple_china_sentences():
    """When multiple sentences mention China, average all their scores."""
    u = make_utterance_xml([
        {"words": [("China", "china"), ("builds", "build")],
         "sentiment": 1.0, "label": "neuneg"},
        {"words": [("Chinese", "chinese"), ("trade", "trade")],
         "sentiment": 3.0, "label": "neupos"},
        {"words": [("The", "the"), ("EU", "eu"), ("responds", "respond")],
         "sentiment": 4.0, "label": "mixpos"},
    ])
    result = _parse_utterance_local(u, "2020-01-01", "tweedekamer", "test")
    expected = round((1.0 + 3.0) / 2, 3)
    assert result["china_sentiment_avg"] == expected, \
        f"Expected {expected}, got {result['china_sentiment_avg']}"
    print("  PASS  test_china_sentiment_averages_multiple_china_sentences")


def test_great_power_cooccurrence_counts():
    """Co-occurrence counts should reflect lemma hits, not text hits."""
    u = make_utterance_xml([
        {"words": [("China", "china"), ("and", "and"), ("Russia", "russia"),
                   ("and", "and"), ("Putin", "putin")],
         "sentiment": 2.0, "label": "neuneg"},
        {"words": [("NATO", "nato"), ("responded", "respond")],
         "sentiment": 3.0, "label": "neupos"},
    ])
    result = _parse_utterance_local(u, "2020-01-01", "tweedekamer", "test")
    assert result["mentions_russia"] == 2, \
        f"Expected 2 Russia mentions (russia + putin), got {result['mentions_russia']}"
    assert result["mentions_nato"] == 1, \
        f"Expected 1 NATO mention, got {result['mentions_nato']}"
    assert result["mentions_us"] == 0, \
        f"Expected 0 US mentions, got {result['mentions_us']}"
    print("  PASS  test_great_power_cooccurrence_counts")


def test_great_power_zero_when_absent():
    """Speeches without great power terms should have 0 counts, not None."""
    u = make_utterance_xml([
        {"words": [("The", "the"), ("budget", "budget"), ("passed", "pass")],
         "sentiment": 2.5, "label": "neupos"},
    ])
    result = _parse_utterance_local(u, "2020-01-01", "tweedekamer", "test")
    for col in ["mentions_us", "mentions_russia", "mentions_eu", "mentions_nato"]:
        assert result[col] == 0, f"Expected 0 for {col}, got {result[col]}"
    print("  PASS  test_great_power_zero_when_absent")


# ── Runner ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running parser tests...\n")
    tests = [
        test_china_sentiment_is_none_when_no_china,
        test_china_sentiment_only_uses_china_sentences,
        test_china_sentiment_averages_multiple_china_sentences,
        test_great_power_cooccurrence_counts,
        test_great_power_zero_when_absent,
    ]
    failed = 0
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"  FAIL  {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {test.__name__}: {e}")
            failed += 1

    print(f"\n{len(tests) - failed}/{len(tests)} tests passed.")
    sys.exit(0 if failed == 0 else 1)
