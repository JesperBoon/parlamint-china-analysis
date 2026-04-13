"""
dry_run_lemmas.py

Scans the existing speeches.parquet `text` column for candidate China lemmas
and reports hits per term, so we can spot false positives BEFORE re-parsing.

Substring-based (lowercase). Approximate, but fast: ~seconds, not minutes.

Run with: python3 scripts/dry_run_lemmas.py
"""

import os
import pandas as pd

PARQUET = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed", "speeches.parquet"
)

CURRENT = [
    "china", "chinese", "beijing", "peking", "xi", "huawei",
    "taiwan", "hong kong", "xinjiang", "tibet", "ccp", "bri",
]

CANDIDATES = {
    "cities": [
        "shanghai", "shenzhen", "guangzhou", "chongqing", "wuhan",
        "chengdu", "tianjin", "nanjing", "macau", "macao",
    ],
    "politics/economics": [
        "belt and road", "one china", "south china sea",
        "renminbi", "yuan", "made in china", "wolf warrior",
        "dual circulation",
    ],
    "human rights": [
        "uyghur", "uighur", "tiananmen", "falun gong",
    ],
    "leaders/institutions": [
        "xi jinping", "mao", "communist party", "pla", "cctv",
    ],
    "tech/companies": [
        "alibaba", "tencent", "tiktok", "bytedance", "zte",
    ],
}


def count_hits(series: pd.Series, term: str) -> tuple[int, int]:
    """Return (speech_count, total_substring_occurrences) for term."""
    term_l = term.lower()
    mask = series.str.contains(term_l, case=False, regex=False, na=False)
    speeches = int(mask.sum())
    totals = int(series[mask].str.lower().str.count(term_l).sum()) if speeches else 0
    return speeches, totals


def main():
    print(f"Loading {PARQUET}...")
    df = pd.read_parquet(PARQUET, columns=["text"])
    texts = df["text"].fillna("")
    print(f"  {len(texts):,} speeches loaded\n")

    print(f"{'TERM':<25} {'SPEECHES':>10} {'TOTAL HITS':>12}")
    print("-" * 50)

    print("\n[CURRENT LEMMAS — baseline]")
    for term in CURRENT:
        s, t = count_hits(texts, term)
        print(f"  {term:<23} {s:>10,} {t:>12,}")

    for category, terms in CANDIDATES.items():
        print(f"\n[CANDIDATES — {category}]")
        for term in terms:
            s, t = count_hits(texts, term)
            flag = ""
            if s > 5000:
                flag = "  <-- HIGH: inspect for false positives"
            elif s == 0:
                flag = "  <-- zero hits"
            print(f"  {term:<23} {s:>10,} {t:>12,}{flag}")

    print("\nNote: substring matching = upper bound. 'yuan' matches")
    print("the currency but also the surname. 'mao' matches the")
    print("leader but also words like 'maori'. Read carefully.")


if __name__ == "__main__":
    main()
