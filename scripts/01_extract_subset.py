"""
01_extract_subset.py

Streams through ParlaMint-NL-en.ana.tgz and extracts only debate files
that contain China-related terms, plus all metadata files.
Nothing is fully unpacked — only matching files land on disk.
"""

import tarfile
import os
import sys

TGZ_PATH = os.path.expanduser("~/Downloads/ParlaMint-NL-en.ana.tgz")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "subset")
OUT_DIR = os.path.normpath(OUT_DIR)

CHINA_TERMS = [
    b"china", b"chinese", b"beijing", b"peking",
    b"xi jinping", b"huawei", b"belt and road", b"bri",
    b"taiwan", b"hong kong", b"xinjiang", b"tibet",
    b"ccp", b"communist party of china",
]

# Files to always extract regardless of content
METADATA_NAMES = {
    "ParlaMint-NL-listPerson.xml",
    "ParlaMint-NL-listOrg.xml",
    "ParlaMint-NL-en.ana.xml",
    "ParlaMint-taxonomy-sentiment.ana.xml",
    "README-NL-en.ana.md",
}


def contains_china_term(content_bytes: bytes) -> bool:
    lower = content_bytes.lower()
    return any(term in lower for term in CHINA_TERMS)


def extract_member(tar: tarfile.TarFile, member: tarfile.TarInfo, out_dir: str):
    dest = os.path.join(out_dir, member.name)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    f = tar.extractfile(member)
    if f is None:
        return
    with open(dest, "wb") as out:
        out.write(f.read())


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Reading: {TGZ_PATH}")
    print(f"Writing subset to: {OUT_DIR}\n")

    total = 0
    matched = 0
    metadata_saved = 0

    with tarfile.open(TGZ_PATH, "r:gz") as tar:
        for member in tar:
            if member.isdir():
                continue

            filename = os.path.basename(member.name)
            total += 1

            # Always save metadata files
            if filename in METADATA_NAMES:
                extract_member(tar, member, OUT_DIR)
                metadata_saved += 1
                print(f"  [meta]  {member.name}")
                continue

            # Only process debate XML files
            if not filename.endswith(".ana.xml"):
                continue

            # Read and check for China terms
            f = tar.extractfile(member)
            if f is None:
                continue
            content = f.read()

            if contains_china_term(content):
                dest = os.path.join(OUT_DIR, member.name)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with open(dest, "wb") as out:
                    out.write(content)
                matched += 1
                print(f"  [match] {member.name}  ({len(content):,} bytes)")
            else:
                # Progress indicator every 500 files
                if total % 500 == 0:
                    print(f"  ... scanned {total} files, {matched} matches so far")

    print(f"\nDone.")
    print(f"  Total files scanned : {total}")
    print(f"  Metadata files saved: {metadata_saved}")
    print(f"  China-related files : {matched}")
    print(f"  Subset location     : {OUT_DIR}")


if __name__ == "__main__":
    main()
