"""
3-merge.py
==========
Merge per-partial-plan R records from two sources to create
an 80-query pool for Figure 5 & 6 reproduction (paper §6.2):

  - Calibration queries (47, templates c/d/e/f):
      ./2-calibration/calib/cp_hashmap_{49,99,149}_raw.json

  - Test queries (33, template b):
      ./2-calibration/test/raw_33b_{49,99,149}.json

Output:
  ./3-repro-figure5-6/raw_80q_{49,99,149}.json

Usage (run from the repository root):
    python 3-merge.py
"""

import json
import os
from collections import defaultdict

CHECKPOINTS   = [49, 99, 149]
CALIB_RAW_DIR = "./2-calibration/calib"
TEST_RAW_DIR  = "./2-calibration/test"
OUTPUT_DIR    = "./3-repro-figure5-6"


def merge_checkpoint(ckpt):
    calib_path = os.path.join(CALIB_RAW_DIR, f"cp_hashmap_{ckpt}_raw.json")
    test_path  = os.path.join(TEST_RAW_DIR,  f"raw_33b_{ckpt}.json")
    out_path   = os.path.join(OUTPUT_DIR,    f"raw_80q_{ckpt}.json")

    calib_recs = json.load(open(calib_path))
    test_recs  = json.load(open(test_path))
    merged     = calib_recs + test_recs

    q_calib  = set(r["qname"] for r in calib_recs)
    q_test   = set(r["qname"] for r in test_recs)
    q_all    = set(r["qname"] for r in merged)
    patterns = set(r["pattern"] for r in merged)

    print(f"  checkpoint_{ckpt}:")
    print(f"    calib  : {len(calib_recs):4d} records, {len(q_calib):2d} queries")
    print(f"    test   : {len(test_recs):4d} records, {len(q_test):2d} queries")
    print(f"    merged : {len(merged):4d} records, {len(q_all):2d} queries, "
          f"{len(patterns):2d} patterns")

    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"    saved  → {out_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Merging calibration (47q) + test (33b) records into 80-query pool...\n")
    for ckpt in CHECKPOINTS:
        merge_checkpoint(ckpt)
    print("\nDone. Next: run 3-repro-figure5-6.ipynb")


if __name__ == "__main__":
    main()
