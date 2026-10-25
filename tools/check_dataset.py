#!/usr/bin/env python
"""Verify that data/raw contains the right CICIDS2017 files (GeneratedLabelledFlows / TrafficLabelling).
Usage: python tools/check_dataset.py [data/raw]"""
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw")
files = sorted(root.rglob("*.csv"))
if not files:
    sys.exit(f"No CSV files under {root}. Unzip GeneratedLabelledFlows.zip there.")
need = ["flow id", "source ip", "destination ip", "timestamp", "label"]
ok = True
for f in files:
    hdr = None
    for enc in ("cp1252", "utf-8", "latin-1"):
        try:
            hdr = pd.read_csv(f, nrows=2, encoding=enc, skipinitialspace=True)
            break
        except UnicodeDecodeError:
            continue
    cols = [str(c).strip().lower() for c in hdr.columns]
    miss = [c for c in need if c not in cols]
    print(f"{'OK ' if not miss else 'BAD'} {f.name:70s} {len(cols)} columns", f"missing: {miss}" if miss else "")
    ok &= not miss
print("\nExpected: 8 files (Mon, Tue, Wed, Thu-Morning, Thu-Afternoon, Fri-Morning, Fri-Afternoon-PortScan, Fri-Afternoon-DDoS).")
sys.exit(0 if ok else 1)
