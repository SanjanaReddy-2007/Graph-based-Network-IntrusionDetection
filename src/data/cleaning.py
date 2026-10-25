"""Cleaning of CICIDS2017 flows. Every decision is counted in a report dict that is saved to disk.

Cleaning decisions (also documented in README):
 1. Header whitespace stripped; identifier columns canonicalised.
 2. Rows with missing Label / IP / Timestamp dropped.
 3. Non-identifier columns coerced to numeric; +/-inf -> NaN; rows with any NaN feature dropped
    (a few thousand rows, mostly Flow Bytes/s division-by-zero artefacts).
 4. Exact duplicate rows dropped (CICFlowMeter emits duplicates).
 5. Timestamps parsed; the AM/PM ambiguity is repaired (see ``parse_timestamps``).
 6. Labels normalised (the en-dash in 'Web Attack – XSS' is mis-decoded by many tools).
 7. Duplicate feature columns ('Fwd Header Length.1') and globally constant columns dropped.
 8. Negative values are NOT clipped: several columns legitimately use -1 sentinels
    (Init_Win_bytes_*), others contain a few negative-duration bugs. All features go through a
    signed log1p transform which is well defined for negatives and tames heavy tails.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..utils import get_logger

log = get_logger(__name__)

ID_COLS = ["Flow ID", "Source IP", "Source Port", "Destination IP", "Destination Port", "Timestamp"]
NON_FEATURE = ID_COLS + ["Label"]


def normalize_label(s: pd.Series) -> pd.Series:
    """'Web Attack \x96 XSS' / 'Web Attack – XSS' / 'Web Attack � XSS' -> 'Web Attack - XSS'."""
    s = s.astype(str).str.replace(r"[^\x20-\x7E]+", "-", regex=True)  # mis-decoded en-dash -> '-'
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    # canonical spelling of benign
    s = s.where(s.str.upper() != "BENIGN", "BENIGN")
    return s


_TS_FORMATS = ["%m/%d/%Y %H:%M", "%m/%d/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%d/%m/%Y %H:%M:%S"]


def parse_timestamps(s: pd.Series, fix_am_pm: bool = True, hour_threshold: int = 8) -> pd.Series:
    """Parse CICIDS2017 timestamps and repair the AM/PM ambiguity.

    Different copies of the dataset use different layouts (``7/7/2017 8:59`` month-first, minute resolution, 12-hour
    clock *without* AM/PM; or ``03/07/2017 08:55:58`` day-first with seconds). The layout is auto-detected: the one
    that parses the most rows and puts them in July (the capture week was 3-7 July 2017) wins.

    AM/PM: the captures ran roughly 08:00-17:00 but the 12-hour clock has no AM/PM marker, so 1:30 can mean 13:30.
    Common repair: hours < ``hour_threshold`` are afternoon (add 12 h). Hours 8..12 are unchanged.
    """
    s = s.astype(str).str.strip()
    sample = s.iloc[:20000]
    scored = []
    for fmt in _TS_FORMATS:
        t = pd.to_datetime(sample, format=fmt, errors="coerce")
        ok = t.notna()
        scored.append((float(ok.mean()), float((t[ok].dt.month == 7).mean()) if ok.any() else 0.0, fmt))
    scored.sort(reverse=True)
    ts = pd.to_datetime(s, format=scored[0][2], errors="coerce")
    for _, _, fmt in scored[1:]:  # fill rows that use another layout
        miss = ts.isna()
        if not miss.any():
            break
        ts = ts.where(~miss, pd.to_datetime(s[miss], format=fmt, errors="coerce").reindex(ts.index))
    if fix_am_pm:
        pm = ts.dt.hour < hour_threshold
        ts = ts.where(~pm, ts + pd.Timedelta(hours=12))
    return ts


def clean_file(df: pd.DataFrame, cfg: dict, report: Dict[str, int]) -> pd.DataFrame:
    """Clean one raw CSV frame. Returns a frame with identifier columns, float32 features, Label, ts."""
    dcfg = cfg["data"]
    n0 = len(df)
    report["rows_raw"] = report.get("rows_raw", 0) + n0

    df = df.dropna(subset=["Label", "Source IP", "Destination IP", "Timestamp"])
    report["dropped_missing_id_or_label"] = report.get("dropped_missing_id_or_label", 0) + (n0 - len(df))

    feat_cols = [c for c in df.columns if c not in NON_FEATURE]
    feats = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    feats = feats.replace([np.inf, -np.inf], np.nan)
    # columns that are (almost) entirely non-numeric are not features
    bad_cols = [c for c in feat_cols if feats[c].isna().mean() > 0.5]
    if bad_cols:
        log.warning("Dropping mostly-non-numeric columns: %s", bad_cols)
        feats = feats.drop(columns=bad_cols)
        report["dropped_non_numeric_cols"] = sorted(set(report.get("dropped_non_numeric_cols", []) + bad_cols))
    ok = ~feats.isna().any(axis=1)
    report["dropped_nan_or_inf_rows"] = report.get("dropped_nan_or_inf_rows", 0) + int((~ok).sum())
    df = pd.concat([df[ID_COLS + ["Label"]], feats.astype(np.float32)], axis=1)[ok.values]

    if dcfg.get("drop_duplicates", True):
        n1 = len(df)
        df = df.drop_duplicates()
        report["dropped_duplicate_rows"] = report.get("dropped_duplicate_rows", 0) + (n1 - len(df))

    df = df.copy()
    df["ts"] = parse_timestamps(df["Timestamp"], dcfg.get("fix_am_pm", True), dcfg.get("am_pm_hour_threshold", 8))
    bad_ts = df["ts"].isna()
    report["dropped_bad_timestamp"] = report.get("dropped_bad_timestamp", 0) + int(bad_ts.sum())
    df = df[~bad_ts]
    df["Label"] = normalize_label(df["Label"])
    df["Source Port"] = pd.to_numeric(df["Source Port"], errors="coerce")
    df["Destination Port"] = pd.to_numeric(df["Destination Port"], errors="coerce")
    return df


def drop_redundant_columns(df: pd.DataFrame, feat_cols: List[str], report: dict) -> List[str]:
    """Drop '.1' duplicate columns and globally constant columns (label-free decision)."""
    keep = []
    dup = []
    for c in feat_cols:
        if c.endswith(".1") and c[:-2] in df.columns and np.array_equal(df[c].values, df[c[:-2]].values):
            dup.append(c)
        else:
            keep.append(c)
    const = [c for c in keep if df[c].nunique(dropna=False) <= 1]
    keep = [c for c in keep if c not in const]
    report["dropped_duplicate_columns"] = dup
    report["dropped_constant_columns"] = const
    return keep


def label_groups(label: pd.Series, rare: List[str]) -> pd.Series:
    """Group rare attack classes (Heartbleed, Infiltration, SQL injection) into 'Rare'."""
    rare_l = {r.lower() for r in rare}
    return label.where(~label.str.lower().isin(rare_l), "Rare")
