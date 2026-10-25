"""End-to-end preparation (raw CSV -> cached arrays) and the in-memory ``Dataset`` container."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ..utils import get_logger, resolve_path, save_json, stable_hash
from .cleaning import ID_COLS, NON_FEATURE, clean_file, drop_redundant_columns, label_groups
from .features import signed_log1p
from .loading import list_csvs, read_one
from .subsample import subsample_frame

log = get_logger(__name__)

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def prepare(cfg: Dict[str, Any]) -> None:
    """Read all raw CSVs, clean, subsample, sort chronologically and cache to ``processed_dir``."""
    raw_dir = resolve_path(cfg, "raw_dir")
    out_dir = resolve_path(cfg, "processed_dir")
    dcfg, scfg = cfg["data"], cfg["subsample"]
    report: Dict[str, Any] = {"files": []}
    frames = []
    for f in list_csvs(raw_dir):
        log.info("Reading %s", f.name)
        df = read_one(f, dcfg.get("encodings", ["cp1252", "utf-8", "latin-1"]))
        n_raw = len(df)
        df = clean_file(df, cfg, report)
        df["day"] = df["ts"].dt.day_name()
        if scfg.get("enabled", True):  # per-file pre-stage cap keeps memory low on the 2.8M-row full set
            df = subsample_frame(df, scfg.get("benign_cap_per_day"), scfg.get("attack_cap_per_label_day"), scfg["seed"])
        report["files"].append({"file": f.name, "rows_raw": n_raw, "rows_after_clean_and_precap": len(df)})
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    del frames
    if dcfg.get("drop_duplicates", True):
        n = len(df)
        df = df.drop_duplicates(subset=[c for c in df.columns if c not in ("ts", "day")])
        report["dropped_duplicate_rows_across_files"] = n - len(df)
    if scfg.get("enabled", True):
        df = subsample_frame(df, scfg.get("benign_cap_per_day"), scfg.get("attack_cap_per_label_day"), scfg["seed"])
    df = df[df["day"].isin(DAY_ORDER)]
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)

    feat_cols = [c for c in df.columns if c not in NON_FEATURE + ["ts", "day"]]
    feat_cols = drop_redundant_columns(df, feat_cols, report)

    n = len(df)
    codes, uniques = pd.factorize(np.concatenate([df["Source IP"].values, df["Destination IP"].values]))
    flow_codes, _ = pd.factorize(df["Flow ID"])
    meta = pd.DataFrame(
        {
            "day": df["day"].values,
            "t": (df["ts"] - df["ts"].min()).dt.total_seconds().values,
            "src_id": codes[:n].astype(np.int32),
            "dst_id": codes[n:].astype(np.int32),
            "src_port": df["Source Port"].values,
            "dst_port": df["Destination Port"].values,
            "flow_key": flow_codes.astype(np.int64),
            "label": df["Label"].values,
        }
    )
    meta["label_group"] = label_groups(meta["label"], dcfg.get("rare_labels", []))
    meta["y"] = (meta["label"] != "BENIGN").astype(np.int8)
    X = df[feat_cols].to_numpy(dtype=np.float32)

    meta.to_pickle(out_dir / "meta.pkl")
    np.save(out_dir / "X_raw.npy", X)
    save_json(feat_cols, out_dir / "features.json")
    save_json([str(u) for u in uniques], out_dir / "ips.json")
    report["n_rows_final"] = n
    report["n_features"] = len(feat_cols)
    report["n_ips"] = len(uniques)
    report["rows_per_day_label"] = {f"{d}|{l}": int(c) for (d, l), c in meta.groupby(["day", "label"]).size().items()}
    save_json(report, out_dir / "cleaning_report.json")
    log.info("Prepared %d flows, %d features, %d unique IPs. Report: %s", n, len(feat_cols), len(uniques),
             out_dir / "cleaning_report.json")
    log.info("Attack fraction: %.3f", meta["y"].mean())


@dataclass
class Dataset:
    """In-memory processed dataset. Row i == node i, rows are sorted chronologically."""

    meta: pd.DataFrame
    X_log: np.ndarray  # signed-log1p features (NOT yet standardised: scaling is per-protocol, train-only)
    feature_cols: List[str]
    ip_table: List[str]
    sig: str

    @property
    def n(self) -> int:
        return len(self.meta)

    @property
    def y(self) -> np.ndarray:
        return self.meta["y"].to_numpy(dtype=np.int64)

    @property
    def t(self) -> np.ndarray:
        return self.meta["t"].to_numpy(dtype=np.float64)

    @property
    def day(self) -> np.ndarray:
        return self.meta["day"].to_numpy()

    @property
    def src_id(self) -> np.ndarray:
        return self.meta["src_id"].to_numpy(dtype=np.int64)

    @property
    def dst_id(self) -> np.ndarray:
        return self.meta["dst_id"].to_numpy(dtype=np.int64)

    @property
    def flow_key(self) -> np.ndarray:
        return self.meta["flow_key"].to_numpy(dtype=np.int64)

    @property
    def attack_label(self) -> np.ndarray:
        return self.meta["label"].to_numpy()


def load_dataset(cfg: Dict[str, Any]) -> Dataset:
    d = resolve_path(cfg, "processed_dir")
    if not (d / "meta.pkl").exists():
        raise FileNotFoundError("Processed data not found. Run:  python run.py --experiment prepare")
    meta = pd.read_pickle(d / "meta.pkl")
    X = np.load(d / "X_raw.npy")
    with open(d / "features.json") as f:
        feats = json.load(f)
    with open(d / "ips.json") as f:
        ips = json.load(f)
    assert (np.diff(meta["t"].to_numpy()) >= 0).all(), "meta must be sorted chronologically"
    sig = stable_hash(meta["t"].to_numpy()[:2000], len(meta), feats)
    return Dataset(meta=meta, X_log=signed_log1p(X), feature_cols=feats, ip_table=ips, sig=sig)
