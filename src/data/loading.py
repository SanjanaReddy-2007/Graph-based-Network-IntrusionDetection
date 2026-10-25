"""Reading the CICIDS2017 *GeneratedLabelledFlows* (a.k.a. TrafficLabelling) CSV files."""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import pandas as pd

from ..utils import get_logger

log = get_logger(__name__)

# canonical identifier column names and the (lower-case, stripped) spellings we accept
ID_ALIASES = {
    "flow id": "Flow ID",
    "source ip": "Source IP",
    "src ip": "Source IP",
    "source port": "Source Port",
    "src port": "Source Port",
    "destination ip": "Destination IP",
    "dst ip": "Destination IP",
    "destination port": "Destination Port",
    "dst port": "Destination Port",
    "timestamp": "Timestamp",
    "label": "Label",
}
REQUIRED = ["Flow ID", "Source IP", "Destination IP", "Timestamp", "Label"]


def canonicalise_columns(cols: Sequence[str]) -> List[str]:
    """Strip whitespace (CICIDS headers have leading spaces) and map identifier aliases."""
    out = []
    for c in cols:
        s = str(c).strip()
        out.append(ID_ALIASES.get(s.lower(), s))
    return out


def list_csvs(raw_dir: str | Path) -> List[Path]:
    raw_dir = Path(raw_dir)
    files = sorted(p for p in raw_dir.rglob("*.csv") if not p.name.startswith("."))
    if not files:
        raise FileNotFoundError(
            f"No CSV files found under {raw_dir}. Unzip GeneratedLabelledFlows.zip into that folder "
            "(see README, 'Getting the data')."
        )
    return files


def read_one(path: Path, encodings: Sequence[str] = ("cp1252", "utf-8", "latin-1")) -> pd.DataFrame:
    """Read one CSV trying several encodings (CICIDS2017 files are not valid UTF-8)."""
    last = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False, skipinitialspace=True)
            df.columns = canonicalise_columns(df.columns)
            missing = [c for c in REQUIRED if c not in df.columns]
            if missing:
                raise ValueError(
                    f"{path.name}: missing columns {missing}. This looks like the 'MachineLearningCSV' "
                    "version (no IPs / timestamps). Use GeneratedLabelledFlows.zip instead."
                )
            return df
        except UnicodeDecodeError as e:  # try next encoding
            last = e
    raise last  # type: ignore[misc]
