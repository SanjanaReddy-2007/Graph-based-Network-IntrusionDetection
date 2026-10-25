"""Configurable down-sampling so the project fits on a laptop / free Colab GPU."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def subsample_frame(
    df: pd.DataFrame,
    benign_cap: Optional[int],
    attack_cap: Optional[int],
    seed: int,
    label_col: str = "Label",
    group_col: str = "day",
) -> pd.DataFrame:
    """Cap BENIGN rows per ``group_col`` and attack rows per (``group_col``, label). ``None`` = no cap."""
    rng = np.random.default_rng(seed)
    keep = np.zeros(len(df), dtype=bool)
    groups = df.groupby([group_col, label_col], sort=False).indices
    for (_, label), idx in groups.items():
        cap = benign_cap if label == "BENIGN" else attack_cap
        if cap is None or len(idx) <= cap:
            keep[idx] = True
        else:
            keep[rng.choice(idx, size=cap, replace=False)] = True
    return df[keep]
