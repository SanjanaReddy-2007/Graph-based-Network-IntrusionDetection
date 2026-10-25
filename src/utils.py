"""Shared utilities: config loading (with inheritance + CLI overrides), seeding, logging, hashing."""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``upd`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path, overrides: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Load a YAML config. Supports ``inherit: other.yaml`` and ``a.b.c=value`` overrides."""
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = ROOT / path
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if "inherit" in cfg:
        base_name = cfg.pop("inherit")
        base = load_config(path.parent / base_name)
        cfg = deep_update(base, cfg)
    for ov in overrides or []:
        key, _, val = ov.partition("=")
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def resolve_path(cfg: Dict[str, Any], key: str) -> Path:
    """Resolve a path from cfg['paths'][key] relative to the project root; create if missing."""
    p = Path(cfg["paths"][key])
    if not p.is_absolute():
        p = ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int) -> None:
    """Seed python, numpy and (if installed) torch."""
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


_CONFIGURED = False


def get_logger(name: str = "nids") -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
            stream=sys.stdout,
        )
        _CONFIGURED = True
    return logging.getLogger(name)


def stable_hash(*objs: Any) -> str:
    """MD5 over a mix of arrays / JSON-serialisable objects."""
    h = hashlib.md5()
    for o in objs:
        if isinstance(o, np.ndarray):
            h.update(np.ascontiguousarray(o).tobytes())
            h.update(str(o.dtype).encode())
        else:
            h.update(json.dumps(o, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def save_json(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
