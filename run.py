#!/usr/bin/env python
"""Single entry point.

  python run.py --experiment prepare                 # clean + cache the CICIDS2017 CSVs
  python run.py --experiment eda
  python run.py --experiment protocols               # RF / MLP / GraphSAGE under P1, P2, P3
  python run.py --experiment overlap_sweep           # controlled overlap variation
  python run.py --experiment ablation                # topology ablation
  python run.py --experiment figures | report | slides
  python run.py --experiment all                     # everything above, in order
  python run.py --experiment smoke                   # synthetic-data end-to-end check (NOT real results)
Options: --config configs/quick.yaml   --set train.epochs=50   --set seeds=[0,1]
"""
from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
import time

from src.utils import ROOT, get_logger, load_config

log = get_logger("run")
ORDER = ["prepare", "eda", "protocols", "overlap_sweep", "ablation", "figures", "report", "slides"]


def run_one(name: str, cfg: dict) -> None:
    t0 = time.time()
    if name == "prepare":
        from src.data.dataset import prepare
        prepare(cfg)
    else:
        importlib.import_module(f"experiments.{name}").main(cfg)
    log.info("[%s] done in %.1f s", name, time.time() - t0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", required=True, choices=ORDER + ["all", "smoke"])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", action="append", default=[], help="override, e.g. --set train.epochs=50")
    args = ap.parse_args()
    if args.experiment == "smoke":
        subprocess.check_call([sys.executable, str(ROOT / "tools" / "make_synthetic_cicids.py")])
        cfg = load_config("configs/smoke.yaml", args.set)
        for name in ["prepare", "eda", "protocols", "overlap_sweep", "ablation", "figures", "report", "slides"]:
            run_one(name, cfg)
        return
    cfg = load_config(args.config, args.set)
    for name in (ORDER if args.experiment == "all" else [args.experiment]):
        run_one(name, cfg)


if __name__ == "__main__":
    main()
