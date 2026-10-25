"""End-to-end smoke test on SYNTHETIC data (numpy backend): prepare -> protocols -> sweep -> ablation -> report."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class TestSmoke(unittest.TestCase):
    def test_pipeline(self):
        import numpy as np
        from src.utils import load_config
        tmp = Path(tempfile.mkdtemp())
        subprocess.check_call([sys.executable, str(ROOT / "tools" / "make_synthetic_cicids.py"), "--out", str(tmp / "raw")],
                              stdout=subprocess.DEVNULL)
        cfg = load_config("configs/smoke.yaml", [f"paths.raw_dir={tmp/'raw'}", f"paths.processed_dir={tmp/'proc'}",
                                                 f"paths.cache_dir={tmp/'cache'}", f"paths.results_dir={tmp/'res'}",
                                                 "seeds=[0,1]", "train.epochs=15", "train.patience=5",
                                                 "overlap.host_levels=[1.0,0.0]", "overlap.edge_levels=[1.0,0.0]",
                                                 "ablation.topologies=[original,none]", "stats.bootstrap_B=20"])
        from src.data.dataset import prepare
        from experiments import ablation, overlap_sweep, protocols, report
        import pandas as pd
        prepare(cfg); protocols.main(cfg); overlap_sweep.main(cfg); ablation.main(cfg); report.main(cfg)
        res = tmp / "res"
        runs = pd.read_csv(res / "protocols_runs.csv")
        self.assertEqual(set(runs.protocol), {"P1", "P2", "P3"})
        self.assertEqual(set(runs.model), {"rf", "mlp", "sage"})
        sp = runs[runs.model != "rf"].groupby(["protocol", "seed"]).n_params.agg(["min", "max"])
        self.assertTrue(((sp["max"] - sp["min"]) / sp["max"] <= 0.05).all())
        ov = pd.read_csv(res / "protocols_overlap.csv")
        self.assertTrue((ov[ov.protocol != "P1"].vis_train_nbr_arc_ratio.fillna(0) == 0).all())  # inductive: nothing leaks
        self.assertGreater(ov[ov.protocol == "P1"].vis_train_nbr_arc_ratio.mean(), 0.3)
        self.assertTrue((res / "report.md").exists())


if __name__ == "__main__":
    unittest.main()
