"""Unit tests (unittest-style, also collected by pytest). Run:  python -m unittest discover -s tests -v"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.cleaning import normalize_label, parse_timestamps  # noqa: E402
from src.data.features import Scaler  # noqa: E402
from src.data.splitting import host_grouped_split, stratified_random_split, temporal_split  # noqa: E402
from src.eval import leakage as lk  # noqa: E402
from src.eval.metrics import compute_metrics  # noqa: E402
from src.eval.stats import correlations, paired_bootstrap_gap, seed_level_tests  # noqa: E402
from src.graph.construction import build_arcs, to_adj  # noqa: E402
from src.graph.disruption import disrupt  # noqa: E402
from src.models.nn_numpy import NumpyNet, weighted_ce  # noqa: E402
from src.models.param_matching import match_mlp_hidden, mlp_param_count, sage_param_count  # noqa: E402
from src.overlap.metrics import compute_overlap  # noqa: E402
from src.overlap.variation import drop_cross_split_edges  # noqa: E402
from src.train.task import GraphData, class_weights  # noqa: E402


class TestOverlap(unittest.TestCase):
    def setUp(self):
        # nodes 0,1 train; 2 val; 3,4 test.  arcs (u->v): in-neighbours of 3: {0,1,4}; of 4: {3}; of 0: {1}
        self.src = np.array([0, 1, 4, 3, 1])
        self.dst = np.array([3, 3, 3, 4, 0])
        self.split = np.array([0, 0, 1, 2, 2])
        self.sid = np.array([10, 11, 12, 10, 13])  # hosts (src)
        self.did = np.array([20, 20, 21, 21, 22])

    def test_known_values(self):
        m, per = compute_overlap(self.src, self.dst, self.split, self.sid, self.did)
        # train hosts {10,11,20}; test hosts {10,13,21,22}; intersection {10}; union {10,11,13,20,21,22}
        self.assertAlmostEqual(m["host_jaccard"], 1 / 6)
        self.assertAlmostEqual(m["host_containment"], 1 / 4)
        # test node 3 has neighbours {0,1,4}: 2/3 train; test node 4 has neighbour {3}: 0 train
        np.testing.assert_allclose(per, [2 / 3, 0.0])
        self.assertAlmostEqual(m["one_hop_mean"], 1 / 3)
        self.assertAlmostEqual(m["test_with_train_nbr"], 0.5)
        # arcs into test nodes: 4 (0->3,1->3,4->3,3->4); sources not test: 2 of 4
        self.assertAlmostEqual(m["test_cross_edge_ratio"], 0.5)
        self.assertAlmostEqual(m["train_nbr_arc_ratio"], 0.5)
        # cross-split arcs overall: (0->3),(1->3) cross; (4->3),(3->4) within; (1->0) within => 2/5
        self.assertAlmostEqual(m["cross_edge_ratio"], 2 / 5)

    def test_no_overlap(self):
        m, _ = compute_overlap(np.array([3]), np.array([4]), self.split, self.sid, self.did)
        self.assertEqual(m["one_hop_mean"], 0.0)
        self.assertEqual(m["test_cross_edge_ratio"], 0.0)

    def test_two_hop_and_homophily(self):
        y = np.array([0, 0, 0, 1, 1])
        m, _ = compute_overlap(self.src, self.dst, self.split, self.sid, self.did, y=y, two_hop_sample=10)
        self.assertGreaterEqual(m["two_hop_train_frac"], 0.0)
        self.assertAlmostEqual(m["edge_homophily"], 3 / 5)  # (0,3)x (1,3)x (4,3)ok (3,4)ok (1,0)ok

    def test_edge_drop_keeps_within(self):
        src = np.array([0, 3, 0, 3]); dst = np.array([3, 0, 1, 4])
        split = np.array([0, 0, 1, 2, 2])
        s, d = drop_cross_split_edges(src, dst, split, 0.0, np.random.default_rng(0))
        for a, b in zip(s, d):
            self.assertEqual(split[a], split[b])
        s, d = drop_cross_split_edges(src, dst, split, 1.0, np.random.default_rng(0))
        self.assertEqual(len(s), 4)


class TestGraph(unittest.TestCase):
    def brute(self, t, src, dst, window, per_type):
        n = len(t)
        E = set()
        for key in (src, dst, src * 1000 + dst):
            for i in range(n):
                cnt = 0
                for j in range(i - 1, -1, -1):
                    if key[j] == key[i]:
                        cnt += 1
                        if cnt > per_type:
                            break
                        if t[i] - t[j] <= window:
                            E.add((i, j)); E.add((j, i))
        return E

    def test_matches_bruteforce(self):
        rng = np.random.default_rng(1)
        n = 120
        t = np.sort(rng.uniform(0, 600, n)); src = rng.integers(0, 6, n); dst = rng.integers(0, 4, n)
        a, b = build_arcs(t, src, dst, window=100, per_type=3, max_degree=10 ** 6)
        self.assertEqual(set(zip(a.tolist(), b.tolist())), self.brute(t, src, dst, 100, 3))

    def test_degree_cap_symmetry_loops(self):
        rng = np.random.default_rng(2)
        n = 400
        t = np.sort(rng.uniform(0, 100, n)); src = rng.integers(0, 3, n); dst = rng.integers(0, 3, n)
        a, b = build_arcs(t, src, dst, window=1e9, per_type=8, max_degree=5)
        self.assertLessEqual(np.bincount(b, minlength=n).max(), 5)
        self.assertFalse((a == b).any())
        A = to_adj(n, a, b)
        rows = np.asarray(A.sum(1)).ravel()
        self.assertTrue(np.allclose(rows[rows > 0], 1.0))

    def test_window_respected(self):
        t = np.array([0., 10., 1000.]); src = np.array([1, 1, 1]); dst = np.array([2, 2, 2])
        a, b = build_arcs(t, src, dst, window=50, per_type=5, max_degree=10)
        self.assertEqual(set(zip(a.tolist(), b.tolist())), {(0, 1), (1, 0)})

    def test_disruption(self):
        rng = np.random.default_rng(0)
        n = 300
        t = np.sort(rng.uniform(0, 1000, n)); a, b = build_arcs(t, rng.integers(0, 8, n), rng.integers(0, 8, n), 500, 4, 15)
        ra, rb = disrupt(a, b, n, "rewire", 0)
        self.assertLess(abs(len(ra) - len(a)) / len(a), 0.1)
        # in/out degrees preserved up to removed loops/duplicates
        self.assertLessEqual(np.abs(np.bincount(rb, minlength=n) - np.bincount(b, minlength=n)).sum() / len(a), 0.1)
        self.assertEqual(len(disrupt(a, b, n, "none", 0)[0]), 0)
        self.assertEqual(len(disrupt(a, b, n, "original", 0)[0]), len(a))
        self.assertLess(len(set(zip(ra.tolist(), rb.tolist())) & set(zip(a.tolist(), b.tolist()))), len(a) * 0.2)
        sa, sb = disrupt(a, b, n, "shuffle", 0)
        self.assertEqual(len(sa), len(a))
        with self.assertRaises(ValueError):
            disrupt(a, b, n, "bogus", 0)


class TestSplitsAndLeakage(unittest.TestCase):
    def test_stratified(self):
        y = (np.random.default_rng(0).random(2000) < 0.2).astype(int)
        s = stratified_random_split(y, {"train": .6, "val": .2, "test": .2}, 0)
        lk.assert_disjoint(s)
        self.assertEqual(sum(len(v) for v in s.values()), 2000)
        for k in s:
            self.assertAlmostEqual(y[s[k]].mean(), y.mean(), delta=0.02)
            self.assertTrue((np.diff(s[k]) > 0).all())

    def test_temporal(self):
        day = np.repeat(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"], 100)
        t = np.arange(500, dtype=float)
        s = temporal_split(day, t, ["Monday", "Tuesday", "Wednesday"], ["Thursday", "Friday"], 0.2)
        lk.assert_temporal_order(t, s)
        self.assertEqual(len(s["test"]), 200)
        bad = {"train": s["test"], "val": s["val"], "test": s["train"]}
        with self.assertRaises(lk.LeakageError):
            lk.assert_temporal_order(t, bad)

    def test_host_grouped(self):
        rng = np.random.default_rng(0)
        n = 6000
        src = rng.integers(0, 150, n); dst = rng.integers(150, 200, n)
        y = (rng.random(n) < .3).astype(int)
        prev = None
        for lvl in (1.0, 0.5, 0.0):
            sp = host_grouped_split(src, dst, y, lvl, 0, heavy_frac=0.02)
            lk.assert_disjoint(sp)
            def hosts(ix): return set(src[ix]) | set(dst[ix])
            cont = len(hosts(sp["train"]) & hosts(sp["test"])) / len(hosts(sp["test"]))
            if prev is not None:
                self.assertLessEqual(cont, prev + 1e-9)
            prev = cont
        self.assertLess(cont, 0.9)

    def test_leak_assertions_fire(self):
        with self.assertRaises(lk.LeakageError):
            lk.assert_disjoint({"train": np.array([1, 2]), "val": np.array([3]), "test": np.array([2])})
        X = np.random.default_rng(0).normal(size=(50, 3)).astype(np.float32)
        sc = Scaler().fit(X, np.arange(30))
        lk.assert_scaler_train_only(sc, np.arange(30))
        with self.assertRaises(lk.LeakageError):
            lk.assert_scaler_train_only(sc, np.arange(40))
        with self.assertRaises(lk.LeakageError):
            lk.assert_no_cross_split_arcs(np.array([0]), np.array([1]), np.array([0, 2]))

    def test_scaler_train_only_stats(self):
        X = np.vstack([np.zeros((10, 1)), 100 * np.ones((10, 1))]).astype(np.float32)
        sc = Scaler(clip=1e9).fit(X, np.arange(10))
        self.assertAlmostEqual(float(sc.mean_[0]), 0.0)


class TestPreprocessing(unittest.TestCase):
    def test_label_and_timestamp(self):
        import pandas as pd
        lab = normalize_label(pd.Series(["BENIGN", "Web Attack \x96 XSS", "Web Attack \u2013 Brute Force", "FTP-Patator", "Web Attack \ufffd Sql Injection"]))
        self.assertEqual(list(lab), ["BENIGN", "Web Attack - XSS", "Web Attack - Brute Force", "FTP-Patator", "Web Attack - Sql Injection"])
        ts = parse_timestamps(pd.Series(["7/7/2017 8:59", "7/7/2017 1:30", "7/7/2017 12:05", "7/7/2017 3:15:20"]))
        self.assertEqual([x.hour for x in ts], [8, 13, 12, 15])
        ts2 = parse_timestamps(pd.Series(["03/07/2017 08:55:58", "07/07/2017 15:01:02"]))  # day-first layout
        self.assertEqual([(x.month, x.day, x.hour) for x in ts2], [(7, 3, 8), (7, 7, 15)])


class TestModels(unittest.TestCase):
    def test_param_matching(self):
        for d in (20, 70, 78):
            for h in (16, 64, 128):
                m, pm, ps = match_mlp_hidden(d, h)
                self.assertLessEqual(abs(pm - ps) / ps, 0.05)
                self.assertEqual(ps, sage_param_count(d, h)); self.assertEqual(pm, mlp_param_count(d, m))

    def test_numpy_param_count_matches_formula(self):
        self.assertEqual(NumpyNet("sage", 20, 16, 2, 0.0, 0).n_params, sage_param_count(20, 16))
        self.assertEqual(NumpyNet("mlp", 20, 13, 2, 0.0, 0).n_params, mlp_param_count(20, 13))

    def test_torch_param_count_if_available(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")
        from src.models.nn_torch import build_net
        for kind, fn, w in (("sage", sage_param_count, 16), ("mlp", mlp_param_count, 13)):
            net = build_net(kind, 20, w, 2, 0.0)
            self.assertEqual(sum(p.numel() for p in net.parameters()), fn(20, w))

    def test_gradients_finite_difference(self):
        rng = np.random.default_rng(0)
        n, d, h = 30, 5, 4
        x = rng.normal(size=(n, d)).astype(np.float64)
        src = rng.integers(0, n, 80); dst = rng.integers(0, n, 80)
        adj = to_adj(n, src[src != dst], dst[src != dst])
        y = rng.integers(0, 2, n)
        G = GraphData(x.astype(np.float32), y, adj.astype(np.float32), np.arange(n))
        idx = np.arange(0, 20); w = class_weights(y[idx])
        for kind in ("sage", "mlp"):
            net = NumpyNet(kind, d, h, 2, 0.0, 1)
            for k in net.P: net.P[k] = net.P[k].astype(np.float64)
            # non-zero biases: with b=0 an all-dead row sits exactly on the ReLU kink, where the subgradient and the
            # finite-difference slope legitimately differ
            for k in ("b1", "b2", "bo"): net.P[k] = rng.normal(0, 0.3, net.P[k].shape)
            G.x = x; G._cache.clear(); G.adj = adj.astype(np.float64)
            Z, cache = net.forward(G); _, dZ = weighted_ce(Z, y, idx, w, True)
            grads = net.backward(G, cache, dZ.astype(np.float64))
            for name in net.P:
                p = net.P[name]; flat = p.reshape(-1)
                for j in rng.choice(flat.size, min(4, flat.size), replace=False):
                    old = flat[j]; eps = 1e-5
                    flat[j] = old + eps; lp, _ = weighted_ce(net.forward(G)[0], y, idx, w, False)
                    flat[j] = old - eps; lm, _ = weighted_ce(net.forward(G)[0], y, idx, w, False)
                    flat[j] = old
                    num = (lp - lm) / (2 * eps); ana = grads[name].reshape(-1)[j]
                    self.assertAlmostEqual(num, ana, delta=5e-4 + 5e-3 * abs(num), msg=f"{kind}:{name}")

    def test_sage_without_edges_equals_mlp_structure(self):
        # with an empty graph the neighbour term vanishes: forward = relu(xWs+b) path
        n, d = 10, 4
        G = GraphData(np.random.default_rng(0).normal(size=(n, d)).astype(np.float32), np.zeros(n, int),
                      to_adj(n, np.zeros(0, int), np.zeros(0, int)), np.arange(n))
        net = NumpyNet("sage", d, 6, 2, 0.0, 0)
        self.assertTrue(np.allclose(G.ax, 0))
        Z, _ = net.forward(G)
        H1 = np.maximum(G.x @ net.P["Ws1"] + net.P["b1"], 0)
        H2 = np.maximum(H1 @ net.P["Ws2"] + net.P["b2"], 0)
        np.testing.assert_allclose(Z, H2 @ net.P["Wo"] + net.P["bo"], rtol=1e-5, atol=1e-6)


class TestEval(unittest.TestCase):
    def test_metrics_vs_sklearn(self):
        from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score
        rng = np.random.default_rng(0)
        y = (rng.random(500) < .2).astype(int); p = np.clip(y * .5 + rng.random(500) * .6, 0, 1)
        m = compute_metrics(y, p); pred = (p >= .5).astype(int)
        self.assertAlmostEqual(m["macro_f1"], f1_score(y, pred, average="macro"))
        self.assertAlmostEqual(m["weighted_f1"], f1_score(y, pred, average="weighted"))
        self.assertAlmostEqual(m["mcc"], matthews_corrcoef(y, pred))
        self.assertAlmostEqual(m["accuracy"], accuracy_score(y, pred))
        self.assertAlmostEqual(m["fpr"], ((pred == 1) & (y == 0)).sum() / (y == 0).sum())

    def test_bootstrap_detects_gap(self):
        rng = np.random.default_rng(0)
        ys, pa, pb = [], [], []
        for _ in range(3):
            y = (rng.random(800) < .3).astype(int)
            ys.append(y); pa.append(np.clip(y * .8 + rng.normal(0, .15, 800), 0, 1)); pb.append(rng.random(800))
        r = paired_bootstrap_gap(ys, pa, pb, B=100)
        self.assertGreater(r["ci_lo"], 0)
        r0 = paired_bootstrap_gap(ys, pa, pa, B=50)
        self.assertEqual(r0["gap"], 0.0)

    def test_seed_tests_and_corr(self):
        r = seed_level_tests([0.1, 0.12, 0.09, 0.11, 0.1])
        self.assertLess(r["t_pvalue"], 0.01)
        self.assertGreaterEqual(r["wilcoxon_pvalue"], 0.0625 - 1e-9)  # 5 seeds -> min attainable p
        c = correlations([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
        self.assertAlmostEqual(c["spearman"], 1.0)


if __name__ == "__main__":
    unittest.main()
