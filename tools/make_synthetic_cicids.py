#!/usr/bin/env python
"""Generate a SMALL SYNTHETIC dataset in the exact file format of CICIDS2017 'GeneratedLabelledFlows'
(leading-space headers, 12-hour clock without AM/PM, cp1252 en-dash in web-attack labels, inf/NaN rows,
duplicates, a constant column and a '.1' duplicate column).

PURPOSE: end-to-end smoke tests of the code only. Nothing produced from this data is a research result.
Usage: python tools/make_synthetic_cicids.py [--out data/synthetic_raw]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["Flow Duration", "Total Fwd Packets", "Total Backward Packets", "Total Length of Fwd Packets",
            "Total Length of Bwd Packets", "Fwd Packet Length Max", "Fwd Packet Length Mean", "Bwd Packet Length Max",
            "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Fwd IAT Total", "SYN Flag Count",
            "ACK Flag Count", "Average Packet Size", "Init_Win_bytes_forward", "Init_Win_bytes_backward",
            "Idle Mean", "Active Mean"]
DAYS = [(3, "Monday"), (4, "Tuesday"), (5, "Wednesday"), (6, "Thursday"), (7, "Friday")]
ATTACKS = {  # day -> [(label, n_flows, attacker_ip or None for internal)]
    "Monday": [],
    "Tuesday": [("FTP-Patator", 260, "172.16.0.1"), ("SSH-Patator", 220, "172.16.0.1")],
    "Wednesday": [("DoS Hulk", 380, "172.16.0.1"), ("DoS slowloris", 150, "172.16.0.1"), ("Heartbleed", 12, "172.16.0.1")],
    "Thursday": [("Web Attack \u2013 Brute Force", 200, "172.16.0.1"), ("Infiltration", 20, None)],
    "Friday": [("PortScan", 330, "172.16.0.1"), ("DDoS", 350, "172.16.0.1"), ("Bot", 120, None)],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic_raw")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.csv"):
        f.unlink()
    d = len(FEATURES)
    internal = [f"192.168.10.{i}" for i in range(5, 45)]
    servers = ["192.168.10.50", "192.168.10.3", "8.8.8.8", "205.174.165.73"]
    sig = {}
    def signature(label):
        if label not in sig:
            v = np.zeros(d)
            v[rng.choice(d, 6, replace=False)] = rng.choice([-1.0, 1.0], 6) * rng.uniform(0.6, 1.1, 6)
            sig[label] = v
        return sig[label]
    base_loc = rng.uniform(1.5, 5.0, d)

    def feats(n, label):
        z = rng.normal(0, 1.0, (n, d)) + (signature(label) if label != "BENIGN" else 0)
        x = np.exp(base_loc + 0.9 * z)
        x[:, FEATURES.index("Init_Win_bytes_backward")] = np.where(rng.random(n) < 0.2, -1, x[:, 17])
        return x

    def stamp(day_num, secs):
        h, m = int(secs // 3600), int((secs % 3600) // 60)
        return f"7/{day_num}/2017 {(h % 12) or 12}:{m:02d}"  # 12-hour clock, no AM/PM (known CICIDS issue)

    rows_by_day = {}
    for day_num, day in DAYS:
        recs = []
        n_ben = 1600
        t = rng.uniform(8 * 3600, 17 * 3600, n_ben)
        src = rng.choice(internal, n_ben, p=rng.dirichlet(np.ones(len(internal)) * 2))
        # each internal host talks mostly to 1-2 preferred destinations -> benign relational structure
        pref = {h: rng.choice(servers, 2, replace=False) for h in internal}
        dst = np.array([rng.choice(pref[h]) if rng.random() < 0.8 else rng.choice(servers) for h in src])
        X = feats(n_ben, "BENIGN")
        for i in range(n_ben):
            recs.append((src[i], dst[i], t[i], "BENIGN", X[i]))
        for label, n, atk in ATTACKS[day]:
            n_burst = max(1, n // 80)
            starts = rng.uniform(8.5 * 3600, 16 * 3600, n_burst)
            X = feats(n, label)
            for i in range(n):
                b = rng.integers(0, n_burst)
                s_ip = atk if atk else rng.choice(internal[:6])
                d_ip = "192.168.10.50" if label != "Bot" else "205.174.165.73"
                if label == "PortScan":
                    d_ip = "192.168.10.50"
                recs.append((s_ip, d_ip, starts[b] + rng.uniform(0, 900), label, X[i]))
        recs.sort(key=lambda r: r[2])
        rows_by_day[day] = (day_num, recs)

    def write(day, recs, day_num, path):
        data = []
        for src, dst, t, label, x in recs:
            sp, dp, proto = int(rng.integers(1024, 65535)), int(rng.choice([22, 21, 80, 443, 53])), 6
            data.append([f"{dst}-{src}-{dp}-{sp}-{proto}", src, sp, dst, dp, proto, stamp(day_num, t)] + list(x)
                        + [0.0, label])
        cols = ["Flow ID", "Source IP", "Source Port", "Destination IP", "Destination Port", "Protocol", "Timestamp"] \
            + FEATURES + ["Bwd PSH Flags", "Label"]
        df = pd.DataFrame(data, columns=cols)
        df.insert(len(FEATURES) + 7, "Fwd Header Length", df["Total Fwd Packets"] * 20)
        df.insert(len(FEATURES) + 8, "Fwd Header Length.1", df["Fwd Header Length"])
        df.loc[df.sample(frac=0.005, random_state=1).index, "Flow Bytes/s"] = np.inf
        df.loc[df.sample(frac=0.003, random_state=2).index, "Flow Packets/s"] = np.nan
        df = pd.concat([df, df.sample(frac=0.004, random_state=3)])  # exact duplicates
        df.columns = [df.columns[0]] + [" " + c for c in df.columns[1:]]  # leading spaces like the real files
        df.to_csv(path, index=False, encoding="cp1252")

    for day, (day_num, recs) in rows_by_day.items():
        if day == "Thursday":  # two files, like the real dataset
            half = len(recs) // 2
            write(day, recs[:half], day_num, out / "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")
            write(day, recs[half:], day_num, out / "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv")
        else:
            write(day, recs, day_num, out / f"{day}-WorkingHours.pcap_ISCX.csv")
    print(f"Wrote synthetic CICIDS-format CSVs to {out} (SMOKE TEST DATA ONLY)")


if __name__ == "__main__":
    main()
