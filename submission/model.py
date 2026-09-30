"""Shrunk logistic model: P(correct) = sigmoid(a_b + w * theta_s + d_sb).

theta_s  ability of the model by name, learned from the training benchmarks
         (0 for a model never seen in training)
a_b      difficulty of the held-out benchmark, estimated at test time from
         every revealed label on it, with a prior at the training average
d_sb     this subject's own offset on this benchmark, shrunk towards 0 until
         it has enough labels of its own

Both a_b and d_sb are MAP estimates under Gaussian priors, found with a few
Newton steps, so one early label moves the prediction a little, not to 0 or 1.
"""
import json
import math
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
PARAMS = {"mu0": -0.3, "sd_a": 0.7, "sd_d": 1.0, "w": 0.5, "sd_theta": 1.0}
THETA = {}
_cache = {}


def sig(x):
    return 1 / (1 + math.exp(-max(-30, min(30, x))))


def logit(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def fit(train_benches, data_dir=HERE.parent / "mdb"):
    """Learn model abilities and the prior on benchmark difficulty."""
    import pandas as pd
    per_name = defaultdict(list)
    bench_means = []
    for b in train_benches:
        subj = pd.read_parquet(data_dir / b / "subjects.parquet").set_index("subject_id")
        resp = pd.read_parquet(data_dir / b / "response.parquet")
        resp = resp[resp["response"].isin([0.0, 1.0])]
        acc = resp.groupby("subject_id")["response"].agg(["sum", "count"])
        # Smoothed per-subject rate so a model with few items is not extreme.
        m = acc["sum"].sum() / acc["count"].sum()
        centre = logit(m)
        bench_means.append(centre)
        for sid, row in acc.iterrows():
            rate = (row["sum"] + 2 * m) / (row["count"] + 2)
            name = subj.loc[sid, "normalized_name"] if sid in subj.index else None
            if isinstance(name, str) and name:
                per_name[name].append(logit(rate) - centre)
    THETA.clear()
    k = 1 / PARAMS["sd_theta"] ** 2
    for name, xs in per_name.items():
        THETA[name] = sum(xs) / (len(xs) + k)
    if bench_means:
        PARAMS["mu0"] = sum(bench_means) / len(bench_means)
    _cache.clear()


def save(path=HERE / "irt_params.json"):
    path.write_text(json.dumps({"params": PARAMS, "theta": THETA}, indent=0))


def load(path=HERE / "irt_params.json"):
    if path.exists():
        d = json.loads(path.read_text())
        PARAMS.update(d["params"])
        THETA.update(d["theta"])


def map_offset(obs, base, sd, start=0.0):
    """MAP of a shared offset x given (y, fixed logit) pairs and a N(0, sd) prior."""
    x = start
    for _ in range(8):
        g = -x / sd ** 2
        h = -1 / sd ** 2
        for y, f in obs:
            p = sig(f + base + x)
            g += y - p
            h -= p * (1 - p)
        x -= g / h
    return x


def skey(subject):
    return tuple(sorted(subject.items()))


def estimates(labeled):
    labeled = labeled or []
    # The list's id alone can be reused by a new list, so the key also carries
    # its length and its first and last entries.
    key = (id(labeled), len(labeled), repr(labeled[:1]), repr(labeled[-1:]))
    if key in _cache:
        return _cache[key]
    by_bench = defaultdict(list)
    for (subject, item), y in labeled or []:
        by_bench[item.get("benchmark_id", "")].append((subject, int(y)))
    out = {}
    for b, rows in by_bench.items():
        th = [(y, PARAMS["w"] * THETA.get(s.get("normalized_name", ""), 0.0)) for s, y in rows]
        a = PARAMS["mu0"] + map_offset(th, PARAMS["mu0"], PARAMS["sd_a"])
        per_subject = defaultdict(list)
        for s, y in rows:
            per_subject[skey(s)].append(y)
        offsets = {}
        for sk, ys in per_subject.items():
            t = PARAMS["w"] * THETA.get(dict(sk).get("normalized_name", ""), 0.0)
            offsets[sk] = map_offset([(y, t) for y in ys], a, PARAMS["sd_d"])
        out[b] = (a, offsets)
    if len(_cache) > 64:
        _cache.clear()
    _cache[key] = out
    return out


def predict(input, labeled=None):
    subject, item = input
    theta = PARAMS["w"] * THETA.get(subject.get("normalized_name", ""), 0.0)
    est = estimates(labeled).get(item.get("benchmark_id", ""))
    if est is None:
        return sig(PARAMS["mu0"] + theta)
    a, offsets = est
    return sig(a + theta + offsets.get(skey(subject), 0.0))


load()
