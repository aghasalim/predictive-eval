"""v2 plus a difficulty for each question.

Each subject-benchmark pair splits its items into acquisition and evaluation
pools on its own, so an item being predicted for one subject has often been
revealed for another. b_i, the item's own offset, is estimated from every
revealed label on that exact item, on top of everything else in the model, and
shrunk towards 0.

h(item)   a ridge regression from cheap text features (switched off by
          default, h_scale 0, because it did not transfer across benchmarks)
g_b(tag)  an online offset for items sharing the same item_features tag
b_i       an online offset for the item itself, keyed by its content

P(correct) = sigmoid(a_b + w * theta_s + d_sb + h(item) + g_b(tag) + b_i)
"""
import json
import math
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
PARAMS = {"mu0": -0.3, "sd_a": 0.7, "sd_d": 1.0, "w": 0.5, "sd_theta": 1.0,
          "sd_g": 1.0, "h_scale": 0.0, "ridge": 10.0, "sd_i": 1.0}
THETA = {}
COEF = []
_cache = {}
_feat_cache = {}


def sig(x):
    return 1 / (1 + math.exp(-max(-30, min(30, x))))


def logit(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def features(text):
    """Cheap, benchmark-agnostic descriptors of an item's text."""
    if text in _feat_cache:
        return _feat_cache[text]
    n = max(len(text), 1)
    f = [
        1.0,
        math.log1p(len(text)) / 10,
        math.log1p(text.count("\n")) / 5,
        sum(c.isdigit() for c in text) / n * 10,
        sum(c in "$\\^_{}=" for c in text) / n * 10,
        math.log1p(text.count("```") + text.count("def ") + text.count(";\n")) / 3,
        math.log1p(len(re.findall(r"https?://", text))) / 2,
        1.0 if re.search(r"\b(error|exception|traceback|bug|fails?)\b", text, re.I) else 0.0,
    ]
    if len(_feat_cache) > 50000:
        _feat_cache.clear()
    _feat_cache[text] = f
    return f


def h(item):
    if not COEF:
        return 0.0
    return PARAMS["h_scale"] * sum(c * x for c, x in zip(COEF, features(item.get("item_content", ""))))


def fit(train_benches, data_dir=HERE.parent / "mdb"):
    import numpy as np
    import pandas as pd
    per_name = defaultdict(list)
    bench_means, X, Y = [], [], []
    for b in train_benches:
        subj = pd.read_parquet(data_dir / b / "subjects.parquet").set_index("subject_id")
        items = pd.read_parquet(data_dir / b / "items.parquet").set_index("item_id")
        resp = pd.read_parquet(data_dir / b / "response.parquet")
        resp = resp[resp["response"].isin([0.0, 1.0])]
        acc = resp.groupby("subject_id")["response"].agg(["sum", "count"])
        m = acc["sum"].sum() / acc["count"].sum()
        centre = logit(m)
        bench_means.append(centre)
        for sid, row in acc.iterrows():
            rate = (row["sum"] + 2 * m) / (row["count"] + 2)
            name = subj.loc[sid, "normalized_name"] if sid in subj.index else None
            if isinstance(name, str) and name:
                per_name[name].append(logit(rate) - centre)
        # Item difficulty relative to the benchmark, smoothed, capped at 2,000
        # items per benchmark so one large benchmark does not dominate.
        ia = resp.groupby("item_id")["response"].agg(["sum", "count"])
        ia = ia.sample(min(len(ia), 2000), random_state=0)
        for iid, row in ia.iterrows():
            if iid not in items.index:
                continue
            rate = (row["sum"] + 2 * m) / (row["count"] + 2)
            X.append(features(str(items.loc[iid, "content"])))
            Y.append(logit(rate) - centre)
    THETA.clear()
    k = 1 / PARAMS["sd_theta"] ** 2
    for name, xs in per_name.items():
        THETA[name] = sum(xs) / (len(xs) + k)
    if bench_means:
        PARAMS["mu0"] = sum(bench_means) / len(bench_means)
    X, Y = np.array(X), np.array(Y)
    reg = PARAMS["ridge"] * np.eye(X.shape[1])
    reg[0, 0] = 0
    COEF[:] = list(np.linalg.solve(X.T @ X + reg, X.T @ Y))
    _cache.clear()


def save(path=HERE / "irt3_params.json"):
    path.write_text(json.dumps({"params": PARAMS, "theta": THETA, "coef": COEF}, indent=0))


def load(path=HERE / "irt3_params.json"):
    if path.exists():
        d = json.loads(path.read_text())
        PARAMS.update(d["params"])
        THETA.update(d["theta"])
        COEF[:] = d["coef"]


def map_offset(obs, sd):
    """MAP of a shared offset x given (y, fixed logit) pairs and a N(0, sd) prior."""
    x = 0.0
    for _ in range(8):
        g, hh = -x / sd ** 2, -1 / sd ** 2
        for y, f in obs:
            p = sig(f + x)
            g += y - p
            hh -= p * (1 - p)
        x -= g / hh
    return x


def skey(subject):
    return tuple(sorted(subject.items()))


def ikey(item):
    """The same question asked of different subjects has the same content."""
    return hash((item.get("benchmark_id", ""), item.get("item_content", "")))


def theta_of(subject):
    return PARAMS["w"] * THETA.get(subject.get("normalized_name", ""), 0.0)


def estimates(labeled):
    labeled = labeled or []
    key = (id(labeled), len(labeled), repr(labeled[:1]), repr(labeled[-1:]))
    if key in _cache:
        return _cache[key]
    by_bench = defaultdict(list)
    for (subject, item), y in labeled:
        by_bench[item.get("benchmark_id", "")].append((subject, item, int(y)))
    out = {}
    for b, rows in by_bench.items():
        base = [(y, PARAMS["mu0"] + theta_of(s) + h(it)) for s, it, y in rows]
        a = PARAMS["mu0"] + map_offset(base, PARAMS["sd_a"])
        tags = defaultdict(list)
        for s, it, y in rows:
            tags[it.get("item_features", "")].append((y, a + theta_of(s) + h(it)))
        g = {t: map_offset(obs, PARAMS["sd_g"]) for t, obs in tags.items() if t}
        per_subject = defaultdict(list)
        for s, it, y in rows:
            t = it.get("item_features", "")
            per_subject[skey(s)].append((y, a + theta_of(s) + h(it) + g.get(t, 0.0)))
        d = {sk: map_offset(obs, PARAMS["sd_d"]) for sk, obs in per_subject.items()}
        per_item = defaultdict(list)
        for s, it, y in rows:
            t = it.get("item_features", "")
            per_item[ikey(it)].append((y, a + theta_of(s) + h(it) + g.get(t, 0.0) + d[skey(s)]))
        bi = {k: map_offset(obs, PARAMS["sd_i"]) for k, obs in per_item.items()}
        out[b] = (a, d, g, bi)
    if len(_cache) > 64:
        _cache.clear()
    _cache[key] = out
    return out


def predict(input, labeled=None):
    subject, item = input
    base = theta_of(subject) + h(item)
    est = estimates(labeled).get(item.get("benchmark_id", ""))
    if est is None:
        return sig(PARAMS["mu0"] + base)
    a, d, g, bi = est
    return sig(a + base + d.get(skey(subject), 0.0) + g.get(item.get("item_features", ""), 0.0)
               + bi.get(ikey(item), 0.0))


load()
