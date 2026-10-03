"""v3 plus an LLM's guess at how hard each question is.

For every item, a small OpenAI model is asked once how likely a typical AI model
is to answer it correctly (cached by content). L(item) is the logit of that guess.
The item term becomes

h(item) = lam * (L(item) - Lbar_b) + kappa * (Lbar_b - Lbar_train)

where Lbar_b is the running mean of L over items seen on the item's benchmark and
Lbar_train its mean on the training benchmarks. kappa moves the benchmark's level
before any label is revealed; lam ranks items within a benchmark. With no API key
or a failed call an item gets L = Lbar_b, so the model falls back to v3.

The rest is v3, a difficulty for each question:

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
          "sd_g": 1.0, "h_scale": 0.0, "ridge": 10.0, "sd_i": 2.0,
          "lam": 0.0, "kappa": 0.0, "L_train": 0.0}
THETA = {}
COEF = []
_cache = {}
_feat_cache = {}
_llm = {}          # item key -> LLM probability (None if the call failed)
_bench_L = {}      # benchmark id -> {item key: L}
CONFIG = {}        # api_key, model; read from private_config.json next to this file


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
    if PARAMS["lam"] == 0 and PARAMS["kappa"] == 0:
        return 0.0
    b = item.get("benchmark_id", "")
    L = llm_logit(item)
    seen = _bench_L.setdefault(b, {})
    k = ikey(item)
    if L is not None and k not in seen:
        seen[k] = L
    mean_b = sum(seen.values()) / len(seen) if seen else PARAMS["L_train"]
    own = L if L is not None else mean_b
    return PARAMS["lam"] * (own - mean_b) + PARAMS["kappa"] * (mean_b - PARAMS["L_train"])


PROMPT = ("You estimate item difficulty for AI benchmarks. Below is one benchmark item. "
          "Estimate the probability that a typical current large language model answers it "
          "correctly. Reply with only a number between 0 and 1.\n\nItem:\n{content}\n\nFeatures: {features}")


def llm_logit(item):
    """Logit of the LLM's success estimate for this item, or None without a key or on failure."""
    import hashlib
    content = item.get("item_content", "")
    key = hashlib.md5((content[:4000] + "|" + item.get("item_features", "")).encode()).hexdigest()
    if key in _llm:
        p = _llm[key]
        return None if p is None else logit(p)
    api_key = CONFIG.get("api_key")
    if not api_key:
        return None
    import urllib.request
    body = {"model": CONFIG.get("model", "gpt-5.6-luna"), "max_completion_tokens": CONFIG.get("max_tokens", 2000),
            "messages": [{"role": "user", "content": PROMPT.format(content=content[:4000], features=item.get("item_features", ""))}]}
    p = None
    for _ in range(3):
        try:
            req = urllib.request.Request("https://api.openai.com/v1/chat/completions", json.dumps(body).encode(),
                                         {"Authorization": "Bearer " + api_key, "Content-Type": "application/json"})
            text = json.load(urllib.request.urlopen(req, timeout=120))["choices"][0]["message"]["content"]
            m = re.search(r"\d*\.?\d+", text or "")
            if m:
                p = min(max(float(m.group()), 0.01), 0.99)
            break
        except Exception:
            continue
    _llm[key] = p
    return None if p is None else logit(p)


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


def save(path=HERE / "llm_irt_params.json"):
    path.write_text(json.dumps({"params": PARAMS, "theta": THETA, "coef": COEF}, indent=0))


def load(path=HERE / "llm_irt_params.json"):
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
_cfg = HERE / "private_config.json"
if _cfg.exists():
    CONFIG.update(json.loads(_cfg.read_text()))
_cache_file = HERE / "llm_cache.json"
if _cache_file.exists():
    _llm.update(json.loads(_cache_file.read_text()))
