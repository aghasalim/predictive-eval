"""Local copy of the competition's scoring protocol, on the public training data.

One benchmark is held out at a time (benchmark-level cold start). For every
subject-benchmark pair with at least 80 items, items are split 50/50 into an
acquisition pool and an evaluation pool. Labels are revealed in a random order
at budgets 0, 1, 3, 7, 15 and 31 per pair, and the evidence at each budget is
every pair's revealed labels in the sampled round. Score is Brier ALC, averaged
over pairs, lower is better.
"""
import argparse
import importlib.util
import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
DATA = ROOT / "mdb"
BUDGETS = (0, 1, 3, 7, 15, 31)
WEIGHTS = (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)
SUBJECT_KEYS = ("normalized_name", "provider", "release_date", "access_date", "harness",
                "harness_version", "reasoning_effort", "subject_features_extra")
ITEM_KEYS = ("item_content", "item_features", "interactors", "benchmark_id")


def s(v):
    return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)


def load(bench):
    d = DATA / bench
    subj = pd.read_parquet(d / "subjects.parquet").set_index("subject_id")
    items = pd.read_parquet(d / "items.parquet").set_index("item_id")
    resp = pd.read_parquet(d / "response.parquet")
    resp = resp[resp["response"].isin([0.0, 1.0])]
    return subj, items, resp


def binary_benchmarks():
    out = []
    for d in sorted(p.name for p in DATA.iterdir() if (p / "benchmarks.parquet").exists()):
        b = pd.read_parquet(DATA / d / "benchmarks.parquet").iloc[0]
        if b["granularity"] == "item" and b["response_type"] in ("binary", "mixed"):
            out.append(d)
    return out


def rounds(bench, n_rounds, seed, max_pairs=8):
    """Yield (pairs, subjects, items) for sampled formative rounds."""
    subj, items, resp = load(bench)
    anon = f"bench_{abs(hash(bench)) % 10**6}"
    by_pair = defaultdict(list)
    for r in resp.itertuples(index=False):
        by_pair[r.subject_id].append((r.item_id, int(r.response)))
    eligible = [sid for sid, rs in by_pair.items() if len({i for i, _ in rs}) >= 80]
    rng = random.Random(seed)

    def subject_dict(sid):
        row = subj.loc[sid]
        return {k: s(row.get(k)) for k in SUBJECT_KEYS}

    def item_dict(iid):
        row = items.loc[iid]
        return {"item_content": s(row.get("content")), "item_features": s(row.get("item_features")),
                "interactors": "", "benchmark_id": anon}

    for _ in range(n_rounds):
        chosen = rng.sample(eligible, min(max_pairs, len(eligible)))
        pairs = []
        for sid in chosen:
            item_ids = sorted({i for i, _ in by_pair[sid]})
            rng.shuffle(item_ids)
            half = len(item_ids) // 2
            acq, ev = item_ids[:half], item_ids[half:half + 60]
            labels = defaultdict(list)
            for i, y in by_pair[sid]:
                labels[i].append(y)
            acq_order = [(i, rng.choice(labels[i])) for i in acq]
            pairs.append({"sid": sid, "subject": subject_dict(sid), "acq": acq_order,
                          "eval": [(i, labels[i]) for i in ev]})
        yield pairs, subject_dict, item_dict


def score(model, bench, n_rounds=20, seed=0):
    pair_scores = []
    per_budget = defaultdict(list)
    for pairs, _, item_dict in rounds(bench, n_rounds, seed):
        briers = {id(p): [] for p in pairs}
        for bi, n in enumerate(BUDGETS):
            labeled = [[[p["subject"], item_dict(i)], y] for p in pairs for i, y in p["acq"][:n]]
            for p in pairs:
                se = cnt = 0
                for i, ys in p["eval"]:
                    q = float(model.predict([p["subject"], item_dict(i)], labeled))
                    for y in ys:
                        se += (q - y) ** 2
                        cnt += 1
                briers[id(p)].append(se / cnt)
                per_budget[n].append(se / cnt)
        for b in briers.values():
            pair_scores.append(sum(w * x for w, x in zip(WEIGHTS, b)))
    return sum(pair_scores) / len(pair_scores), {n: sum(v) / len(v) for n, v in per_budget.items()}


def load_model(path, train_benches):
    spec = importlib.util.spec_from_file_location("m", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    if hasattr(m, "fit"):
        m.fit(train_benches)
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--rounds", type=int, default=20)
    args = ap.parse_args()
    benches = binary_benchmarks()
    results = {}
    for path in args.models:
        rows = {}
        for held in benches:
            model = load_model(path, [b for b in benches if b != held])
            try:
                alc, by_budget = score(model, held, args.rounds)
            except ValueError:
                continue
            rows[held] = {"alc": round(alc, 4), **{f"b{k}": round(v, 4) for k, v in by_budget.items()}}
        mean = sum(r["alc"] for r in rows.values()) / len(rows)
        results[path] = {"mean_alc": round(mean, 4), "per_benchmark": rows}
        print(path, json.dumps(results[path], indent=1))
