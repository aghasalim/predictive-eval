"""Every local number in the report, from one run on identical draws.

Each model is scored on the same cells: 5 held-out benchmarks x 4 seeds x 8
formative rounds of up to 8 subject-benchmark pairs. Learned versions use the
settings they were submitted with. Writes report/local_scores.json.
"""
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import sim  # noqa: E402

SEEDS = (0, 1, 2, 3)
SHIPPED = {"sd_a": 0.7, "sd_d": 1.0, "w": 0.5, "sd_theta": 1.0}
MODELS = {
    "constant 0.5": ("models/const.py", {}),
    "empirical mean": ("paiec_baseline/empirical_mean/model.py", {}),
    "v1": ("models/irt.py", SHIPPED),
    "v2": ("models/irt2.py", {**SHIPPED, "sd_g": 1.0, "h_scale": 0.0}),
    "v2 with text features": ("models/irt2.py", {**SHIPPED, "sd_g": 1.0, "h_scale": 1.0}),
    "v3": ("models/irt3.py", {**SHIPPED, "sd_g": 1.0, "h_scale": 0.0, "sd_i": 2.0}),
    "v4": ("models/irt4.py", {**SHIPPED, "sd_g": 1.0, "h_scale": 0.0, "sd_i": 2.0,
                              "lam": 2.0, "tau": 0.2, "k_e": 0.5}),
}
PAIRS = (("v1", "empirical mean"), ("v2", "v1"), ("v2 with text features", "v2"), ("v3", "v2"), ("v4", "v3"))


def run(path, over):
    cells, budgets = {}, {}
    for held in sim.binary_benchmarks():
        m = sim.load_model(path, [b for b in sim.binary_benchmarks() if b != held])
        if hasattr(m, "preload"):
            m.preload("mdb/emb.npz")
        if hasattr(m, "PARAMS"):
            m.PARAMS.update(over)
        for name in ("_cache", "_centre"):
            getattr(m, name, {}).clear()
        for s in SEEDS:
            alc, by_budget = sim.score(m, held, n_rounds=8, seed=s)
            cells[f"{held}/{s}"] = alc
            for n, v in by_budget.items():
                budgets.setdefault(n, []).append(v)
    return cells, {n: st.mean(v) for n, v in budgets.items()}


if __name__ == "__main__":
    out = {"models": {}, "paired": {}}
    cells = {}
    for name, (path, over) in MODELS.items():
        cells[name], by_budget = run(path, over)
        per = {b: st.mean(v for k, v in cells[name].items() if k.startswith(b + "/")) for b in sim.binary_benchmarks()}
        out["models"][name] = {"alc": st.mean(cells[name].values()), "per_benchmark": per, "per_budget": by_budget}
        print(name, round(out["models"][name]["alc"], 4), {b: round(v, 4) for b, v in per.items()}, flush=True)
    for a, b in PAIRS:
        d = [cells[a][k] - cells[b][k] for k in cells[a]]
        out["paired"][f"{a} - {b}"] = {"diff": st.mean(d), "se": st.stdev(d) / len(d) ** 0.5,
                                       "better": sum(x < -1e-9 for x in d), "worse": sum(x > 1e-9 for x in d), "n": len(d)}
        print(a, "-", b, out["paired"][f"{a} - {b}"], flush=True)
    Path(__file__).with_name("local_scores.json").write_text(json.dumps(out, indent=1))
