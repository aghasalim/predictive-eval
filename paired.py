"""Paired comparison on identical draws: every (held-out benchmark, seed)
cell is scored by both models, and the difference is taken per cell."""
import statistics
import sim

benches = sim.binary_benchmarks()
SEEDS = (0, 1, 2, 3)


def cells(path, over):
    out = {}
    for held in benches:
        m = sim.load_model(path, [b for b in benches if b != held])
        m.PARAMS.update(over)
        m._cache.clear()
        for s in SEEDS:
            out[(held, s)] = sim.score(m, held, n_rounds=8, seed=s)[0]
    return out


base = cells("models/irt.py", {})
print(f"v1 mean {statistics.mean(base.values()):.4f}", flush=True)
for sd_g in (0.3, 0.5, 1.0):
    alt = cells("models/irt2.py", {"h_scale": 0.0, "sd_g": sd_g})
    d = [alt[k] - base[k] for k in base]
    se = statistics.stdev(d) / len(d) ** 0.5
    better = sum(x < 0 for x in d)
    print(f"tags sd_g={sd_g}: mean {statistics.mean(alt.values()):.4f}  diff {statistics.mean(d):+.4f} "
          f"(se {se:.4f})  better in {better} of {len(d)} cells", flush=True)
