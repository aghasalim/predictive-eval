"""v3's item offsets against the same model with them switched off (which is
v2), on identical draws: every (held-out benchmark, seed) cell is scored by
both, and the difference is taken per cell."""
import statistics
import sim

benches = sim.binary_benchmarks()
SEEDS = (0, 1, 2, 3)


def cells(over):
    out = {}
    for held in benches:
        m = sim.load_model("models/irt3.py", [b for b in benches if b != held])
        m.PARAMS.update(over)
        m._cache.clear()
        for s in SEEDS:
            out[(held, s)] = sim.score(m, held, n_rounds=8, seed=s)[0]
    return out


base = cells({"sd_i": 1e-6})
print(f"item offsets off (v2): mean {statistics.mean(base.values()):.4f}", flush=True)
for sd_i in (0.5, 1.0, 2.0):
    alt = cells({"sd_i": sd_i})
    d = [alt[k] - base[k] for k in base]
    se = statistics.stdev(d) / len(d) ** 0.5
    print(f"sd_i={sd_i}: mean {statistics.mean(alt.values()):.4f}  diff {statistics.mean(d):+.4f} (se {se:.4f})  "
          f"better {sum(x < -1e-9 for x in d)}, worse {sum(x > 1e-9 for x in d)} of {len(d)}", flush=True)
    per = {b: statistics.mean(alt[(b, s)] - base[(b, s)] for s in SEEDS) for b in benches}
    print("   per benchmark:", {b: round(v, 4) for b, v in per.items()}, flush=True)
