"""Local paired check of the shrinkage arms: v3 with different sd_d and sd_a, on the same
20 cells as report/local_scores.py. Prints each setting's difference from arm A."""
import statistics as st
import sys
sys.path.insert(0, ".")
import sim

SHIP = {"sd_a": 0.7, "sd_d": 1.0, "w": 0.5, "sd_theta": 1.0, "sd_g": 1.0, "h_scale": 0.0, "sd_i": 2.0}
ARMS = {"A": {}, "B": {"sd_d": 1.5, "sd_a": 1.0}, "C": {"sd_d": 0.7},
        "D": {"sd_d": 2.0, "sd_a": 1.0}, "E": {"sd_d": 3.0, "sd_a": 1.5}, "F": {"sd_d": 1.5, "sd_a": 1.0, "sd_i": 3.0}}


def cells(over):
    out = {}
    for held in sim.binary_benchmarks():
        m = sim.load_model("models/irt3.py", [b for b in sim.binary_benchmarks() if b != held])
        m.PARAMS.update({**SHIP, **over}); m._cache.clear()
        for s in (0, 1, 2, 3):
            out[(held, s)] = sim.score(m, held, n_rounds=8, seed=s)[0]
    return out


base = cells({})
print(f"A: {st.mean(base.values()):.4f}", flush=True)
for name, over in ARMS.items():
    if name == "A":
        continue
    c = cells(over)
    d = [c[k] - base[k] for k in base]
    print(f"{name} {over}: {st.mean(c.values()):.4f}  diff {st.mean(d):+.4f} (se {st.stdev(d) / len(d) ** .5:.4f})  "
          f"better {sum(x < -1e-9 for x in d)} worse {sum(x > 1e-9 for x in d)}", flush=True)
