"""Score irt2 with one component switched off at a time."""
import sys
import sim

benches = sim.binary_benchmarks()
variants = {
    "tags only (h off)": {"h_scale": 0.0},
    "text only (tags off)": {"sd_g": 1e-6},
    "both, half-strength text": {"h_scale": 0.5},
}
for name, over in variants.items():
    per = {}
    for held in benches:
        m = sim.load_model("models/irt2.py", [b for b in benches if b != held])
        m.PARAMS.update(over)
        m._cache.clear()
        per[held] = round(sim.score(m, held, n_rounds=10, seed=0)[0], 4)
    print(f"{name:26} mean {sum(per.values()) / len(per):.4f}  {per}", flush=True)
