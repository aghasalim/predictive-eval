"""Grid over the shrinkage settings, scored leave-one-benchmark-out."""
import itertools
import sys

import sim

benches = sim.binary_benchmarks()
grid = list(itertools.product([0.4, 0.7], [0.7, 1.0, 1.5], [0.25, 0.5]))
for sd_a, sd_d, w in grid:
    scores = []
    for held in benches:
        m = sim.load_model("models/irt.py", [b for b in benches if b != held])
        m.PARAMS.update(sd_a=sd_a, sd_d=sd_d, w=w)
        m._cache.clear()
        scores.append(sim.score(m, held, n_rounds=4, seed=1)[0])
    print(f"sd_a={sd_a} sd_d={sd_d} w={w} mean ALC {sum(scores)/len(scores):.4f}", flush=True)
