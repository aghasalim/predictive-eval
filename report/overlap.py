"""Share of evaluation items that, at budget 31, already have a revealed label
from another pair in the same round, on the 20 shared cells. Writes report/overlap.json."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import sim  # noqa: E402

out = {}
for b in sim.binary_benchmarks():
    hit = tot = 0
    for seed in (0, 1, 2, 3):
        for pairs, _, _ in sim.rounds(b, 8, seed):
            for p in pairs:
                others = {i for q in pairs if q is not p for i, _ in q["acq"][:31]}
                hit += sum(i in others for i, _ in p["eval"])
                tot += len(p["eval"])
    out[b] = hit / tot
    print(b, round(100 * out[b]))
Path(__file__).with_name("overlap.json").write_text(json.dumps(out, indent=1))
