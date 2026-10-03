"""Ask the LLM about training items once and cache the answers (for tuning lam and kappa).

Usage: .venv/bin/python llm_cache.py PER_BENCHMARK
Reads the key and model from models/private_config.json, which is git-ignored:
    {"api_key": "...", "model": "..."}
Writes models/llm_cache.json (also git-ignored). Prints calls made and failures.
"""
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "models")
import llm_irt as m  # noqa: E402
import sim  # noqa: E402

per = int(sys.argv[1])
if not m.CONFIG.get("api_key"):
    sys.exit("put your key in models/private_config.json first")
todo = []
for b in sim.binary_benchmarks():
    _, items, _ = sim.load(b)
    ids = sorted(items.index)
    random.Random(0).shuffle(ids)
    for iid in ids[:per]:
        row = items.loc[iid]
        todo.append({"item_content": sim.s(row.get("content")), "item_features": sim.s(row.get("item_features")),
                     "benchmark_id": b})
before = len(m._llm)
with ThreadPoolExecutor(8) as pool:
    list(pool.map(m.llm_logit, todo))
json.dump(m._llm, open("models/llm_cache.json", "w"))
print(f"items {len(todo)}, new calls {len(m._llm) - before}, failed {sum(v is None for v in m._llm.values())}")
