"""Mean platform Brier per label budget over every pair result from the v1, v2 and v3
submissions and the arm submissions (scoring logs cached in arms/logs/). Writes
arms/budget_profile.json."""
import json
import statistics as st
import sys

sys.path.insert(0, "arms")
from feedback import pairs  # noqa: E402

ids = [953625, 953903, 953905, 953907, 953862, 953902, 953904, 953906, 953908, 954619]
ids += [json.loads(l)["id"] for l in open("arms/log.jsonl")]
rows = [r for i in ids for r in pairs(i).values() if all(b in r for b in (0, 1, 3, 7, 15, 31))]
out = {"pairs": len(rows), "budget": {b: st.mean(r[b] for r in rows) for b in (0, 1, 3, 7, 15, 31)}}
json.dump(out, open("arms/budget_profile.json", "w"), indent=1)
print(out)
