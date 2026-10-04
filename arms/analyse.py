"""Arm comparison at the pair level: Brier ALC of each pair regressed on the pair's own
budget 0 Brier (identical predictions in every arm, so it only measures how hard the pair
is) plus arm indicators. Standard errors are clustered by submission."""
import json
import sys

import numpy as np

sys.path.insert(0, "arms")
from feedback import pairs  # noqa: E402

rows = []
for r in (json.loads(l) for l in open("arms/log.jsonl")):
    for k, p in pairs(r["id"]).items():
        if 0 in p and "alc" in p:
            rows.append((r["arm"], r["id"], p[0], p["alc"]))
arms = sorted({a for a, *_ in rows})
y = np.array([x[3] for x in rows])
X = np.column_stack([np.ones(len(rows)), [x[2] for x in rows]] + [[1.0 if x[0] == a else 0.0 for x in rows] for a in arms[1:]])
beta, *_ = np.linalg.lstsq(X, y, rcond=None)
res = y - X @ beta
XtX = np.linalg.inv(X.T @ X)
meat = np.zeros((X.shape[1],) * 2)
for sid in {x[1] for x in rows}:
    idx = [i for i, x in enumerate(rows) if x[1] == sid]
    g = X[idx].T @ res[idx]
    meat += np.outer(g, g)
se = np.sqrt(np.diag(XtX @ meat @ XtX))
print(f"pairs {len(rows)}, submissions {len({x[1] for x in rows})}, slope on budget 0 Brier {beta[1]:.3f}")
for j, a in enumerate(arms[1:], start=2):
    print(f"{a} minus {arms[0]}: {beta[j]:+.4f} (se {se[j]:.4f})")
