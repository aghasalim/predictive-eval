"""Submit the arm zips in rotation and collect their platform scores.

    python arms/run_arms.py submit ROUNDS   # submits A, B, C, A, B, C, ... ROUNDS times each
    python arms/run_arms.py collect         # reads status, ALC and per-budget feedback

Every arm makes the same predictions with no labels, so each submission's own
budget 0 Brier measures how hard its draw was. The comparison subtracts it.
Uses the token saved by paiec_baseline/submit_api.py --login.
"""
import json
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
LOG = HERE / "log.jsonl"
BASE = "https://www.codabench.org/api"
TOKEN = (Path.home() / ".config/paiec/codabench-token").read_text().strip()
S = requests.Session()
S.headers["Authorization"] = f"Token {TOKEN}"
SUBMIT = HERE.parent / "paiec_baseline" / "submit_api.py"


def submit(rounds):
    for r in range(rounds):
        for arm in "ABC":
            out = subprocess.run([sys.executable, str(SUBMIT), str(HERE / f"arm_{arm}.zip")],
                                 capture_output=True, text=True).stdout
            m = __import__("re").search(r"Submission(?: created)?: (\d+)", out)
            sid = int(m.group(1)) if m else None
            with open(LOG, "a") as f:
                f.write(json.dumps({"arm": arm, "id": sid, "round": r}) + "\n")
            print(arm, sid, flush=True)
            time.sleep(20)


def collect():
    """Mean platform ALC per arm, and ALC minus the pair's own budget 0 Brier, which
    is the same model in every arm and so measures how hard each draw was."""
    sys.path.insert(0, str(HERE))
    from feedback import pairs
    rows = [json.loads(l) for l in open(LOG)]
    by, adj, pending = {}, {}, 0
    for r in rows:
        s = S.get(f"{BASE}/submissions/{r['id']}/", timeout=60).json()
        if s.get("status") != "Finished":
            pending += s.get("status") not in ("Failed",)
            continue
        ps = pairs(r["id"])
        if not ps:
            continue
        by.setdefault(r["arm"], []).append(sum(p["alc"] for p in ps.values()) / len(ps))
        adj.setdefault(r["arm"], []).append(sum(p["alc"] - 0.1 * p[0] for p in ps.values()) / len(ps))
    for arm in sorted(by):
        xs, ys = by[arm], adj[arm]
        sd = lambda v: st.stdev(v) if len(v) > 1 else float("nan")
        print(f"{arm}: n {len(xs)}  ALC {st.mean(xs):.4f} (sd {sd(xs):.4f})  ALC minus 0.1 x B0 {st.mean(ys):.4f} (sd {sd(ys):.4f})")
    for a, b in (("A", "B"), ("A", "C")):
        if a in adj and b in adj and len(adj[a]) > 1 and len(adj[b]) > 1:
            d = st.mean(adj[b]) - st.mean(adj[a])
            se = (st.variance(adj[a]) / len(adj[a]) + st.variance(adj[b]) / len(adj[b])) ** 0.5
            print(f"{b} minus {a}: {d:+.4f} (se {se:.4f})")
    print("pending:", pending)


if __name__ == "__main__":
    submit(int(sys.argv[2])) if sys.argv[1] == "submit" else collect()
