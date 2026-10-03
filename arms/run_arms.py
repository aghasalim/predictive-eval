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
            m = __import__("re").search(r"Submission created: (\d+)", out)
            sid = int(m.group(1)) if m else None
            with open(LOG, "a") as f:
                f.write(json.dumps({"arm": arm, "id": sid, "round": r}) + "\n")
            print(arm, sid, flush=True)
            time.sleep(20)


def budgets(sid):
    """Mean Brier per label budget from the submission's detailed results, if published."""
    try:
        d = S.get(f"{BASE}/submissions/{sid}/get_details/", timeout=60).json()
        url = d.get("detailed_result")
        if not url:
            return None
        text = requests.get(url, timeout=60).text
        return text
    except Exception:
        return None


def collect():
    rows = [json.loads(l) for l in open(LOG)]
    out = []
    for r in rows:
        s = S.get(f"{BASE}/submissions/{r['id']}/", timeout=60).json()
        alc = next((float(x["score"]) for x in s.get("scores", []) if x.get("column_key") == "brier"), None)
        out.append(r | {"status": s.get("status"), "alc": alc})
    by = {}
    for r in out:
        if r["status"] == "Finished" and r["alc"] is not None:
            by.setdefault(r["arm"], []).append(r["alc"])
    for arm, xs in sorted(by.items()):
        sd = st.stdev(xs) if len(xs) > 1 else float("nan")
        print(f"{arm}: n {len(xs)}  mean ALC {st.mean(xs):.4f}  sd {sd:.4f}")
    (HERE / "collected.json").write_text(json.dumps(out, indent=1))
    print("pending:", sum(r["status"] not in ("Finished", "Failed") for r in out))


if __name__ == "__main__":
    submit(int(sys.argv[2])) if sys.argv[1] == "submit" else collect()
