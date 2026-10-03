"""Pair-level platform feedback: each pair's Brier ALC and per-budget Brier from a
submission's scoring log. Subject and benchmark ids are permanent across submissions,
so arms can be compared on the pairs they share."""
import re
from pathlib import Path

import requests

S = requests.Session()
S.headers["Authorization"] = "Token " + (Path.home() / ".config/paiec/codabench-token").read_text().strip()
API = "https://www.codabench.org/api"
ROW = re.compile(r"^(subject_\d+)\s+(benchmark_\d+)\s+\d+\s+([\d.]+)\s+([\d.]+)\s*$", re.M)


def pairs(sid):
    """{(subject, benchmark): {"alc": x, 0: b0, 1: b1, ...}} for one submission."""
    d = S.get(f"{API}/submissions/{sid}/get_details/", timeout=60).json()
    log = next((l for l in d["logs"] if l["name"] == "scoring_stdout"), None)
    if not log or not log.get("data_file"):
        return {}
    text = requests.get(log["data_file"], timeout=60).text
    out = {}
    head, *parts = re.split(r"Label budget: (\d+)", text)
    for s, b, alc, _ in ROW.findall(head):
        out.setdefault((s, b), {})["alc"] = float(alc)
    for budget, body in zip(parts[0::2], parts[1::2]):
        for s, b, brier, _ in ROW.findall(body):
            out.setdefault((s, b), {})[int(budget)] = float(brier)
    return out
