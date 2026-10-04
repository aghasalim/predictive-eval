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


CACHE = Path(__file__).resolve().parent / "logs"


def scoring_log(sid):
    """The submission's scoring stdout, kept in arms/logs/ after the first download."""
    f = CACHE / f"{sid}.txt"
    if f.exists():
        return f.read_text()
    for _ in range(3):
        try:
            d = S.get(f"{API}/submissions/{sid}/get_details/", timeout=60).json()
            log = next((l for l in d["logs"] if l["name"] == "scoring_stdout"), None)
            if not log or not log.get("data_file"):
                return ""
            text = requests.get(log["data_file"], timeout=60).text
            if "Label budget" in text:
                CACHE.mkdir(exist_ok=True)
                f.write_text(text)
            return text
        except (ValueError, requests.RequestException):
            continue
    return ""


def pairs(sid):
    """{(subject, benchmark): {"alc": x, 0: b0, 1: b1, ...}} for one submission."""
    text = scoring_log(sid)
    out = {}
    head, *parts = re.split(r"Label budget: (\d+)", text)
    for s, b, alc, _ in ROW.findall(head):
        out.setdefault((s, b), {})["alc"] = float(alc)
    for budget, body in zip(parts[0::2], parts[1::2]):
        for s, b, brier, _ in ROW.findall(body):
            out.setdefault((s, b), {})[int(budget)] = float(brier)
    return out
