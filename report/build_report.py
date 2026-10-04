"""Fill REPORT.in.md from the JSON outputs and check the numbers written by hand.

Writes report/REPORT.md. Exits non-zero if a placeholder is left, a hand-written
number has no source, or the text contains an en or em dash.
"""
import json
import re
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import repeats  # noqa: E402  (prints its comparison; the numbers below are recomputed)

data = json.loads((HERE / "data.json").read_text())
scores = json.loads((HERE / "local_scores.json").read_text())
overlap = json.loads((HERE / "overlap.json").read_text())
M, P = scores["models"], scores["paired"]
arms = json.loads((HERE.parent / "arms" / "analysis.json").read_text())
prof = json.loads((HERE.parent / "arms" / "budget_profile.json").read_text())
local_arms = dict(re.findall(r"^([A-F]) .*?diff ([+-][\d.]+)", (HERE.parent / "arms" / "local_arms.log").read_text(), re.M))
BENCH = ["matharena", "multi_swebench", "real_webagents", "researchcodebench", "swe_rebench"]
f4 = "{:.4f}".format

fill = {
    "data_rows": "\n".join(f"| {b} | {d['subjects']} | {d['items']:,} | {d['responses']:,} | {d['pairs_80']} | "
                           f"{d['accuracy']:.0%} |" for b, d in data.items() if d["used"]),
    "score_rows": "\n".join(f"| {n} | {f4(m['alc'])} | " + " | ".join(f4(m['per_benchmark'][b]) for b in BENCH) + " |"
                            for n, m in M.items()),
    "paired_rows": "\n".join(f"| {k.replace(' - ', ' against ')} | {v['diff']:+.4f} | {v['se']:.4f} | {v['better']} of {v['n']} | "
                             f"{v['worse']} of {v['n']} |" for k, v in P.items()),
    "budget_rows": "\n".join(f"| {n} | " + " | ".join(f4(m['per_budget'][str(b)]) for b in (0, 1, 3, 7, 15, 31)) + " |"
                             for n, m in M.items() if n != "constant 0.5"),
    "v3_alc": f4(M["v3"]["alc"]),
    "mean_alc": f4(M["empirical mean"]["alc"]),
    "d_v1_mean": f4(M["empirical mean"]["alc"] - M["v1"]["alc"]),
    "d_v1_const": f4(M["constant 0.5"]["alc"] - M["v1"]["alc"]),
    "v1_b0": f4(M["v1"]["per_budget"]["0"]),
    "v1_b1": f4(M["v1"]["per_budget"]["1"]),
    "mean_b1": f4(M["empirical mean"]["per_budget"]["1"]),
    "arm_n": str(arms["submissions"]), "arm_pairs": str(arms["pairs"]),
    "arm_nA": f'{arms["per_arm_submissions"]["A"]} of A', "arm_nB": f'{arms["per_arm_submissions"]["B"]} of B',
    "arm_nC": f'{arms["per_arm_submissions"]["C"]} of C',
    "arm_BA": f'{arms["B-A"]["diff"]:+.4f} (standard error {arms["B-A"]["se"]:.4f})',
    "arm_CA": f'{arms["C-A"]["diff"]:+.4f} (standard error {arms["C-A"]["se"]:.4f})',
    "loc_B": local_arms["B"].lstrip("+"), "loc_E": local_arms["E"].lstrip("+"),
    "prof_n": str(prof["pairs"]), **{f"prof_{b}": f4(prof["budget"][str(b)]) for b in (0, 1, 7, 31)},
    "d_text": f4(P["v2 with text features - v2"]["diff"]),
    "d_v4": f"{P['v4 - v3']['diff']:+.4f} (standard error {P['v4 - v3']['se']:.4f})",
}
text = (HERE / "REPORT.in.md").read_text()
for k, v in fill.items():
    text = text.replace("{" + k + "}", v)

problems = re.findall(r"\{[a-z0-9_]+\}", text)
# Numbers typed by hand in the template, each with its source.
pf, fb = repeats.platform, repeats.feedback
v1, v2 = list(pf["v1"].values()), list(pf["v2"].values())
adj = {v: [pf[v][i] - fb[i][0] for i in pf[v] if i in fb] for v in pf}
se = lambda a, b: (st.variance(a) / len(a) + st.variance(b) / len(b)) ** 0.5  # noqa: E731
used = [o for b, o in overlap.items() if o > 0]
hand = [f4(st.mean(v1)), f4(st.mean(v2)), f4(st.mean(v1) - st.mean(v2)), f4(se(v1, v2)),
        f4(st.mean(adj["v1"]) - st.mean(adj["v2"])), f4(se(adj["v1"], adj["v2"])),
        f"{min(used):.0%}", f"{max(used):.0%}", "0.182867", "0.179449", "0.163046", "0.250000",
        f4(M["v3"]["alc"] - M["v2"]["alc"]).lstrip("-")]
hand += [f"{x:.6f}" for x in v1 + v2]
problems += [f"hand number not in text: {h}" for h in hand if h not in text]
if re.search("[–—]", text):
    problems.append("en or em dash")
(HERE / "REPORT.md").write_text(text)
if problems:
    sys.exit("problems: " + ", ".join(problems))
print(f"REPORT.md written, {len(fill)} fields filled, {len(hand)} hand numbers checked")
