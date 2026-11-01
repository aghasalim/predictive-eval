"""README figures from report/local_scores.json and report/data.json. Writes report/fig_*.png."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
COLORS = {"constant 0.5": "#bbbbbb", "empirical mean": "#8c8c8c", "v2 with text features": "#e08214",
          "v1": "#9ecae1", "v2": "#4292c6", "v3": "#08306b", "v4": "#b2182b"}


def fig_budget(s):
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, c in COLORS.items():
        pb = s["models"][name]["per_budget"]
        ks = sorted(pb, key=int)
        ax.plot(range(len(ks)), [pb[k] for k in ks], "o-", color=c, ms=4,
                label=f"{name} (ALC {s['models'][name]['alc']:.4f})")
    ax.set_xticks(range(len(ks)), ks)
    ax.set_xlabel("labels revealed for the pair (budget)")
    ax.set_ylabel("mean Brier score (lower is better)")
    ax.set_ylim(0.17, 0.30)
    em = s["models"]["empirical mean"]["per_budget"]["1"]
    ax.annotate(f"{em:.3f} at budget 1", (1, 0.30), xytext=(1.15, 0.292), fontsize=8, color="#8c8c8c")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(HERE / "fig_budget.png", dpi=150)


def fig_paired(s):
    p = s["paired"]
    names = list(p)
    fig, ax = plt.subplots(figsize=(8, 3))
    for i, n in enumerate(names):
        d, se = p[n]["diff"], p[n]["se"]
        ax.errorbar(d, i, xerr=1.96 * se, fmt="o", color="#08306b" if d < 0 else "#b2182b", capsize=3)
        ax.text(1.02, i, f"better in {p[n]['better']}, worse in {p[n]['worse']} of {p[n]['n']}",
                va="center", fontsize=8, transform=ax.get_yaxis_transform())
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.set_xlim(-0.05, 0.02)
    ax.set_xlabel("difference in local Brier ALC, same draws (negative is better), 95% interval")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE / "fig_paired.png", dpi=150)


def fig_data(d):
    used = {k: v for k, v in d.items() if v["used"]}
    names = list(used)
    fig, (a, b) = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True)
    a.barh(names, [used[n]["responses"] for n in names], color="#4292c6")
    for i, n in enumerate(names):
        a.text(used[n]["responses"], i, f" {used[n]['subjects']} subject{'s' * (used[n]['subjects'] > 1)}, {used[n]['items']:,} items",
               va="center", fontsize=8)
    a.set_xlim(0, 1.8 * max(v["responses"] for v in used.values()))
    a.set_xlabel("responses (thousands)")
    a.xaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}")
    b.barh(names, [used[n]["accuracy"] for n in names], color="#8c8c8c")
    for i, n in enumerate(names):
        b.text(used[n]["accuracy"], i, f" {used[n]['accuracy']:.2f}", va="center", fontsize=8)
    b.set_xlim(0, 1)
    b.set_xlabel("share of responses correct")
    a.invert_yaxis()
    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE / "fig_data.png", dpi=150)


if __name__ == "__main__":
    s = json.loads((HERE / "local_scores.json").read_text())
    fig_budget(s)
    fig_paired(s)
    fig_data(json.loads((HERE / "data.json").read_text()))
