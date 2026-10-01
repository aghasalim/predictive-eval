"""Compare v1 and v2 over repeated submissions of the same zips.

Platform ALC is the score Codabench shows for each submission. Budget 0 and
budget 1 means come from each submission's formative feedback and are kept only
where the parsed ALC matched that submission's platform score (feedback panels
sometimes showed another submission's data). 953899 is marked Failed by the
platform and is left out.
"""
import statistics as st

platform = {"v1": {"953625": 0.193979, "953903": 0.193899, "953905": 0.191791, "953907": 0.198023},
            "v2": {"953862": 0.213320, "953902": 0.173103, "953904": 0.179449, "953906": 0.163046, "953908": 0.188681}}
feedback = {  # id: (b0, b1), only where parsed ALC matched the platform score
    "953625": (0.2259, 0.2066), "953905": (0.2192, 0.2185), "953907": (0.2274, 0.2126),
    "953862": (0.2393, 0.2424), "953902": (0.2320, 0.1832), "953904": (0.2203, 0.1898),
    "953906": (0.2129, 0.1747), "953908": (0.2077, 0.2221),
}


def summary(xs):
    return st.mean(xs), st.stdev(xs), len(xs)


def compare(name, a, b):
    ma, sa, na = summary(a)
    mb, sb, nb = summary(b)
    se = (sa ** 2 / na + sb ** 2 / nb) ** 0.5
    d = mb - ma
    print(f"{name}: v1 {ma:.4f} (sd {sa:.4f}, n {na})  v2 {mb:.4f} (sd {sb:.4f}, n {nb})  "
          f"v2 minus v1 {d:+.4f}, se {se:.4f}, {d / se:+.1f} se")


compare("platform ALC", list(platform["v1"].values()), list(platform["v2"].values()))
adj = {v: [platform[v][i] - feedback[i][0] for i in platform[v] if i in feedback] for v in platform}
compare("ALC minus own budget 0", adj["v1"], adj["v2"])
step = {v: [feedback[i][1] - feedback[i][0] for i in platform[v] if i in feedback] for v in platform}
compare("budget 0 to budget 1 change", step["v1"], step["v2"])
