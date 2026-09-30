# Results

Brier ALC, lower is better. Always predicting 0.5 scores 0.25.

| date | submission | file | local ALC (sim.py) | platform ALC | notes |
|---|---|---|---|---|---|
| 2026-09-30 | 953625 | irt_v1.zip | 0.2203 | 0.193979 | shrunk logistic, ability by model name, 9 subject-benchmark pairs in the formative sample |
| 2026-09-30 | 953862 | irt_v2.zip | 0.2171 | 0.213320 | v1 plus online offsets for items that share an item_features tag; text features tested and left off (0.2276 locally) |

Platform feedback for 953625, mean Brier score over the 9 pairs at each label budget:

| budget | 0 | 1 | 3 | 7 | 15 | 31 |
|---|---|---|---|---|---|---|
| mean Brier | 0.2259 | 0.2066 | 0.2014 | 0.1833 | 0.1770 | 0.1774 |

The worst pairs were subject 681510 on benchmark 119137 (0.278) and subject 542695 on benchmark 893575 (0.265); the best was subject 707538 on benchmark 213669 (0.042). The formative sample is redrawn for every submission, so the platform score of one submission is noisy.

Platform feedback for 953862 (v2), mean Brier score over its 9 pairs:

| budget | 0 | 1 | 3 | 7 | 15 | 31 |
|---|---|---|---|---|---|---|
| mean Brier | 0.2393 | 0.2424 | 0.2155 | 0.2051 | 0.1900 | 0.1878 |

v1 and v2 make identical predictions at budget 0, because the tag offsets need revealed labels. Their budget 0 scores still differ, 0.2259 against 0.2393, so that gap of about 0.013 is the formative sample alone. Only one pair (subject 681510, benchmark 119137) appears in both samples: 0.278 for v1 and 0.298 for v2. One submission each cannot separate the two models. The one sign against v2 is that it got worse from budget 0 to budget 1 (0.2393 to 0.2424) while v1 improved (0.2259 to 0.2066), which suggests the tag offsets move too far on the first label.
