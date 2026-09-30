# Results

Brier ALC, lower is better. Always predicting 0.5 scores 0.25.

| date | submission | file | local ALC (sim.py) | platform ALC | notes |
|---|---|---|---|---|---|
| 2026-09-30 | 953625 | irt_v1.zip | 0.2203 | 0.193979 | shrunk logistic, ability by model name, 9 subject-benchmark pairs in the formative sample |

Platform feedback for 953625, mean Brier score over the 9 pairs at each label budget:

| budget | 0 | 1 | 3 | 7 | 15 | 31 |
|---|---|---|---|---|---|---|
| mean Brier | 0.2259 | 0.2066 | 0.2014 | 0.1833 | 0.1770 | 0.1774 |

The worst pairs were subject 681510 on benchmark 119137 (0.278) and subject 542695 on benchmark 893575 (0.265); the best was subject 707538 on benchmark 213669 (0.042). The formative sample is redrawn for every submission, so the platform score of one submission is noisy.
