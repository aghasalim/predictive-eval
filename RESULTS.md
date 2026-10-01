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

## Repeat submissions, 2026-10-01

I submitted the same two zips again, alternating them, so each model has five scores on freshly drawn samples. `repeats.py` holds the numbers and the comparison.

| model | submissions | platform ALC |
|---|---|---|
| v1 | 953625, 953903, 953905, 953907 | 0.193979, 0.193899, 0.191791, 0.198023 |
| v2 | 953862, 953902, 953904, 953906, 953908 | 0.213320, 0.173103, 0.179449, 0.163046, 0.188681 |

953899 (v1) is marked Failed by the platform although it shows 0.192684, so I left it out.

Mean platform ALC is 0.1944 for v1 and 0.1835 for v2, a difference of 0.0109 in v2's favour with a standard error of 0.0086. Subtracting each submission's own budget 0 score, which is the same model for both and so measures how hard the sample was, gives a difference of 0.0094 in v2's favour with a standard error of 0.0075. Both point the same way as the local simulation, and both are a little over one standard error, so they are a lean, not a result. The worry from the first v2 submission did not hold up: averaged over its submissions, v2 improves more from budget 0 to budget 1 than v1 does (0.0200 against 0.0116), with a standard error of 0.0134.

v2's scores spread much more than v1's (standard deviation 0.0191 against 0.0026). The samples differ between submissions, so part of that is the draw, but the tag offsets probably add to it.

## v3 and an environment probe, 2026-10-01

v3 adds an online offset for each question, learned from other subjects' revealed labels on the same question. Each pair splits its items on its own, so at budget 31 between 13% and 72% of evaluation items had already been labelled for another subject on four of the five public benchmarks (none on swe_rebench, which has one subject). `paired3.py` compares it with the offsets switched off, which is v2, on identical draws: 0.2146 against 0.2171 at sd_i 2.0, a difference of 0.0025 with a standard error of 0.0004, better in 16 of 20 cells and worse in none (the 4 ties are swe_rebench). Submitted as 954619.

954620 is not a predictor. It imports numpy, torch and sentence-transformers and loads all-MiniLM-L6-v2 through models.txt at import time, then predicts 0.5. If it finishes, those are available in the scoring container; if it fails, at least one is not.

Results: 954619 (v3) scored 0.182867. Its sample is a fresh draw, so this one score sits within v2's spread (0.163 to 0.213) and does not separate v3 from v2 on the platform; the paired local comparison above is still the better evidence. 954620 (the probe) finished with 0.250000, so numpy, torch and sentence-transformers all load in the scoring container and a text embedding model is usable.

## v4, text embedding neighbours, 2026-10-01

`models/irt4.py` adds to v3 an offset shared between questions that read alike: each labelled observation leaves a residual against the model without item offsets, and a question takes a similarity weighted average of other questions' residuals on the same benchmark, from all-MiniLM-L6-v2 vectors, shrunk and scaled. `paired4.py` compares it with the offset switched off, which is v3, on identical draws (v3 scores 0.2146).

Over 11 settings the best was 0.2143, a difference of 0.0003 with a standard error of 0.0002, which is no gain once the choice among 11 settings is counted. The offset helps on matharena (up to 0.0026) and a little on multi_swebench, and hurts on researchcodebench, whose items are long and open with shared text; stronger settings cost up to 0.0251 there. Centring the vectors on each benchmark's mean and embedding both ends of long items removed most of that damage (`tune4b.log`) but did not leave a gain. Not submitted; v3 stays the current model.
