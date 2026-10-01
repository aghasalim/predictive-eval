# predictive-eval

My entry to the NeurIPS 2026 Predictive AI Evaluation Competition on Codabench: predict whether an AI system answers a benchmark item correctly, on a benchmark it was never trained on, while a few labels are revealed at a time.

The model is a small shrunk logistic predictor. Ability by model name comes from the training benchmarks, and the new benchmark's difficulty, the system's own offset, tag offsets and per question offsets are all estimated online as MAP estimates under Gaussian priors. The technical report is [report/REPORT.md](report/REPORT.md).

| model | local Brier ALC | best platform ALC |
|---|---|---|
| constant 0.5 | 0.2500 | |
| organisers' empirical mean | 0.2625 | |
| v1 | 0.2203 | 0.191791 |
| v2 | 0.2171 | 0.163046 |
| v3 | 0.2146 | 0.182867 |

Lower is better. Local scores hold out one benchmark at a time on the public measurement-db release (`sim.py`); each platform submission is scored on a fresh sample of about nine pairs, so single platform scores are noisy. `RESULTS.md` is the running log, and the report covers two text based ideas that did not help.

## Running it

```bash
python -m venv .venv && .venv/bin/pip install pandas pyarrow numpy sentence-transformers
./fetch.sh                                   # gated data (accept the terms on Hugging Face first) and the organisers' baselines
.venv/bin/python sim.py models/irt3.py       # local score of one model
.venv/bin/python report/local_scores.py      # every model on the same draws, about 20 minutes on a laptop CPU
.venv/bin/python report/build_report.py      # fills the report and checks every number in it
```

`submission/`, `submission2/` and `submission3/` are the archives as submitted: a `model.py` and its fitted parameters. The data is not included; `fetch.sh` downloads it.
