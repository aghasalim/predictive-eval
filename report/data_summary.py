"""Size of each measurement-db benchmark used locally. Writes report/data.json."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import sim  # noqa: E402

out = {}
for d in sorted(p.name for p in sim.DATA.iterdir() if (p / "benchmarks.parquet").exists()):
    b = pd.read_parquet(sim.DATA / d / "benchmarks.parquet").iloc[0]
    subj, items, resp = sim.load(d) if d in sim.binary_benchmarks() else (None, None, None)
    row = {"response_type": str(b["response_type"]), "granularity": str(b["granularity"]), "used": d in sim.binary_benchmarks()}
    if row["used"]:
        per = resp.groupby("subject_id")["item_id"].nunique()
        row.update(subjects=int(resp["subject_id"].nunique()), items=int(resp["item_id"].nunique()),
                   responses=int(len(resp)), pairs_80=int((per >= 80).sum()), accuracy=float(resp["response"].mean()))
    out[d] = row
    print(d, row)
Path(__file__).with_name("data.json").write_text(json.dumps(out, indent=1))
