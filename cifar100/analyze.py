"""Summaries and paired 95% Student-t intervals for a CIFAR-100 result directory."""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

root = Path(sys.argv[1])
reference = sys.argv[2] if len(sys.argv) > 2 else "terel"
values = {}
for run in sorted(p for p in root.iterdir() if p.is_dir()):
    values[run.name] = {
        int(f.stem.split("-")[1]): json.loads(f.read_text())["metrics"]["accuracy"]
        for f in run.glob("seed-*.json")
    }
summary = {}
for name, by_seed in values.items():
    x = np.array([by_seed[s] for s in sorted(by_seed)])
    entry = {"seeds": sorted(by_seed), "values": x.tolist(), "mean": x.mean(),
             "sample_sd": x.std(ddof=1) if len(x) > 1 else float("nan")}
    if name != reference and reference in values:
        common = sorted(set(by_seed) & set(values[reference]))
        if len(common) > 1:
            d = np.array([values[reference][s] - by_seed[s] for s in common])
            half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
            entry[f"{reference}_minus"] = {"mean": d.mean(), "ci95": [d.mean() - half, d.mean() + half]}
    summary[name] = entry
    print(f"{name:22s} {100*entry['mean']:6.2f} ± {100*entry['sample_sd']:.2f}",
          "" if f"{reference}_minus" not in entry else
          "  Δ={:+.2f} [{:+.2f},{:+.2f}]".format(*(100*v for v in [entry[f'{reference}_minus']['mean'], *entry[f'{reference}_minus']['ci95']])))
(root / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
