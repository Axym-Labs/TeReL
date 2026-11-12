"""Print one line per finished run in a round directory: primary probe accuracy, head sensitivity, diagnostics."""
import json, sys
from pathlib import Path
for f in sorted(Path(sys.argv[1]).glob("*.json")):
    d = json.loads(f.read_text())
    heads = " ".join(f"{r['learning_rate']:g}:{100*r['accuracy']:.2f}" for r in d["grid"])
    diag = d.get("diagnostics", {})
    print(f"{f.stem:26s} {100*d['reference_accuracy']:6.2f}  heads[{heads}]  rank={diag.get('effective_rank', float('nan')):.0f} medvar={diag.get('median_feature_variance', float('nan')):.3g}")
