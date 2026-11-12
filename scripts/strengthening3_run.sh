#!/bin/bash
# Freeze the strengthening-3 manifest from a clean commit and run the gated MNIST test matrix.
# Usage: [OUT=dir] scripts/strengthening3_run.sh [MAXJOBS]   (records go to $OUT/final-results)
set -euo pipefail
cd "$(dirname "$0")/.."
max=${1:-14}
base=${OUT:-artifacts/strengthening3}
out=$base/final-results
manifest=$base/manifest.json
export MANIFEST=$manifest
protocol=configs/strengthening3-protocol.yaml
ledger=configs/strengthening3-validation-ledger.json
mkdir -p "$out" "$(dirname $manifest)"
if [ ! -f "$manifest" ]; then
  .venv/bin/python - <<'EOF'
import json, os, yaml
from pathlib import Path
from terel.resubmission.provenance import build_run_manifest
configuration = yaml.safe_load(Path("configs/strengthening3-final-matrix.yaml").read_text())
manifest = build_run_manifest(phase="confirmatory", frozen=True, selection_complete=True,
    protocol_path="configs/strengthening3-protocol.yaml",
    validation_ledger_path="configs/strengthening3-validation-ledger.json",
    repository=".", configuration=configuration)
Path(os.environ["MANIFEST"]).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print("manifest", manifest["code_commit"], manifest["configuration_sha256"])
EOF
fi
runs=$(.venv/bin/python -c "import yaml;print(' '.join(r['id'] for r in yaml.safe_load(open('configs/strengthening3-final-matrix.yaml'))['datasets']['mnist']['runs']))")
seeds=$(.venv/bin/python -c "import yaml;print(' '.join(str(s) for s in yaml.safe_load(open('configs/strengthening3-final-matrix.yaml'))['seeds']))")
export CUBLAS_WORKSPACE_CONFIG=:4096:8
for run in $runs; do
  for seed in $seeds; do
    [ -f "$out/mnist/$run/seed-$seed.json" ] && continue
    while [ "$(pgrep -fc 'terel.resubmission.confirmatory run')" -ge "$max" ]; do sleep 15; done
    nohup .venv/bin/python -m terel.resubmission.confirmatory run --manifest "$manifest" \
      --validation-ledger "$ledger" --protocol "$protocol" --repository . --output "$out" \
      --dataset mnist --run-id "$run" --seed "$seed" --allow-test > "$out/$run-$seed.log" 2>&1 &
    sleep 3
  done
done
wait
# final pass: verifies every record against the manifest and writes the complete ledger
.venv/bin/python -m terel.resubmission.confirmatory run --manifest "$manifest" --validation-ledger "$ledger" \
  --protocol "$protocol" --repository . --output "$out" --dataset mnist --allow-test > "$out/final-pass.log" 2>&1 || true
cat "$out/confirmatory-ledger.json"
