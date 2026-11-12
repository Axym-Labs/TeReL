"""Build the strengthening-3 validation ledger and final MNIST matrix from validation records.

Reads cifar100/mnist-select2/<run>-s<seed>.json (validation only), selects the probe learning
rate per method by mean accuracy over seeds 401-403, and writes
configs/strengthening3-validation-ledger.json and configs/strengthening3-final-matrix.yaml.
"""

import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "cifar100" / "mnist-select2"
SEEDS = (401, 402, 403)
# final-matrix run id -> validation record id
SOURCES = {
    "terel": "mnist-terel",
    "terel-offline": "mnist-offline-lr0.4",
    "random": "mnist-random",
    "supervised-linear": "mnist-supervised-linear",
    "supervised-offline": "mnist-bp-lr0.3",
    "batch-sfa": "mnist-batch-sfa",
    "incremental-sfa": "mnist-incremental-sfa",
}
INHERIT_TEREL = ("no-temporal", "no-variance", "no-covariance", "shuffled-order", "terel-online-scaled")


def selection(record_id):
    records = [json.loads((RECORDS / f"{record_id}-s{seed}.json").read_text()) for seed in SEEDS]
    rates = [row["learning_rate"] for row in records[0]["grid"]]
    table = {rate: [next(r["accuracy"] for r in rec["grid"] if r["learning_rate"] == rate) for rec in records]
             for rate in rates}
    best = max(rates, key=lambda rate: np.mean(table[rate]))
    return {
        "record_id": record_id,
        "encoder": records[0]["encoder"],
        "probe_learning_rate": best,
        "accuracy_by_probe_learning_rate": {str(rate): {"values": v, "mean": float(np.mean(v))} for rate, v in table.items()},
    }


def main(supcon_record="mnist-supcon-lr0.3"):
    sources = dict(SOURCES, **{"layer-local-supcon": supcon_record})
    chosen = {run_id: selection(record_id) for run_id, record_id in sources.items()}
    ledger = {
        "schema_version": 1,
        "protocol": "configs/strengthening3-protocol.yaml",
        "selection_complete": True,
        "heldout_accessed": False,
        "selection_split": "MNIST train-derived validation",
        "selection_seeds": list(SEEDS),
        "selections": chosen,
    }
    (ROOT / "configs" / "strengthening3-validation-ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")

    old = yaml.safe_load((ROOT / "configs" / "strengthening2-final-matrix.yaml").read_text())
    online = yaml.safe_load((ROOT / "configs" / "strengthening2-online-continuation.yaml").read_text())
    mnist = old["datasets"]["mnist"]
    runs = []
    terel_rate = chosen["terel"]["probe_learning_rate"]
    for run in mnist["runs"]:
        run = dict(run)
        if run["id"] in chosen:
            run["encoder"] = chosen[run["id"]]["encoder"]
            run["probe"] = {"learning_rate": chosen[run["id"]]["probe_learning_rate"]}
        elif run["id"] in INHERIT_TEREL:
            run["probe"] = {"learning_rate": terel_rate}
        else:
            raise ValueError(f"no selection for {run['id']}")
        runs.append(run)
    continuation = dict(online["datasets"]["mnist"]["runs"][0])
    continuation["probe"] = {"learning_rate": terel_rate}
    runs.append(continuation)
    matrix = {
        "schema_version": 1,
        "evaluation_split": "test",
        "seeds": old["seeds"],
        "selection_seeds": list(SEEDS),
        "selection_policy": (
            "All gradient-trained components use plain SGD. Baseline learning rates and per-method "
            "probe learning rates were selected from strengthening3-validation-ledger.json; the "
            "TeReL configuration and continuation scale are unchanged from strengthening-2."
        ),
        "probe": {"epochs": 60, "batch_size": 1024, "optimizer": "plain_sgd", "learning_rate": terel_rate,
                  "weight_decay": 0.0, "readout": "all", "standardize": True},
        "datasets": {"mnist": {"data_root": str(Path.home() / "main" / "data" / "mnist"),
                               "num_classes": 10, "selected_configuration_id": mnist["selected_configuration_id"],
                               "runs": runs}},
    }
    (ROOT / "configs" / "strengthening3-final-matrix.yaml").write_text(yaml.safe_dump(matrix, sort_keys=False))
    for run_id, entry in chosen.items():
        rate = entry["probe_learning_rate"]
        print(f"{run_id:20s} probe lr {rate:<5g} validation {100 * entry['accuracy_by_probe_learning_rate'][str(rate)]['mean']:.2f}")


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])
