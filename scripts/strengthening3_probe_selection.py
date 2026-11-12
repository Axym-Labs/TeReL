"""Validation-only probe selection for the MNIST comparison.

Trains an encoder once and fits plain-SGD linear heads with several learning rates on its
standardized features. The records in validation/strengthening3/ were produced with
  --config validation/strengthening3/selection.yaml --output validation/strengthening3
  --grid '{"learning_rate":[0.01,0.03,0.1,0.3,1.0],"batch_size":[1024],"weight_decay":[0.0]}'
for seeds 401, 402, and 403.
"""

import argparse
import itertools
import json
from pathlib import Path

import torch
import yaml

import terel.resubmission.experiments as experiments
from terel.resubmission.data import load_mnist_protocol
from terel.resubmission.evaluation import classification_metrics, fit_linear_probe

GRID = {
    "learning_rate": (0.01, 0.03, 0.1, 0.3),
    "batch_size": (64, 256, 1024),
    "weight_decay": (0.0, 1e-4),
}


def run_with_grid(*, splits, dataset_name, num_classes, seed, encoder, probe, device, grid=GRID,
                  epochs=60):
    views = []
    original = experiments._probe_view

    def recording_view(representations, dataset):
        out = original(representations, dataset)
        views.append(out)
        return out

    experiments._probe_view = recording_view
    try:
        result = experiments.run_representation_experiment(
            splits=splits, dataset_name=dataset_name, num_classes=num_classes, seed=seed,
            encoder=encoder, probe=probe, evaluation_split="validation", device=device)
    finally:
        experiments._probe_view = original
    (train_x, train_y), (eval_x, eval_y) = views[0], views[1]
    rows = []
    for lr, bs, wd in itertools.product(grid["learning_rate"], grid["batch_size"], grid["weight_decay"]):
        head, summary = fit_linear_probe(
            train_x, train_y, num_classes=num_classes, seed=seed + 10_000, epochs=epochs,
            batch_size=bs, optimizer_name="plain_sgd", learning_rate=lr, weight_decay=wd,
            device=device, standardize=probe.standardize)
        with torch.no_grad():
            logits = head(eval_x.to(device)).cpu()
        acc = classification_metrics(logits, eval_y, num_classes=num_classes)["accuracy"]
        rows.append({"learning_rate": lr, "batch_size": bs, "weight_decay": wd, "epochs": epochs,
                     "accuracy": acc, "final_loss": summary.final_loss})
    return result, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--runs", nargs="*")
    parser.add_argument("--output", required=True)
    parser.add_argument("--grid", help="JSON grid overriding GRID")
    parser.add_argument("--seed", type=int, default=401)
    arguments = parser.parse_args()
    config = yaml.safe_load(Path(arguments.config).read_text())
    splits, num_classes = load_mnist_protocol(config["data_root"]), 10
    probe = experiments.ProbeExperimentConfig(**config["probe"])
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    for run in config["runs"]:
        if arguments.runs and run["id"] not in arguments.runs:
            continue
        values = dict(run["encoder"])
        if values.get("hidden_dims") is not None:
            values["hidden_dims"] = tuple(values["hidden_dims"])
        encoder = experiments.EncoderExperimentConfig(**values)
        result, rows = run_with_grid(
            splits=splits, dataset_name="mnist", num_classes=num_classes, seed=arguments.seed,
            encoder=encoder, probe=probe, device=torch.device("cuda"),
            grid=json.loads(arguments.grid) if arguments.grid else GRID)
        best = max(rows, key=lambda r: r["accuracy"])
        print(run["id"], "reference", result["metrics"]["accuracy"], "best sgd", best, flush=True)
        (output / f"{run['id']}-s{arguments.seed}.json").write_text(json.dumps(
            {"reference_probe": config["probe"], "reference_accuracy": result["metrics"]["accuracy"],
             "encoder": run["encoder"], "diagnostics": result["representation_diagnostics"],
             "paired_inference": result.get("paired_inference"), "grid": rows}, indent=1, default=str))


if __name__ == "__main__":
    main()
