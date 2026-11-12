"""Relative size of the temporal, variance, and covariance contributions to the TeReL learning state.

Short audit run (first N training samples, one epoch) with audit_residual_components on.
Prints the per-layer RMS of each component's contribution to the neuron state.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
import yaml

import terel.resubmission.experiments as experiments
from terel.resubmission.data import DatasetSplits, TemporalTensorDataset


def head(dataset, n):
    return TemporalTensorDataset(features=dataset.features[:n], labels=dataset.labels[:n],
                                 boundaries=dataset.boundaries[:n])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--dataset", choices=("cifar100", "mnist"), required=True)
    parser.add_argument("--samples", type=int, default=8000)
    parser.add_argument("--set", default="{}", help="JSON encoder overrides")
    arguments = parser.parse_args()
    config = yaml.safe_load(Path(arguments.config).read_text())
    if arguments.dataset == "cifar100":
        from terel.resubmission.cifar100 import load_cifar100_protocol
        splits, num_classes = load_cifar100_protocol(config["datasets"]["cifar100"]["data_root"] if "datasets" in config else config["data_root"]), 100
        runs = config["runs"]
    else:
        from terel.resubmission.data import load_mnist_protocol
        mnist = config["datasets"]["mnist"]
        splits, num_classes = load_mnist_protocol(mnist["data_root"]), 10
        runs = mnist["runs"]
    splits = DatasetSplits(train=head(splits.train, arguments.samples),
                           validation=head(splits.validation, 2000), test=head(splits.test, 10),
                           metadata=splits.metadata)
    run = next(r for r in runs if r["id"] == arguments.run)
    values = dict(run["encoder"], epochs=1, audit_residual_components=True, **json.loads(arguments.set))
    for key in ("hidden_dims", "locally_connected"):
        if values.get(key) is not None:
            values[key] = tuple(tuple(v) if isinstance(v, list) else v for v in values[key])
    encoder = experiments.EncoderExperimentConfig(**values)
    probe = experiments.ProbeExperimentConfig(**dict(config["probe"], epochs=5))
    result = experiments.run_representation_experiment(
        splits=splits, dataset_name=arguments.dataset, num_classes=num_classes, seed=401,
        encoder=encoder, probe=probe, evaluation_split="validation", device=torch.device("cuda"))
    training = result["encoder_training"]
    rms = {name: training[name + "_state_rms_mean"] for name in ("temporal", "variance", "covariance")}
    weighted = {"temporal": [values["similarity_coefficient"] * v for v in rms["temporal"]],
                "variance": [values["similarity_coefficient"] * v for v in rms["variance"]],
                "covariance": [values["similarity_coefficient"] * v for v in rms["covariance"]]}
    print(json.dumps({"dataset": arguments.dataset, "run": arguments.run, "overrides": arguments.set,
                      "component_rms_in_update": weighted,
                      "temporal_share": [t / (t + v + c) for t, v, c in zip(*weighted.values())]}, indent=1))


if __name__ == "__main__":
    main()
