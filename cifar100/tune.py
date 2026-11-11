"""Validation-only Optuna search for TeReL on CIFAR-100 (never reads the test split)."""

import argparse
import json
import time
from pathlib import Path

import optuna
import torch
import yaml

from terel.resubmission.cifar100 import load_cifar100_protocol
from terel.resubmission.experiments import (
    EncoderExperimentConfig,
    ProbeExperimentConfig,
    run_representation_experiment,
)

HERE = Path(__file__).parent
BASE = yaml.safe_load((HERE / "screen.yaml").read_text())


def configuration(trial):
    values = dict(BASE["runs"][0]["encoder"])
    values["hidden_dims"] = tuple(values["hidden_dims"])
    values.update(
        epochs=trial.suggest_categorical("epochs", (2, 4)),
        learning_rate=trial.suggest_float("learning_rate", 1e-3, 5e-2, log=True),
        weight_decay=trial.suggest_categorical("weight_decay", (0.0, 1e-6, 1e-5, 1e-4)),
        variance_coefficient=trial.suggest_float("variance_coefficient", 0.5, 10.0, log=True),
        covariance_coefficient=trial.suggest_float("covariance_coefficient", 0.05, 3.0, log=True),
        variance_target=trial.suggest_float("variance_target", 0.25, 2.0, log=True),
        statistics_momentum=trial.suggest_categorical("statistics_momentum", (0.95, 0.99, 0.995)),
        lateral_momentum=trial.suggest_categorical("lateral_momentum", (0.95, 0.99, 0.995)),
        residual_lateral_coefficient=trial.suggest_float("lateral_coefficient", 0.01, 100.0, log=True),
        combined_lateral_state_weight=trial.suggest_float("combined_lateral_state_weight", 0.1, 0.9),
        chunk_size=trial.suggest_categorical("chunk_size", (8, 16, 32)),
    )
    return EncoderExperimentConfig(**values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage", required=True)
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    splits = load_cifar100_protocol(BASE["data_root"])
    probe = ProbeExperimentConfig(**BASE["probe"])
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    def objective(trial):
        encoder = configuration(trial)
        start = time.time()
        try:
            result = run_representation_experiment(
                splits=splits, dataset_name="cifar100", num_classes=100, seed=401,
                encoder=encoder, probe=probe, evaluation_split="validation", device=device,
            )
        except (RuntimeError, ValueError, FloatingPointError) as error:
            (output / f"trial-{trial.number:03d}.json").write_text(json.dumps({"params": trial.params, "error": str(error)}))
            return 0.0
        accuracy = result["metrics"]["accuracy"]
        record = {"params": trial.params, "accuracy": accuracy, "seconds": time.time() - start,
                  "diagnostics": result["representation_diagnostics"]}
        (output / f"trial-{trial.number:03d}.json").write_text(json.dumps(record, indent=1, default=str))
        return accuracy if accuracy == accuracy else 0.0

    study = optuna.load_study(study_name="terel-cifar100", storage=arguments.storage)
    study.optimize(objective, n_trials=arguments.trials)


if __name__ == "__main__":
    main()
