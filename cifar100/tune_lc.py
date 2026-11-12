"""Validation-only Optuna search for TeReL with locally connected layers on CIFAR-100."""

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
    values.update(
        hidden_dims=(4096, 2048),
        locally_connected=((16, 5, 2, 2), (32, 5, 2, 2)),
        epochs=2,
        learning_rate=trial.suggest_float("learning_rate", 3e-4, 1e-2, log=True),
        weight_decay=trial.suggest_categorical("weight_decay", (0.0, 1e-6, 1e-5, 1e-4)),
        variance_coefficient=trial.suggest_float("variance_coefficient", 1.0, 10.0, log=True),
        covariance_coefficient=trial.suggest_float("covariance_coefficient", 0.003, 0.3, log=True),
        variance_target=trial.suggest_float("variance_target", 0.3, 1.2, log=True),
        statistics_momentum=trial.suggest_categorical("statistics_momentum", (0.99, 0.995)),
        lateral_momentum=trial.suggest_categorical("lateral_momentum", (0.99, 0.995)),
        residual_lateral_coefficient=trial.suggest_float("lateral_coefficient", 0.001, 1.0, log=True),
        combined_lateral_state_weight=trial.suggest_float("combined_lateral_state_weight", 0.3, 0.95),
        chunk_size=trial.suggest_categorical("chunk_size", (8, 16)),
    )
    return EncoderExperimentConfig(**values)


def offline_configuration(trial):
    return EncoderExperimentConfig(
        method="terel_offline",
        hidden_dims=(4096, 2048),
        locally_connected=((16, 5, 2, 2), (32, 5, 2, 2)),
        activation="relu",
        normalization="none",
        epochs=trial.suggest_categorical("epochs", (5, 10, 20)),
        batch_size=trial.suggest_categorical("subsequence_size", (64, 128, 256)),
        order_mode="class_chunks",
        chunk_size=16,
        optimizer="sgd",
        optimizer_momentum=trial.suggest_categorical("optimizer_momentum", (0.9, 0.95, 0.99)),
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 3e-2, log=True),
        weight_decay=trial.suggest_categorical("weight_decay", (0.0, 1e-6, 1e-5, 1e-4)),
        similarity_coefficient=1.0,
        variance_coefficient=trial.suggest_float("variance_coefficient", 0.1, 5.0, log=True),
        covariance_coefficient=trial.suggest_float("covariance_coefficient", 0.001, 0.5, log=True),
        variance_target=trial.suggest_float("variance_target", 0.5, 1.5),
    )


def snorm_configuration(trial):
    values = dict(BASE["runs"][0]["encoder"])
    values.update(
        hidden_dims=(4096, 2048),
        locally_connected=((16, 5, 2, 2), (32, 5, 2, 2)),
        normalization="streaming_norm",
        normalization_affine=False,
        normalization_momentum=trial.suggest_categorical("normalization_momentum", (0.99, 0.999)),
        epochs=trial.suggest_categorical("epochs", (2, 4)),
        learning_rate=trial.suggest_float("learning_rate", 3e-4, 2e-2, log=True),
        weight_decay=trial.suggest_categorical("weight_decay", (0.0, 1e-6, 1e-5, 1e-4)),
        variance_coefficient=trial.suggest_float("variance_coefficient", 0.5, 10.0, log=True),
        covariance_coefficient=trial.suggest_float("covariance_coefficient", 0.002, 0.2, log=True),
        variance_target=trial.suggest_float("variance_target", 0.3, 1.5, log=True),
        statistics_momentum=trial.suggest_categorical("statistics_momentum", (0.99, 0.995)),
        lateral_momentum=trial.suggest_categorical("lateral_momentum", (0.99, 0.995)),
        residual_lateral_coefficient=trial.suggest_float("lateral_coefficient", 0.001, 0.3, log=True),
        combined_lateral_state_weight=trial.suggest_float("combined_lateral_state_weight", 0.3, 0.95),
        chunk_size=trial.suggest_categorical("chunk_size", (8, 16, 32)),
    )
    return EncoderExperimentConfig(**values)


def snorm_lcfc_configuration(trial):
    values = snorm_configuration(trial).__dict__ | {
        "hidden_dims": (4096, 2048, 1024),
        "locally_connected": ((16, 5, 2, 2), (32, 5, 2, 2), None),
    }
    return EncoderExperimentConfig(**values)


FAMILIES = {"terel": configuration, "offline": offline_configuration, "snorm": snorm_configuration,
            "snorm-lcfc": snorm_lcfc_configuration}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage", required=True)
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--family", default="terel", choices=sorted(FAMILIES))
    parser.add_argument("--study", default="terel-cifar100-lc")
    arguments = parser.parse_args()
    splits = load_cifar100_protocol(BASE["data_root"])
    probe = ProbeExperimentConfig(**BASE["probe"])
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)

    def objective(trial):
        encoder = FAMILIES[arguments.family](trial)
        start = time.time()
        try:
            result = run_representation_experiment(
                splits=splits, dataset_name="cifar100", num_classes=100, seed=401,
                encoder=encoder, probe=probe, evaluation_split="validation",
                device=torch.device("cuda"),
            )
        except (RuntimeError, ValueError, FloatingPointError) as error:
            (output / f"trial-{trial.number:03d}.json").write_text(
                json.dumps({"params": trial.params, "error": str(error)}))
            return 0.0
        accuracy = result["metrics"]["accuracy"]
        (output / f"trial-{trial.number:03d}.json").write_text(json.dumps(
            {"params": trial.params, "accuracy": accuracy, "seconds": time.time() - start,
             "diagnostics": result["representation_diagnostics"]}, indent=1, default=str))
        return accuracy if accuracy == accuracy else 0.0

    optuna.load_study(study_name=arguments.study, storage=arguments.storage).optimize(
        objective, n_trials=arguments.trials)


if __name__ == "__main__":
    main()
