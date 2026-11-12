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


LAYOUTS = {
    "mnist": ((512, 256), None),
    "lc1632": ((4096, 2048), ((16, 5, 2, 2), (32, 5, 2, 2))),
    "lcfc": ((4096, 2048, 1024), ((16, 5, 2, 2), (32, 5, 2, 2), None)),
    "lc3264": ((8192, 4096), ((32, 5, 2, 2), (64, 5, 2, 2))),
    "lc3264fc": ((8192, 4096, 1024), ((32, 5, 2, 2), (64, 5, 2, 2), None)),
    "lc3": ((8192, 4096, 2048), ((32, 5, 2, 2), (64, 5, 2, 2), (128, 3, 2, 1))),
}


def bp_sgd_configuration(trial, layout):
    hidden, local = LAYOUTS[layout]
    return EncoderExperimentConfig(
        method="bp", hidden_dims=hidden, locally_connected=local, activation="relu",
        normalization="none", optimizer="plain_sgd",
        epochs=trial.suggest_categorical("epochs", (10, 20, 40)),
        batch_size=trial.suggest_categorical("batch_size", (16, 32, 64, 128, 256)),
        learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        weight_decay=trial.suggest_categorical("weight_decay", (0.0, 1e-5, 1e-4, 5e-4)),
    )


def offline_sgd_configuration(trial, layout):
    hidden, local = LAYOUTS[layout]
    return EncoderExperimentConfig(
        method="terel_offline", hidden_dims=hidden, locally_connected=local, activation="relu",
        normalization="none", order_mode="class_chunks", chunk_size=16, optimizer="plain_sgd",
        epochs=trial.suggest_categorical("epochs", (5, 10, 20)),
        batch_size=trial.suggest_categorical("subsequence_size", (32, 64, 128, 256)),
        learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        weight_decay=trial.suggest_categorical("weight_decay", (0.0, 1e-6, 1e-5, 1e-4)),
        similarity_coefficient=1.0,
        variance_coefficient=trial.suggest_float("variance_coefficient", 0.1, 5.0, log=True),
        covariance_coefficient=trial.suggest_float("covariance_coefficient", 0.001, 0.5, log=True),
        variance_target=trial.suggest_float("variance_target", 0.5, 1.5),
    )


def terel_wide_configuration(trial, layout):
    """TeReL-specific search: normalization, stream, lateral dynamics, and momenta."""
    hidden, local = LAYOUTS[layout]
    values = dict(BASE["runs"][0]["encoder"])
    normalization = trial.suggest_categorical("normalization", ("none", "streaming_norm"))
    values.update(
        hidden_dims=hidden, locally_connected=local,
        normalization=normalization, normalization_affine=False,
        normalization_momentum=0.99,
        epochs=trial.suggest_categorical("epochs", (1, 2, 4)),
        chunk_size=trial.suggest_categorical("chunk_size", (8, 16, 32, 64)),
        learning_rate=trial.suggest_float("learning_rate", 3e-4, 2e-2, log=True),
        weight_decay=trial.suggest_categorical("weight_decay", (0.0, 1e-6, 1e-5)),
        variance_coefficient=trial.suggest_float("variance_coefficient", 0.3, 5.0, log=True),
        covariance_coefficient=trial.suggest_float("covariance_coefficient", 0.001, 0.2, log=True),
        variance_target=trial.suggest_float("variance_target", 0.2, 1.2, log=True),
        statistics_momentum=trial.suggest_categorical("statistics_momentum", (0.99, 0.995, 0.999)),
        lateral_momentum=trial.suggest_categorical("lateral_momentum", (0.99, 0.995, 0.999)),
        residual_lateral_coefficient=trial.suggest_float("lateral_coefficient", 3e-4, 0.3, log=True),
        residual_lateral_steps=trial.suggest_categorical("lateral_steps", (1, 2)),
        combined_lateral_state_weight=trial.suggest_float("combined_lateral_state_weight", 0.3, 0.95),
    )
    return EncoderExperimentConfig(**values)


for _layout in LAYOUTS:
    FAMILIES[f"bp-sgd-{_layout}"] = lambda trial, l=_layout: bp_sgd_configuration(trial, l)
    FAMILIES[f"offline-sgd-{_layout}"] = lambda trial, l=_layout: offline_sgd_configuration(trial, l)
    FAMILIES[f"terel-{_layout}"] = lambda trial, l=_layout: terel_wide_configuration(trial, l)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage", required=True)
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--family", default="terel", choices=sorted(FAMILIES))
    parser.add_argument("--study", default="terel-cifar100-lc")
    parser.add_argument("--probe", help="JSON overrides of the probe configuration")
    parser.add_argument("--dataset", default="cifar100", choices=("cifar100", "mnist"))
    parser.add_argument("--seed", type=int, default=401)
    arguments = parser.parse_args()
    if arguments.dataset == "mnist":
        from terel.resubmission.data import load_mnist_protocol
        splits, num_classes = load_mnist_protocol("/home/davwis/main/data/mnist"), 10
    else:
        splits, num_classes = load_cifar100_protocol(BASE["data_root"]), 100
    probe = ProbeExperimentConfig(**(BASE["probe"] | json.loads(arguments.probe or "{}")))
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)

    def objective(trial):
        encoder = FAMILIES[arguments.family](trial)
        start = time.time()
        try:
            result = run_representation_experiment(
                splits=splits, dataset_name=arguments.dataset, num_classes=num_classes,
                seed=arguments.seed,
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
             "probe": probe.__dict__, "diagnostics": result["representation_diagnostics"]}, indent=1, default=str))
        return accuracy if accuracy == accuracy else 0.0

    optuna.load_study(study_name=arguments.study, storage=arguments.storage).optimize(
        objective, n_trials=arguments.trials)


if __name__ == "__main__":
    main()
