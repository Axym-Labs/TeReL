"""CIFAR-100 protocol and a validation/held-out driver reusing the MNIST runner."""

import argparse
import json
import pickle
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml

from .data import DatasetSplits, TemporalTensorDataset, _single_stream_boundaries
from .data import stratified_split_indices
from .experiments import (
    EncoderExperimentConfig,
    ProbeExperimentConfig,
    TestGateContext,
    run_representation_experiment,
)
from .provenance import build_run_manifest


def load_cifar100_protocol(root, *, validation_size: int = 5_000, seed: int = 1701, preprocessing: str = "standardize", zca_epsilon: float = 0.1):
    """Flattened CIFAR-100 (fine labels); validation drawn from the official train split."""
    root = Path(root)

    def read(name):
        with open(root / name, "rb") as handle:
            raw = pickle.load(handle, encoding="latin1")
        images = torch.as_tensor(raw["data"], dtype=torch.float32) / 255.0
        return images, torch.as_tensor(raw["fine_labels"], dtype=torch.long)

    train_images, train_labels = read("train")
    test_images, test_labels = read("test")
    train_indices, validation_indices = stratified_split_indices(
        train_labels.numpy(), validation_size=validation_size, seed=seed
    )
    index = torch.as_tensor(train_indices, dtype=torch.long)
    # Per-channel statistics of the encoder-training images only (3 x 1024 layout).
    channels = train_images[index].reshape(len(index), 3, 1024)
    mean = channels.mean(dim=(0, 2)).repeat_interleave(1024)
    std = channels.std(dim=(0, 2)).repeat_interleave(1024)

    transform = lambda images: (images - mean) / std
    if preprocessing == "zca":
        # ZCA whitening fitted on the encoder-training images only.
        fitted = transform(train_images[index]).double()
        center = fitted.mean(dim=0)
        covariance = torch.cov((fitted - center).T)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        scale = eigenvectors @ torch.diag((eigenvalues.clamp_min(0) + zca_epsilon).rsqrt()) @ eigenvectors.T
        whitening, center = scale.float(), center.float()
        transform = lambda images: ((images - mean) / std - center) @ whitening
    elif preprocessing != "standardize":
        raise ValueError(f"unknown preprocessing {preprocessing}")

    def dataset(images, labels):
        return TemporalTensorDataset(
            features=transform(images),
            labels=labels,
            boundaries=_single_stream_boundaries(len(labels)),
        )

    validation_index = torch.as_tensor(validation_indices, dtype=torch.long)
    return DatasetSplits(
        train=dataset(train_images[index], train_labels[index]),
        validation=dataset(train_images[validation_index], train_labels[validation_index]),
        test=dataset(test_images, test_labels),
        metadata={"dataset": "CIFAR-100", "split_seed": seed, "validation_source": "official_train", "preprocessing": preprocessing},
    )


def _summary(result):
    keep = {}
    metrics = result.get("metrics") or result.get("evaluation", {}).get("metrics")
    keep["accuracy"] = (metrics or {}).get("accuracy")
    paired = result.get("paired_inference")
    if paired:
        keep["online_accuracy"] = paired["online"]["metrics"]["accuracy"]
    return keep


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--runs", nargs="*", help="subset of run ids")
    parser.add_argument("--validation-ledger", help="selection record required for held-out runs")
    parser.add_argument("--freeze", action="store_true", help="write the held-out manifest and exit")
    parser.add_argument("--allow-test", action="store_true")
    arguments = parser.parse_args(argv)
    raw = yaml.safe_load(Path(arguments.config).read_text())
    repository = Path(__file__).resolve().parents[2]
    manifest_path = Path(arguments.output) / "manifest.json"
    if arguments.freeze:
        manifest = build_run_manifest(
            phase="confirmatory", frozen=True, selection_complete=True,
            protocol_path=arguments.config, validation_ledger_path=arguments.validation_ledger,
            repository=repository, configuration=raw,
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
        print("froze", manifest["code_commit"])
        return
    gate = None
    if raw["evaluation_split"] == "test":
        gate = TestGateContext(
            manifest=json.loads(manifest_path.read_text()),
            protocol_path=arguments.config,
            validation_ledger_path=arguments.validation_ledger,
            repository=str(repository),
            explicit_allow_test=arguments.allow_test,
        )
    splits = load_cifar100_protocol(raw["data_root"], preprocessing=raw.get("preprocessing", "standardize"))
    probe = ProbeExperimentConfig(**raw["probe"])
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for run in raw["runs"]:
        if arguments.runs and run["id"] not in arguments.runs:
            continue
        encoder_raw = dict(run["encoder"])
        encoder_raw["hidden_dims"] = tuple(encoder_raw["hidden_dims"])
        if encoder_raw.get("locally_connected"):
            encoder_raw["locally_connected"] = tuple(None if x is None else tuple(x) for x in encoder_raw["locally_connected"])
        encoder = EncoderExperimentConfig(**encoder_raw)
        for seed in raw["seeds"]:
            path = output / run["id"] / f"seed-{seed}.json"
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            start = time.time()
            result = run_representation_experiment(
                splits=splits,
                dataset_name="cifar100",
                num_classes=100,
                seed=seed,
                encoder=encoder,
                probe=probe,
                evaluation_split=raw["evaluation_split"],
                device=device,
                test_gate=gate,
            )
            result["run_id"] = run["id"]
            result["wall_seconds"] = time.time() - start
            path.write_text(json.dumps(result, indent=1, sort_keys=True, default=str) + "\n")
            print(run["id"], seed, _summary(result), f"{time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
