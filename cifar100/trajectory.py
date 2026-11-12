"""Short TeReL run that records what each layer does during training.

Every --every steps: mean running feature variance, share of features whose variance is below
the target (variance gate open), weight norm, lateral-matrix spectral norm, and the RMS of the
temporal / variance / covariance contributions to the update. Ends with the usual validation probe.
"""

import argparse
import json
from pathlib import Path

import torch
import yaml

import terel.resubmission.experiments as experiments
import terel.resubmission.training as training
from terel.resubmission.data import DatasetSplits, TemporalTensorDataset


def head(dataset, n):
    return TemporalTensorDataset(features=dataset.features[:n], labels=dataset.labels[:n],
                                 boundaries=dataset.boundaries[:n])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="round4.yaml")
    parser.add_argument("--run", default="terel-lcfc-best")
    parser.add_argument("--dataset", choices=("cifar100", "mnist"), default="cifar100")
    parser.add_argument("--samples", type=int, default=12000)
    parser.add_argument("--every", type=int, default=1000)
    parser.add_argument("--set", default="{}")
    parser.add_argument("--output", required=True)
    parser.add_argument("--readout", default="all", choices=("all", "last"))
    arguments = parser.parse_args()
    config = yaml.safe_load(Path(arguments.config).read_text())
    if arguments.dataset == "cifar100":
        from terel.resubmission.cifar100 import load_cifar100_protocol
        splits, num_classes, runs = load_cifar100_protocol(config["data_root"]), 100, config["runs"]
    else:
        from terel.resubmission.data import load_mnist_protocol
        mnist = config["datasets"]["mnist"]
        splits, num_classes, runs = load_mnist_protocol(mnist["data_root"]), 10, mnist["runs"]
    splits = DatasetSplits(train=head(splits.train, arguments.samples), validation=splits.validation,
                           test=head(splits.test, 10), metadata=splits.metadata)
    run = next(r for r in runs if r["id"] == arguments.run)
    values = dict(run["encoder"], epochs=1, audit_residual_components=True, **json.loads(arguments.set))
    for key in ("hidden_dims", "locally_connected"):
        if values.get(key) is not None:
            values[key] = tuple(tuple(v) if isinstance(v, list) else v for v in values[key])
    encoder = experiments.EncoderExperimentConfig(**values)
    probe = experiments.ProbeExperimentConfig(**dict(config["probe"], learning_rate=0.01, standardize=True, readout=arguments.readout))

    rows, step = [], [0]
    original = training.local_train_step

    def recording_step(**kwargs):
        try:
            metrics = original(**kwargs)
        except FloatingPointError:
            rows.append({"step": step[0], "diverged": True})
            raise
        step[0] += 1
        if step[0] % arguments.every == 0 or step[0] in (1, 100, 300):
            model = kwargs["model"]
            row = {"step": step[0], "loss": metrics["loss"]}
            for index, (state, layer) in enumerate(zip(model.states, model.layers)):
                lateral = state.lateral
                vector = torch.ones(lateral.shape[0], device=lateral.device)
                for _ in range(8):  # power iteration for the lateral spectral norm
                    vector = lateral @ vector
                    norm = vector.norm()
                    vector = vector / norm.clamp_min(1e-12)
                row[f"L{index}"] = {
                    "variance_mean": float(state.variance.mean()),
                    "gate_open": float((state.variance < kwargs["variance_target"]).float().mean()),
                    "weight_norm": float(layer.weight.norm()),
                    "lateral_spectral": float(norm),
                    **{name: metrics.get(f"layer_{index}/{name}_state_rms") for name in ("temporal", "variance", "covariance")},
                }
            rows.append(row)
        return metrics

    training.local_train_step = recording_step
    out = {"overrides": json.loads(arguments.set), "samples": arguments.samples}
    try:
        result = experiments.run_representation_experiment(
            splits=splits, dataset_name=arguments.dataset, num_classes=num_classes, seed=401,
            encoder=encoder, probe=probe, evaluation_split="validation", device=torch.device("cuda"))
        out["accuracy"] = result["metrics"]["accuracy"]
        out["diagnostics"] = result["representation_diagnostics"]
    except FloatingPointError:
        out["accuracy"] = None
    out["trajectory"] = rows
    Path(arguments.output).write_text(json.dumps(out, indent=1))
    print("accuracy", out["accuracy"])
    for row in rows:
        if row.get("diverged"):
            print(row)
            continue
        print(row["step"], f"loss {row['loss']:.3g}", " | ".join(
            f"var {v['variance_mean']:.3g} gate {v['gate_open']:.2f} |W| {v['weight_norm']:.3g} lat {v['lateral_spectral']:.3g} "
            f"t/v/c {v['temporal'] or 0:.3g}/{v['variance'] or 0:.3g}/{v['covariance'] or 0:.3g}"
            for k, v in row.items() if k.startswith("L")))


if __name__ == "__main__":
    main()
