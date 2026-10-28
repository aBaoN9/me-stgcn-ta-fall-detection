"""DUC-03: nested LOSO runner for Q0 (A3 attention) vs Q1 (A3 + quality prior), recipe of notebook 07.

fit_model/run loop mirror training/component_ablation/training.py; the only change is the model factory,
so the shared loop is imported for helpers and not edited. Full mode needs an M0-approved config and a GPU.
"""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from torch import nn

from training.component_ablation.features import FEATURE_VERSION, load_inputs
from training.component_ablation.models import make_loader, set_seed
from training.component_ablation.training import default_config, metrics, predict, select_thresholds, write_json
from training.contribution_study.quality_pooling import EPSILON, QUALITY_CHANNEL, QualityAwareSkeletonModel

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = {
    "Q0_A3_attention": {"channels": 8, "attention": True, "fall_multiplier": 1.0, "quality_prior": False},
    "Q1_A3_quality_prior": {"channels": 8, "attention": True, "fall_multiplier": 1.0, "quality_prior": True},
}
CODE_FILES = ["training/contribution_study/runner.py", "training/contribution_study/quality_pooling.py",
              "training/component_ablation/training.py", "training/component_ablation/models.py",
              "training/component_ablation/features.py"]
RATES = ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "f2", "false_alarm_rate"]


def make_config(mode="full", epsilon=EPSILON):
    config = default_config(mode)
    config.update(variants=list(VARIANTS), quality_epsilon=epsilon, quality_channel=QUALITY_CHANNEL,
                  quality_definition="mean_v clip(visibility_v2,0,1); centred avg-pool to attention steps",
                  protocol=config["protocol"] + "; only pooling differs between Q0 and Q1")
    return config


def code_hash():
    digest = hashlib.sha256()
    for name in CODE_FILES:
        digest.update(name.encode() + b"\0" + (ROOT / name).read_bytes())
    return digest.hexdigest()


def build_model(variant, config):
    return QualityAwareSkeletonModel(variant["channels"], variant["attention"], config["dropout"],
                                     variant["quality_prior"], config["quality_epsilon"])


def fit_model(data, train_indices, validation_indices, variant, config, seed, device, fixed_epochs=None):
    set_seed(seed)
    model = build_model(variant, config).to(device)
    train_loader = make_loader(data, train_indices, variant, config, True, seed)
    counts = np.bincount(data["labels"][train_indices], minlength=2)
    if np.any(counts == 0):
        raise ValueError("Training fold has a missing class")
    class_weights = len(train_indices) / (2 * counts.astype(np.float32))
    class_weights[1] *= variant["fall_multiplier"]
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    validation_loader = make_loader(data, validation_indices, variant, config, False, seed) if validation_indices is not None else None
    best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
    history = []
    for epoch in range(1, (fixed_epochs or config["max_epochs"]) + 1):
        model.train()
        total_loss = 0
        for features, target in train_loader:
            features, target = features.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(features)
            loss = criterion(logits, target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(target)
        record = {"epoch": epoch, "train_weighted_loss": total_loss / len(train_indices)}
        if validation_loader is not None:
            labels, scores, _ = predict(model, validation_loader, device)
            scores = np.clip(scores, 1e-7, 1 - 1e-7)
            validation_loss = float(-(labels * np.log(scores) + (1 - labels) * np.log(1 - scores)).mean())
            record["validation_unweighted_loss"] = validation_loss
            if validation_loss < best_loss - 1e-6:
                best_loss, best_epoch, stale = validation_loss, epoch, 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                stale += 1
        history.append(record)
        if validation_loader is not None and stale >= config["patience"]:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch if validation_loader is not None else fixed_epochs, history


def run_neural(data, config, output, device):
    subjects = sorted(np.unique(data["subjects"]).tolist())
    for variant_name in config["variants"]:
        variant = VARIANTS[variant_name]
        for seed in config["seeds"]:
            for outer in subjects:
                name = f"{variant_name}_seed{seed}_subject{outer}"
                target = output / f"{name}.json"
                if target.exists():
                    print("Resume completed:", name, flush=True)
                    continue
                train_indices = np.flatnonzero(data["subjects"] != outer)
                test_indices = np.flatnonzero(data["subjects"] == outer)
                inner_scores, inner_labels, inner_epochs, histories, splits = [], [], [], [], []
                for inner in subjects:
                    if inner == outer:
                        continue
                    fit_indices = np.flatnonzero((data["subjects"] != outer) & (data["subjects"] != inner))
                    val_indices = np.flatnonzero(data["subjects"] == inner)
                    assert not set(fit_indices) & set(val_indices) and not set(train_indices) & set(test_indices)
                    run_seed = seed + outer * 1000 + inner * 10
                    model, best_epoch, history = fit_model(data, fit_indices, val_indices, variant, config, run_seed, device)
                    labels, scores, _ = predict(model, make_loader(data, val_indices, variant, config, False, run_seed), device)
                    inner_scores.extend(scores.tolist())
                    inner_labels.extend(labels.tolist())
                    inner_epochs.append(best_epoch)
                    histories.extend([{**row, "inner_subject": inner} for row in history])
                    splits.append({"inner_subject": inner, "train_indices": fit_indices.tolist(), "val_indices": val_indices.tolist(), "best_epoch": best_epoch})
                    del model
                choices = select_thresholds(np.asarray(inner_labels), np.asarray(inner_scores), config)
                epochs = max(1, int(np.median(inner_epochs)))
                model, _, refit_history = fit_model(data, train_indices, None, variant, config, seed + outer * 1000 + 999, device, epochs)
                labels, scores, attention = predict(model, make_loader(data, test_indices, variant, config, False, seed), device)
                if not np.isfinite(scores).all() or not np.isfinite(attention).all():
                    raise FloatingPointError(f"Non-finite outer output in {name}")
                predictions, fold_metrics = [], []
                for policy, choice in choices.items():
                    threshold = choice["threshold"]
                    predicted = (scores >= threshold).astype(int)
                    fold_metrics.append({"variant": variant_name, "seed": seed, "subject": outer, "policy": policy,
                                         "threshold": threshold, "epochs": epochs, "inner_target_met": choice["feasible"], **metrics(labels, predicted)})
                    for index, label, score, prediction in zip(test_indices, labels, scores, predicted):
                        predictions.append({"sample_index": int(index), "video_id": str(data["video_ids"][index]), "subject": outer,
                            "label": int(label), "score": float(score), "prediction": int(prediction), "threshold": threshold,
                            "variant": variant_name, "seed": seed, "policy": policy})
                checkpoint = {"model_state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
                              "variant": variant, "feature_version": FEATURE_VERSION, "train_indices": train_indices.tolist(),
                              "test_indices": test_indices.tolist(), "config": config, "epochs": epochs}
                torch.save(checkpoint, output / f"{name}.pt")
                np.savez_compressed(output / f"{name}_attention.npz", indices=test_indices, attention=attention)
                pd.DataFrame(histories).to_csv(output / f"{name}_inner_history.csv", index=False)
                pd.DataFrame(refit_history).to_csv(output / f"{name}_refit_history.csv", index=False)
                write_json(target, {"complete": True, "inner_splits": splits, "inner_labels": inner_labels, "inner_scores": inner_scores,
                                    "choices": choices, "metrics": fold_metrics, "predictions": predictions, "fits": len(splits) + 1,
                                    "parameters": sum(parameter.numel() for parameter in model.parameters())})
                print("Completed", name, "epochs", epochs, flush=True)
                del model


def summarize_run(data, config, output):
    paths = sorted(output.glob("Q*_seed*_subject*.json"))
    subjects = len(np.unique(data["subjects"]))
    expected = len(config["variants"]) * len(config["seeds"]) * subjects
    if len(paths) != expected:
        raise ValueError(f"Only {len(paths)}/{expected} complete outer runs; no final summary")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    predictions = pd.DataFrame([row for report in reports for row in report["predictions"]])
    folds = pd.DataFrame([row for report in reports for row in report["metrics"]])
    count, falls = len(data["labels"]), int(data["labels"].sum())
    if predictions.duplicated(["variant", "seed", "policy", "sample_index"]).any():
        raise ValueError("Duplicate OOF rows")
    pooled = []
    for (variant, seed, policy), group in predictions.groupby(["variant", "seed", "policy"]):
        if len(group) != count or group["sample_index"].nunique() != count or int(group["label"].sum()) != falls:
            raise ValueError("OOF coverage failed; do not pool duplicates across seeds")
        pooled.append({"variant": variant, "seed": int(seed), "policy": policy, **metrics(group["label"], group["prediction"])})
    pooled = pd.DataFrame(pooled)
    summary = pooled.groupby(["variant", "policy"])[RATES + ["fn", "fp"]].agg(["mean", lambda x: x.std(ddof=1)])
    summary.columns = [f"{column}_{'mean' if stat == 'mean' else 'std'}" for column, stat in summary.columns]
    for name, frame in [("predictions", predictions), ("fold_metrics", folds), ("pooled_by_seed", pooled),
                        ("errors", predictions[predictions["label"] != predictions["prediction"]])]:
        frame.to_csv(output / f"{name}.csv", index=False)
    summary.reset_index().to_csv(output / "seed_mean_std.csv", index=False)
    fits = sum(report["fits"] for report in reports)
    write_json(output / "completion.json", {
        "complete": True, "mode": config["mode"], "scientific_result": config["mode"] == "full" and not data.get("synthetic", False),
        "variants": config["variants"], "seeds": config["seeds"], "outer_runs_expected": expected, "outer_runs": len(paths),
        "fits_expected": expected * subjects, "fits": fits,
        "warning": "Smoke/synthetic runs are software checks only. Seeds repeat the same clips. Prior outer results informed development."})
    return pooled


def run_experiment(data, config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and config["mode"] == "full" and not config["allow_cpu_full"]:
        raise RuntimeError("Full experiment requires a GPU; choose smoke locally, or explicitly allow CPU full")
    signature = {"config": config, "inputs": {key: value["sha256"] for key, value in data["provenance"].items()}, "code_hash": code_hash(),
                 "versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__},
                 "device": device}
    lock = output / "locked_protocol.json"
    if lock.exists() and json.loads(lock.read_text(encoding="utf-8")) != signature:
        raise ValueError("Run configuration/code/data/environment changed; use a NEW output directory")
    write_json(lock, signature)
    write_json(output / "input_provenance.json", data["provenance"])
    started = time.monotonic()
    run_neural(data, config, output, device)
    summary = summarize_run(data, config, output)
    write_json(output / "timing.json", {"seconds_this_invocation": time.monotonic() - started, "device": device})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=ROOT / "data/local")
    parser.add_argument("--output", type=Path, required=True, help="new directory per run")
    parser.add_argument("--mode", choices=["full", "smoke"], default="smoke")
    args = parser.parse_args()
    print(run_experiment(load_inputs(args.input_root), make_config(args.mode), args.output).to_string())
