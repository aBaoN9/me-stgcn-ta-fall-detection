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
LOCKED_INPUTS = ROOT / "results/component_ablation/locked_protocol.json"
POLICIES = ("fixed", "balanced", "sensitivity")
TOLERANCE = 1e-9
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


def verify_inputs(data, config, locked_path=LOCKED_INPUTS):
    """Stop before any fit unless the four inputs match the approved 07 hashes (DUC-REV-01).

    Synthetic data takes a separate, labelled path that is only allowed in smoke mode and never yields
    scientific_result=true; it cannot be used to bypass the guard for real inputs.
    """
    if data.get("synthetic", False):
        if config["mode"] != "smoke":
            raise ValueError("Synthetic data is only allowed in smoke mode")
        return "synthetic"
    expected = json.loads(Path(locked_path).read_text(encoding="utf-8"))["inputs"]
    provenance = data.get("provenance", {})
    problems = [f"{key}: missing input" if key not in provenance else
                f"{key}: sha256 {provenance[key].get('sha256')} != approved {value}"
                for key, value in expected.items() if provenance.get(key, {}).get("sha256") != value]
    problems += [f"{key}: unexpected input" for key in sorted(set(provenance) - set(expected))]
    if problems:
        raise ValueError("Input guard failed; do not train on unapproved data: " + "; ".join(problems))
    return "locked"


def signature_hash(signature):
    return hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()


def same_value(left, right):
    if left is None or right is None:
        return left is None and right is None
    return np.isfinite(left) and np.isfinite(right) and abs(float(left) - float(right)) <= TOLERANCE


def validate_fold(data, config, output, variant_name, seed, outer, signature_sha):
    """Return the problems of one outer fold; empty means its report and artifacts can be trusted (DUC-REV-02)."""
    name = f"{variant_name}_seed{seed}_subject{outer}"
    try:
        report = json.loads((output / f"{name}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return [f"{name}: unreadable report ({error})"]
    subjects = sorted(np.unique(data["subjects"]).tolist())
    inner_expected = [subject for subject in subjects if subject != outer]
    test_indices = np.flatnonzero(data["subjects"] == outer)
    problems = []
    if report.get("complete") is not True:
        problems.append("complete is not true")
    if report.get("signature_sha256") != signature_sha:
        problems.append("report signature differs from this run")
    splits = report.get("inner_splits", [])
    if sorted(split.get("inner_subject") for split in splits) != inner_expected:
        problems.append(f"inner subjects {[split.get('inner_subject') for split in splits]} != {inner_expected}")
    if report.get("fits") != len(inner_expected) + 1:
        problems.append(f"fits {report.get('fits')} != {len(inner_expected) + 1}")
    choices = report.get("choices", {})
    if set(choices) != set(POLICIES):
        problems.append(f"policies {sorted(choices)} != {sorted(POLICIES)}")
    fold_metrics = {row.get("policy"): row for row in report.get("metrics", [])}
    if len(report.get("metrics", [])) != len(POLICIES) or set(fold_metrics) != set(POLICIES):
        problems.append("fold metrics do not cover each policy exactly once")
    predictions = pd.DataFrame(report.get("predictions", []))
    if predictions.empty:
        return [f"{name}: {problem}" for problem in problems + ["no predictions"]]
    for policy in POLICIES:
        rows = predictions[predictions["policy"] == policy].sort_values("sample_index")
        if len(rows) != len(test_indices) or not np.array_equal(rows["sample_index"].to_numpy(), test_indices):
            problems.append(f"{policy}: sample indices do not match subject {outer}")
            continue
        if ((rows["variant"] != variant_name) | (rows["seed"] != seed) | (rows["subject"] != outer)).any():
            problems.append(f"{policy}: variant/seed/subject columns mismatch")
        if not np.array_equal(rows["video_id"].astype(str).to_numpy(), data["video_ids"][test_indices].astype(str)) \
                or not np.array_equal(rows["label"].to_numpy(), data["labels"][test_indices]):
            problems.append(f"{policy}: video_id/label misaligned with source data")
        scores, thresholds = rows["score"].to_numpy(float), rows["threshold"].to_numpy(float)
        if not (np.isfinite(scores).all() and np.isfinite(thresholds).all() and ((scores >= 0) & (scores <= 1)).all()):
            problems.append(f"{policy}: non-finite or out-of-range score/threshold")
            continue
        if policy in choices and not np.allclose(thresholds, choices[policy]["threshold"], atol=TOLERANCE, rtol=0):
            problems.append(f"{policy}: threshold differs from the inner choice")
        if not np.array_equal((scores >= thresholds).astype(int), rows["prediction"].to_numpy()):
            problems.append(f"{policy}: prediction != (score >= threshold)")
        stored = fold_metrics.get(policy)
        if stored is not None:
            recomputed = metrics(rows["label"].to_numpy(), rows["prediction"].to_numpy())
            wrong = [key for key in RATES + ["tp", "tn", "fp", "fn"] if not same_value(recomputed[key], stored.get(key))]
            if wrong:
                problems.append(f"{policy}: stored fold metrics differ from recomputation ({', '.join(wrong)})")
    policy_scores = predictions.groupby("sample_index")["score"].nunique()
    if (policy_scores != 1).any():
        problems.append("policies do not share the same score")
    try:
        with np.load(output / f"{name}_attention.npz", allow_pickle=False) as archive:
            if not np.array_equal(archive["indices"], test_indices) or not np.isfinite(archive["attention"]).all() \
                    or archive["attention"].shape[0] != len(test_indices):
                problems.append("attention artifact misaligned or non-finite")
        checkpoint = torch.load(output / f"{name}.pt", map_location="cpu", weights_only=True)
        if checkpoint.get("test_indices") != test_indices.tolist() or checkpoint.get("variant") != VARIANTS[variant_name]:
            problems.append("checkpoint metadata does not match the fold")
        for suffix in ["inner_history", "refit_history"]:
            if pd.read_csv(output / f"{name}_{suffix}.csv").empty:
                problems.append(f"{suffix} is empty")
    except Exception as error:  # missing or corrupt artifact: report it, never accept the fold
        problems.append(f"artifact missing or unreadable ({type(error).__name__}: {error})")
    return [f"{name}: {problem}" for problem in problems]


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


def run_neural(data, config, output, device, signature_sha):
    subjects = sorted(np.unique(data["subjects"]).tolist())
    for variant_name in config["variants"]:
        variant = VARIANTS[variant_name]
        for seed in config["seeds"]:
            for outer in subjects:
                name = f"{variant_name}_seed{seed}_subject{outer}"
                target = output / f"{name}.json"
                if target.exists():
                    problems = validate_fold(data, config, output, variant_name, seed, outer, signature_sha)
                    if not problems:
                        print("Resume validated:", name, flush=True)
                        continue
                    print("Invalid previous fold, rerunning:", "; ".join(problems), flush=True)
                    target.unlink()
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
                write_json(target, {"complete": True, "signature_sha256": signature_sha, "inner_splits": splits, "inner_labels": inner_labels, "inner_scores": inner_scores,
                                    "choices": choices, "metrics": fold_metrics, "predictions": predictions, "fits": len(splits) + 1,
                                    "parameters": sum(parameter.numel() for parameter in model.parameters())})
                print("Completed", name, "epochs", epochs, flush=True)
                del model


def summarize_run(data, config, output, signature_sha, input_guard):
    (output / "completion.json").unlink(missing_ok=True)
    subjects = sorted(np.unique(data["subjects"]).tolist())
    keys = [(variant, seed, outer) for variant in config["variants"] for seed in config["seeds"] for outer in subjects]
    names = {f"{variant}_seed{seed}_subject{outer}.json" for variant, seed, outer in keys}
    unexpected = sorted(path.name for path in output.glob("Q*_seed*_subject*.json") if path.name not in names)
    problems = [f"unexpected fold report {name}" for name in unexpected]
    for variant, seed, outer in keys:
        problems += validate_fold(data, config, output, variant, seed, outer, signature_sha)
    if problems:
        raise ValueError(f"{len(problems)} fold problems; no final summary:\n" + "\n".join(problems))
    reports = [json.loads((output / f"{variant}_seed{seed}_subject{outer}.json").read_text(encoding="utf-8"))
               for variant, seed, outer in keys]
    predictions = pd.DataFrame([row for report in reports for row in report["predictions"]])
    folds = pd.DataFrame([row for report in reports for row in report["metrics"]])
    count, falls = len(data["labels"]), int(data["labels"].sum())
    if predictions.duplicated(["variant", "seed", "policy", "sample_index"]).any():
        raise ValueError("Duplicate OOF rows")
    groups = predictions.groupby(["variant", "seed", "policy"])
    expected_groups = {(variant, seed, policy) for variant in config["variants"] for seed in config["seeds"] for policy in POLICIES}
    if set(groups.groups) != expected_groups:
        raise ValueError("OOF groups do not match variants x seeds x policies")
    pooled = []
    for (variant, seed, policy), group in groups:
        group = group.sort_values("sample_index")
        if len(group) != count or not np.array_equal(group["sample_index"].to_numpy(), np.arange(count)) \
                or not np.array_equal(group["label"].to_numpy(), data["labels"]) or int(group["label"].sum()) != falls:
            raise ValueError("OOF coverage failed; do not pool duplicates across seeds")
        pooled.append({"variant": variant, "seed": int(seed), "policy": policy, **metrics(group["label"], group["prediction"])})
    pooled = pd.DataFrame(pooled)
    summary = pooled.groupby(["variant", "policy"])[RATES + ["fn", "fp"]].agg(["mean", lambda x: x.std(ddof=1)])
    summary.columns = [f"{column}_{'mean' if stat == 'mean' else 'std'}" for column, stat in summary.columns]
    for name, frame in [("predictions", predictions), ("fold_metrics", folds), ("pooled_by_seed", pooled),
                        ("errors", predictions[predictions["label"] != predictions["prediction"]])]:
        frame.to_csv(output / f"{name}.csv", index=False)
    summary.reset_index().to_csv(output / "seed_mean_std.csv", index=False)
    fits, fits_expected = sum(report["fits"] for report in reports), len(keys) * len(subjects)
    if fits != fits_expected:
        raise ValueError(f"fits {fits} != expected {fits_expected}")
    write_json(output / "completion.json", {
        "complete": True, "mode": config["mode"], "input_guard": input_guard,
        "scientific_result": config["mode"] == "full" and input_guard == "locked" and not data.get("synthetic", False),
        "variants": config["variants"], "seeds": config["seeds"], "outer_runs_expected": len(keys), "outer_runs": len(reports),
        "fits_expected": fits_expected, "fits": fits, "signature_sha256": signature_sha,
        "validation": "every fold report, policy, OOF row, metric and artifact checked against source data",
        "warning": "Smoke/synthetic runs are software checks only. Seeds repeat the same clips. Prior outer results informed development."})
    return pooled


def run_experiment(data, config, output):
    input_guard = verify_inputs(data, config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and config["mode"] == "full" and not config["allow_cpu_full"]:
        raise RuntimeError("Full experiment requires a GPU; choose smoke locally, or explicitly allow CPU full")
    signature = {"config": config, "input_guard": input_guard, "inputs": {key: value["sha256"] for key, value in data["provenance"].items()}, "code_hash": code_hash(),
                 "versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__},
                 "device": device}
    lock = output / "locked_protocol.json"
    if lock.exists() and json.loads(lock.read_text(encoding="utf-8")) != signature:
        raise ValueError("Run configuration/code/data/environment changed; use a NEW output directory")
    write_json(lock, signature)
    write_json(output / "input_provenance.json", data["provenance"])
    started = time.monotonic()
    signature_sha = signature_hash(signature)
    run_neural(data, config, output, device, signature_sha)
    summary = summarize_run(data, config, output, signature_sha, input_guard)
    write_json(output / "timing.json", {"seconds_this_invocation": time.monotonic() - started, "device": device})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=ROOT / "data/local")
    parser.add_argument("--output", type=Path, required=True, help="new directory per run")
    parser.add_argument("--mode", choices=["full", "smoke"], default="smoke")
    args = parser.parse_args()
    print(run_experiment(load_inputs(args.input_root), make_config(args.mode), args.output).to_string())
