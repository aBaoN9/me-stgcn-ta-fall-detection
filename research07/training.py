import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn

from research07.features import FEATURE_VERSION, svm_features
from research07.models import SkeletonModel, make_loader, set_seed


VARIANTS = {
    "A0_pose_average": {"channels": 3, "attention": False, "fall_multiplier": 1.0},
    "A1_motion_average": {"channels": 8, "attention": False, "fall_multiplier": 1.0},
    "A2_pose_attention": {"channels": 3, "attention": True, "fall_multiplier": 1.0},
    "A3_motion_attention": {"channels": 8, "attention": True, "fall_multiplier": 1.0},
    "A3_cost_225": {"channels": 8, "attention": True, "fall_multiplier": 2.25},
}


def default_config(mode="full"):
    if mode not in {"full", "smoke"}:
        raise ValueError("mode must be full or smoke")
    return {"mode": mode, "feature_version": FEATURE_VERSION, "seeds": [42, 142, 242] if mode == "full" else [42],
            "variants": list(VARIANTS), "max_epochs": 50 if mode == "full" else 2,
            "patience": 10 if mode == "full" else 1, "batch_size": 16, "learning_rate": 0.001,
            "weight_decay": 0.0001, "dropout": 0.3, "scheduler": None,
            "target_sensitivity": 0.95, "minimum_specificity": 0.75,
            "threshold_grid": np.linspace(0.05, 0.95, 37).tolist(), "allow_cpu_full": False,
            "epoch_selection": "minimum unweighted inner-validation cross entropy",
            "protocol": "4 outer subjects; 3 inner subjects; refit outer train; no final all-subject retrain"}


def metrics(labels, predictions):
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    ratio = lambda top, bottom: float(top / bottom) if bottom else None
    sensitivity = ratio(true_positive, true_positive + false_negative)
    specificity = ratio(true_negative, true_negative + false_positive)
    return {"accuracy": ratio(true_positive + true_negative, len(labels)),
            "balanced_accuracy": (sensitivity + specificity) / 2 if sensitivity is not None and specificity is not None else None,
            "sensitivity": sensitivity, "specificity": specificity,
            "precision": ratio(true_positive, true_positive + false_positive),
            "f1": ratio(2 * true_positive, 2 * true_positive + false_positive + false_negative),
            "f2": ratio(5 * true_positive, 5 * true_positive + false_positive + 4 * false_negative),
            "false_alarm_rate": ratio(false_positive, false_positive + true_negative),
            "tp": int(true_positive), "tn": int(true_negative), "fp": int(false_positive), "fn": int(false_negative)}


def select_thresholds(labels, scores, config, grid=None, fixed=0.5):
    candidates = []
    for threshold in config["threshold_grid"] if grid is None else grid:
        values = metrics(labels, scores >= threshold)
        meets_recall = values["sensitivity"] >= config["target_sensitivity"]
        feasible = meets_recall and values["specificity"] >= config["minimum_specificity"]
        tier = 2 if feasible else 1 if meets_recall else 0
        safety_rank = (tier, values["specificity"], values["f2"], -abs(threshold - fixed)) if tier else (0, values["sensitivity"], values["specificity"], -abs(threshold - fixed))
        candidates.append({"threshold": float(threshold), "metrics": values, "feasible": feasible,
                           "balanced_rank": (values["balanced_accuracy"], -abs(threshold - fixed)), "safety_rank": safety_rank})
    selected = {"fixed": {"threshold": fixed, "metrics": metrics(labels, scores >= fixed), "feasible": None},
                "balanced": max(candidates, key=lambda item: item["balanced_rank"]),
                "sensitivity": max(candidates, key=lambda item: item["safety_rank"])}
    return selected


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def predict(model, loader, device):
    model.eval()
    labels, probabilities, attention = [], [], []
    with torch.no_grad():
        for features, target in loader:
            logits, weights = model(features.to(device))
            labels.extend(target.tolist())
            probabilities.extend(logits.softmax(dim=1)[:, 1].cpu().tolist())
            attention.extend(weights.cpu().tolist())
    return np.asarray(labels), np.asarray(probabilities), np.asarray(attention)


def fit_model(data, train_indices, validation_indices, variant, config, seed, device, fixed_epochs=None):
    set_seed(seed)
    model = SkeletonModel(variant["channels"], variant["attention"], config["dropout"]).to(device)
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
                predictions = []
                fold_metrics = []
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
                                    "choices": choices, "metrics": fold_metrics, "predictions": predictions,
                                    "parameters": sum(parameter.numel() for parameter in model.parameters())})
                print("Completed", name, "epochs", epochs, flush=True)
                del model


def run_svm(data, config, output):
    values = np.stack([svm_features(pose, mask, duration) for pose, mask, duration in zip(data["poses"], data["masks"], data["durations"])])
    subjects = sorted(np.unique(data["subjects"]).tolist())
    for outer in subjects:
        target = output / f"SVM_v2_seed0_subject{outer}.json"
        if target.exists():
            continue
        candidates = []
        grid = [(1, "scale")] if config["mode"] == "smoke" else [(regularization, gamma) for regularization in [0.1, 1, 10] for gamma in ["scale", 0.01]]
        for regularization, gamma in grid:
            labels_all, scores_all = [], []
            for inner in subjects:
                if inner == outer:
                    continue
                train = np.flatnonzero((data["subjects"] != outer) & (data["subjects"] != inner))
                validation = np.flatnonzero(data["subjects"] == inner)
                estimator = make_pipeline(StandardScaler(), SVC(C=regularization, gamma=gamma, class_weight="balanced"))
                estimator.fit(values[train], data["labels"][train])
                scores_all.extend(estimator.decision_function(values[validation]).tolist())
                labels_all.extend(data["labels"][validation].tolist())
            score = metrics(np.asarray(labels_all), np.asarray(scores_all) >= 0)["balanced_accuracy"]
            candidates.append({"C": regularization, "gamma": gamma, "rank": score, "labels": labels_all, "scores": scores_all})
        best = max(candidates, key=lambda item: item["rank"])
        threshold_grid = np.unique(np.r_[np.quantile(best["scores"], np.linspace(0, 1, 37)), 0])
        choices = select_thresholds(np.asarray(best["labels"]), np.asarray(best["scores"]), config, threshold_grid, fixed=0)
        train, test = np.flatnonzero(data["subjects"] != outer), np.flatnonzero(data["subjects"] == outer)
        estimator = make_pipeline(StandardScaler(), SVC(C=best["C"], gamma=best["gamma"], class_weight="balanced"))
        estimator.fit(values[train], data["labels"][train])
        scores = estimator.decision_function(values[test])
        predictions, fold_metrics = [], []
        for policy, choice in choices.items():
            predicted = scores >= choice["threshold"]
            fold_metrics.append({"variant": "SVM_v2", "seed": 0, "subject": outer, "policy": policy,
                                 "threshold": choice["threshold"], "inner_target_met": choice["feasible"], **metrics(data["labels"][test], predicted)})
            for index, score, prediction in zip(test, scores, predicted):
                predictions.append({"sample_index": int(index), "video_id": str(data["video_ids"][index]), "subject": outer,
                    "label": int(data["labels"][index]), "score": float(score), "prediction": int(prediction), "threshold": choice["threshold"],
                    "variant": "SVM_v2", "seed": 0, "policy": policy})
        write_json(target, {"complete": True, "choices": choices, "search": candidates, "metrics": fold_metrics, "predictions": predictions,
                            "note": "49 motion statistics; scaler fit on training subjects only; scores are margins, not probabilities"})
        print("Completed SVM_v2 subject", outer, flush=True)


def summarize_run(config, output):
    paths = sorted(output.glob("*_seed*_subject*.json"))
    expected = len(config["variants"]) * len(config["seeds"]) * 4 + 4
    if len(paths) != expected:
        raise ValueError(f"Only {len(paths)}/{expected} complete outer runs; no final summary")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    predictions = pd.DataFrame([row for report in reports for row in report["predictions"]])
    folds = pd.DataFrame([row for report in reports for row in report["metrics"]])
    pooled = []
    for (variant, seed, policy), group in predictions.groupby(["variant", "seed", "policy"]):
        if len(group) != 160 or group["sample_index"].nunique() != 160:
            raise ValueError("OOF coverage failed; do not pool duplicates across seeds")
        pooled.append({"variant": variant, "seed": int(seed), "policy": policy, **metrics(group["label"], group["prediction"])})
    pooled = pd.DataFrame(pooled)
    for name, frame in [("predictions", predictions), ("fold_metrics", folds), ("pooled_by_seed", pooled),
                        ("errors", predictions[predictions["label"] != predictions["prediction"]])]:
        frame.to_csv(output / f"{name}.csv", index=False)
    columns = ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "f2", "false_alarm_rate"]
    pooled.groupby(["variant", "policy"])[columns].agg(["mean", "std"]).to_csv(output / "seed_mean_std.csv")
    write_json(output / "completion.json", {"complete": True, "mode": config["mode"], "scientific_result": config["mode"] == "full",
               "outer_runs": len(paths), "warning": "Smoke results are software checks only. Repeated seeds do not increase independent sample count. Prior outer results have already informed development."})
    return pooled


def run_experiment(data, config, output, code_hash):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and config["mode"] == "full" and not config["allow_cpu_full"]:
        raise RuntimeError("Full experiment requires Kaggle GPU; choose smoke locally, or explicitly allow CPU full")
    signature = {"config": config, "inputs": {key: value["sha256"] for key, value in data["provenance"].items()}, "code_hash": code_hash,
                 "versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__}, "device": device}
    lock = output / "locked_protocol.json"
    if lock.exists() and json.loads(lock.read_text(encoding="utf-8")) != signature:
        raise ValueError("Run configuration/code/data/environment changed; use a NEW output directory")
    write_json(lock, signature)
    write_json(output / "input_provenance.json", data["provenance"])
    started = time.monotonic()
    run_neural(data, config, output, device)
    run_svm(data, config, output)
    summary = summarize_run(config, output)
    write_json(output / "timing.json", {"seconds_this_invocation": time.monotonic() - started, "device": device})
    return summary
