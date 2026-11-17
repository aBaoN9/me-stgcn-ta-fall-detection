"""Reload every checkpoint of a finished Q0/Q1 run, re-score its outer subject and rebuild the metrics.

Deliverable check of GB-20261116-001 (model + results reviewers can reproduce). For each
<run>/models/<variant>_seed<seed>_subject<outer>.pt it:
  1. checks the run is complete, the checkpoint metadata and run signature match the fold,
  2. rebuilds the model from code + config, loads the weights and predicts the held-out subject,
  3. compares the scores with predictions.csv (all three policies share one score),
  4. recomputes fixed-policy fold and pooled metrics from the reloaded scores and compares them with
     fold_metrics.csv / pooled_by_seed.csv.
It writes <run>/models/manifest.json (sha256, size, fold metadata) and <run>/model_reload_check.json.
Scores may differ by float noise between GPU training and CPU reload; predictions must not flip unless the
score lies within the tolerance of the threshold, and such cases are listed, never hidden.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from training.component_ablation.features import load_inputs
from training.component_ablation.models import make_loader
from training.component_ablation.training import metrics, predict, write_json
from training.contribution_study.runner import (MODEL_DIR, RATES, VARIANTS, build_model, code_hash, signature_hash,
                                                verify_inputs)

SCORE_TOLERANCE = 1e-4
METRIC_TOLERANCE = 1e-9


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def close(left, right):
    if left is None or right is None or (isinstance(right, float) and np.isnan(right)):
        return (left is None or (isinstance(left, float) and np.isnan(left))) and \
               (right is None or (isinstance(right, float) and np.isnan(right)))
    return abs(float(left) - float(right)) <= METRIC_TOLERANCE


def evaluate_run(data, run, device="cpu"):
    run = Path(run)
    problems, folds, manifest = [], [], []
    completion = json.loads((run / "completion.json").read_text(encoding="utf-8"))
    signature = json.loads((run / "locked_protocol.json").read_text(encoding="utf-8"))
    config, signature_sha = signature["config"], signature_hash(signature)
    if completion.get("complete") is not True or completion.get("signature_sha256") != signature_sha:
        problems.append("completion.json is not complete or belongs to another signature")
    if verify_inputs(data, config) != signature["input_guard"]:
        problems.append("input guard of the reload differs from the run")
    reload_inputs = {key: value["sha256"] for key, value in data["provenance"].items()}
    if reload_inputs != signature["inputs"]:
        problems.append("reload inputs differ from the inputs the run was trained on")
    if code_hash() != signature["code_hash"]:
        problems.append("current code hash differs from the run; reload uses different model code")
    predictions = pd.read_csv(run / "predictions.csv")
    stored_folds = pd.read_csv(run / "fold_metrics.csv")
    stored_pooled = pd.read_csv(run / "pooled_by_seed.csv")
    subjects = sorted(np.unique(data["subjects"]).tolist())
    reloaded = []
    for variant_name in config["variants"]:
        variant = VARIANTS[variant_name]
        for seed in config["seeds"]:
            for outer in subjects:
                name = f"{variant_name}_seed{seed}_subject{outer}"
                path = run / MODEL_DIR / f"{name}.pt"
                if not path.exists():
                    problems.append(f"{name}: checkpoint missing")
                    continue
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                test_indices = np.flatnonzero(data["subjects"] == outer)
                expected = {"variant": variant, "test_indices": test_indices.tolist(), "seed": seed,
                            "outer_subject": outer, "variant_name": variant_name, "signature_sha256": signature_sha}
                wrong = [key for key, value in expected.items() if checkpoint.get(key) != value]
                if wrong:
                    problems.append(f"{name}: checkpoint metadata mismatch ({', '.join(wrong)})")
                if set(checkpoint.get("train_indices", [])) & set(test_indices.tolist()):
                    problems.append(f"{name}: held-out subject leaked into train_indices")
                model = build_model(variant, config)
                model.load_state_dict(checkpoint["model_state_dict"])
                model.to(device)
                labels, scores, _ = predict(model, make_loader(data, test_indices, variant, config, False, seed), device)
                rows = predictions[(predictions["variant"] == variant_name) & (predictions["seed"] == seed)
                                   & (predictions["policy"] == "fixed") & (predictions["subject"] == outer)]
                rows = rows.sort_values("sample_index")
                if not np.array_equal(rows["sample_index"].to_numpy(), test_indices):
                    problems.append(f"{name}: predictions.csv rows do not cover subject {outer}")
                    continue
                difference = np.abs(scores - rows["score"].to_numpy(float))
                threshold = float(rows["threshold"].iloc[0])
                predicted = (scores >= threshold).astype(int)
                flips = np.flatnonzero(predicted != rows["prediction"].to_numpy())
                borderline = [int(test_indices[i]) for i in flips if abs(scores[i] - threshold) <= SCORE_TOLERANCE]
                hard_flips = [int(test_indices[i]) for i in flips if abs(scores[i] - threshold) > SCORE_TOLERANCE]
                if difference.max() > SCORE_TOLERANCE or hard_flips:
                    problems.append(f"{name}: reloaded scores differ (max {difference.max():.2e}, flips {hard_flips})")
                recomputed = metrics(labels, predicted)
                stored = stored_folds[(stored_folds["variant"] == variant_name) & (stored_folds["seed"] == seed)
                                      & (stored_folds["subject"] == outer) & (stored_folds["policy"] == "fixed")]
                if len(stored) != 1 or (not borderline and not all(close(recomputed[key], stored.iloc[0][key])
                                                                   for key in ["tp", "tn", "fp", "fn"])):
                    problems.append(f"{name}: fixed fold confusion matrix differs after reload")
                reloaded += [{"variant": variant_name, "seed": seed, "sample_index": int(i), "label": int(label),
                              "prediction": int(p)} for i, label, p in zip(test_indices, labels, predicted)]
                folds.append({"name": name, "max_abs_score_diff": float(difference.max()), "borderline_flips": borderline,
                              "threshold": threshold, "epochs": checkpoint.get("epochs")})
                manifest.append({"file": f"{MODEL_DIR}/{name}.pt", "sha256": sha256(path), "bytes": path.stat().st_size,
                                 "variant": variant_name, "seed": seed, "refit_seed": checkpoint.get("refit_seed"),
                                 "outer_subject": outer, "train_subjects": [s for s in subjects if s != outer],
                                 "epochs": checkpoint.get("epochs"), "feature_version": checkpoint.get("feature_version"),
                                 "parameters": int(sum(v.numel() for v in checkpoint["model_state_dict"].values()
                                                       if v.dtype.is_floating_point)),
                                 "metrics_scope": "outer-test of this subject only; not an all-subject model"})
    frame = pd.DataFrame(reloaded)
    for (variant_name, seed), group in (frame.groupby(["variant", "seed"]) if not frame.empty else []):
        group = group.sort_values("sample_index")
        if len(group) != len(data["labels"]):
            problems.append(f"{variant_name} seed {seed}: reload covers {len(group)} of {len(data['labels'])} clips")
            continue
        pooled = metrics(group["label"].to_numpy(), group["prediction"].to_numpy())
        stored = stored_pooled[(stored_pooled["variant"] == variant_name) & (stored_pooled["seed"] == seed)
                               & (stored_pooled["policy"] == "fixed")]
        if len(stored) != 1 or not all(close(pooled[key], stored.iloc[0][key]) for key in RATES + ["fn", "fp"]):
            problems.append(f"{variant_name} seed {seed}: pooled fixed metrics differ after reload")
    expected_models = len(config["variants"]) * len(config["seeds"]) * len(subjects)
    if len(manifest) != expected_models:
        problems.append(f"{len(manifest)} checkpoints verified, expected {expected_models}")
    write_json(run / MODEL_DIR / "manifest.json", {
        "run": run.name, "signature_sha256": signature_sha, "code_hash": signature["code_hash"],
        "inputs": signature["inputs"], "mode": config["mode"], "scientific_result": completion.get("scientific_result"),
        "load": "QualityAwareSkeletonModel(channels, attention, dropout, quality_prior, quality_epsilon) from "
                "training.contribution_study.runner.build_model(checkpoint['variant'], config); "
                "load_state_dict(checkpoint['model_state_dict'])",
        "models": manifest})
    report = {"status": "PASS" if not problems else "FAIL", "problems": problems, "folds": folds,
              "score_tolerance": SCORE_TOLERANCE, "reload_device": device,
              "versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__}}
    write_json(run / "model_reload_check.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = evaluate_run(load_inputs(args.input_root), args.run, args.device)
    print(result["status"], *result["problems"], sep="\n")
    raise SystemExit(0 if result["status"] == "PASS" else 1)
