"""Nested-LOSO runner for Duc's modern-backbone benchmark (GB-20261124-001).

One run directory per model. Same inputs, v2 loader/augmentation, recipe, splits, seeds, thresholds and metrics
as A3/07. A3 itself uses the original SkeletonModel/fit_model; every other model uses a mirror of that loop
that differs only in the model factory. Full mode requires real smoke evidence and a passing test report for
the same code, inputs, environment and device; nothing is selected on outer-test data.

    python -m training.modern_benchmark_duc.runner --model skateformer_v2_ce --input-root data/gmdcsa24 \
        --output results/modern_benchmark_duc/<run-id>/skateformer_v2_ce --mode full --device cpu --threads 4 \
        --smoke-evidence <smoke-dir> --test-report <tests.json> [--resume]
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from torch import nn

from training.component_ablation import training as original
from training.component_ablation.features import FEATURE_VERSION, find_unique, load_inputs, sha256
from training.component_ablation.models import make_loader, set_seed
from training.modern_benchmark_duc.models import (A3, ARCHITECTURE, MODEL_IDS, UPSTREAM, BlockedModel, build_model,
                                                  logits_of)

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
INPUT_NAMES = {
    "pose": ("gmdcsa24_pose_32.npz", "gmdcsa24_pose_32.zip"),
    "features": ("gmdcsa24_features_32.npz", "gmdcsa24_features_32.zip"),
    "manifest": ("gmdcsa24_pose_manifest.csv", "pose_mnf.txt"),
    "folds": ("gmdcsa24_cross_subject_folds.csv", "cross.txt"),
}
POLICIES = ("fixed", "balanced", "sensitivity")
RECIPE = original.VARIANTS[A3]            # 8 channels, attention, Fall multiplier 1.0: loader and loss recipe
RATES = ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "f2", "false_alarm_rate"]
COUNTS = ["tp", "tn", "fp", "fn"]
PREDICTION_COLUMNS = ["sample_index", "video_id", "subject", "label", "score", "prediction", "threshold",
                      "variant", "seed", "policy"]
SCORE_TOLERANCE = dict(rtol=1e-5, atol=1e-6)
FORMAT_VERSION = 1


def plain(value):
    """JSON-normalised copy (tuples become lists) so locks, checkpoints and manifests compare exactly."""
    return json.loads(json.dumps(value))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def expected_inputs():
    return read_json(ROOT / "results/component_ablation/locked_protocol.json")["inputs"]


def load_verified(root):
    """Four inputs with the 07 SHA-256, 81 ADL / 79 Fall, boolean masks and four valid subject splits."""
    expected = expected_inputs()
    for key, names in INPUT_NAMES.items():
        if sha256(find_unique(root, names)) != expected[key]:
            raise ValueError(f"Input SHA-256 mismatch: {key}")
    data = load_inputs(root)
    if {key: item["sha256"] for key, item in data["provenance"].items()} != expected:
        raise ValueError("Input provenance changed during validation")
    if np.bincount(data["labels"].astype(int)).tolist() != [81, 79]:
        raise ValueError("Expected 81 ADL and 79 Fall")
    if data["masks"].dtype != np.bool_:
        raise ValueError("Expected boolean detection mask")
    for outer in (1, 2, 3, 4):
        splits(data, outer)
    return data


def splits(data, outer):
    subjects, labels = data["subjects"], data["labels"]
    people = {1, 2, 3, 4}
    if set(np.unique(subjects).tolist()) != people or outer not in people:
        raise ValueError("Invalid subject inventory or outer subject")
    train, test = np.flatnonzero(subjects != outer), np.flatnonzero(subjects == outer)
    inner = [(person, np.flatnonzero((subjects != outer) & (subjects != person)), np.flatnonzero(subjects == person))
             for person in sorted(people - {outer})]
    for _, fit_indices, val in inner:
        if set(fit_indices) & set(val) or (set(fit_indices) | set(val)) != set(train) or set(train) & set(test):
            raise ValueError("Invalid inner split")
        if any(set(labels[idx].tolist()) != {0, 1} for idx in (fit_indices, val, test)):
            raise ValueError("Missing class in split")
    return train, test, inner


def scores_for(model, data, indices, cfg, device):
    loader = make_loader(data, indices, RECIPE, cfg, False, 0)
    model.eval()
    with torch.no_grad():
        scores = torch.cat([logits_of(model, x.to(device)).softmax(1)[:, 1].cpu() for x, _ in loader]).numpy()
    if scores.shape != (len(indices),) or not np.isfinite(scores).all():
        raise ValueError("Nonfinite or misaligned score")
    return scores.astype(np.float64)


def validation_loss(labels, scores):
    scores = np.clip(scores, 1e-7, 1 - 1e-7)
    return float(-(labels * np.log(scores) + (1 - labels) * np.log(1 - scores)).mean())


def fit(model_id, data, train, val, cfg, seed, device, epochs=None):
    """A3: original fit_model untouched. Others: same steps/order, only the model factory differs."""
    if model_id == A3:
        return original.fit_model(data, train, val, RECIPE, cfg, seed, device, epochs)
    return fit_generic(lambda: build_model(model_id), data, train, val, cfg, seed, device, epochs)


def fit_generic(factory, data, train, val, cfg, seed, device, epochs=None):
    """Mirror of original.fit_model; test_generic_loop_matches_original_a3 checks bit-parity with A3."""
    set_seed(seed)
    model = factory().to(device)
    loader = make_loader(data, train, RECIPE, cfg, True, seed)
    counts = np.bincount(data["labels"][train], minlength=2)
    if np.any(counts == 0):
        raise ValueError("Training fold has a missing class")
    weights = len(train) / (2 * counts.astype(np.float32))
    weights[1] *= RECIPE["fall_multiplier"]
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    history, best, best_epoch, stale, state = [], float("inf"), 0, 0, None
    for epoch in range(1, (epochs or cfg["max_epochs"]) + 1):
        model.train()
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(logits_of(model, x), y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at epoch {epoch}")
            loss.backward()
            if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise FloatingPointError(f"Nonfinite gradient at epoch {epoch}")
            optimizer.step()
            total += loss.item() * len(y)
        record = {"epoch": epoch, "train_weighted_loss": total / len(train)}
        if val is not None:
            value = validation_loss(data["labels"][val], scores_for(model, data, val, cfg, device))
            record["validation_unweighted_loss"] = value
            if value < best - 1e-6:
                best, best_epoch, stale = value, epoch, 0
                state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
            else:
                stale += 1
        history.append(record)
        if val is not None and stale >= cfg["patience"]:
            break
    if state is not None:
        model.load_state_dict(state)
    return model, best_epoch if val is not None else epochs, history


def fold_metadata(model_id, cfg, seed, outer, epochs, train, test):
    return dict(variant=model_id, architecture=plain(ARCHITECTURE[model_id]), feature_version=FEATURE_VERSION, seed=seed,
                subject=outer, refit_seed=seed + outer * 1000 + 999, epochs=epochs, train=train.tolist(),
                test=test.tolist())


def load_model(path, device, metadata=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["feature_version"] != FEATURE_VERSION:
        raise ValueError("Checkpoint feature version mismatch")
    if metadata is not None and json.loads(json.dumps({k: v for k, v in checkpoint.items() if k != "state"})) != \
            json.loads(json.dumps(metadata)):
        raise ValueError("Checkpoint metadata mismatch")
    if plain(checkpoint["architecture"]) != plain(ARCHITECTURE[checkpoint["variant"]]):
        raise ValueError("Checkpoint architecture differs from the locked architecture")
    model = build_model(checkpoint["variant"])
    model.load_state_dict(checkpoint["state"])
    return model.to(device).eval()


def run_fold(model_id, data, outer, seed, cfg, device, folder, signature):
    started = time.monotonic()
    train, test, inner = splits(data, outer)
    inner_rows, histories, epochs = [], [], []
    for person, fit_indices, val in inner:
        model, epoch, history = fit(model_id, data, fit_indices, val, cfg, seed + outer * 1000 + person * 10, device)
        scores = scores_for(model, data, val, cfg, device)
        epochs.append(epoch)
        histories.append(dict(subject=person, fit=fit_indices.tolist(), val=val.tolist(), selected_epoch=epoch,
                              history=history))
        inner_rows += [dict(sample_index=int(i), video_id=str(data["video_ids"][i]), subject=int(person),
                            label=int(data["labels"][i]), score=float(s)) for i, s in zip(val, scores)]
        del model
    labels = np.asarray([row["label"] for row in inner_rows])
    choices = original.select_thresholds(labels, np.asarray([row["score"] for row in inner_rows]), cfg)
    selected = int(np.median(epochs))
    model, _, refit = fit(model_id, data, train, None, cfg, seed + outer * 1000 + 999, device, selected)
    metadata = fold_metadata(model_id, cfg, seed, outer, selected, train, test)
    path = folder / "model.pt"
    torch.save(dict(metadata, state={k: v.detach().cpu() for k, v in model.state_dict().items()}), path)
    scores = scores_for(model, data, test, cfg, device)
    np.testing.assert_allclose(scores_for(load_model(path, device, metadata), data, test, cfg, device), scores,
                               **SCORE_TOLERANCE)
    rows = [dict(sample_index=int(i), video_id=str(data["video_ids"][i]), subject=outer, label=int(data["labels"][i]),
                 score=float(s), prediction=int(s >= choice["threshold"]), threshold=choice["threshold"],
                 variant=model_id, seed=seed, policy=policy)
            for policy, choice in choices.items() for i, s in zip(test, scores)]
    original.write_json(folder / "fold.json", dict(
        signature=signature, metadata=metadata, inner=histories, inner_oof=inner_rows, refit=refit, choices=choices,
        predictions=rows, checkpoint_sha256=sha256(path), parameters=sum(p.numel() for p in model.parameters()),
        seconds=time.monotonic() - started))


def check_history(entries, limit, need_validation):
    if not 1 <= len(entries) <= limit or [e["epoch"] for e in entries] != list(range(1, len(entries) + 1)):
        raise ValueError("Invalid epoch history")
    for entry in entries:
        if not all(np.isfinite(float(v)) for v in entry.values()):
            raise ValueError("Nonfinite epoch history")
        if need_validation != ("validation_unweighted_loss" in entry):
            raise ValueError("Validation loss present/absent where it should not be")


def validate_fold(folder, model_id, data, outer, seed, cfg, device, signature):
    """Re-derive every choice of one outer fold from its saved history and re-score its checkpoint."""
    record = read_json(folder / "fold.json")
    train, test, inner = splits(data, outer)
    if record["signature"] != signature or len(record["inner"]) != len(inner):
        raise ValueError("Fold signature or inner-fit count mismatch")
    epochs, oof = [], pd.DataFrame(record["inner_oof"])
    for saved, (person, fit_indices, val) in zip(record["inner"], inner):
        if (saved["subject"], saved["fit"], saved["val"]) != (person, fit_indices.tolist(), val.tolist()):
            raise ValueError("Inner split mismatch")
        check_history(saved["history"], cfg["max_epochs"], True)
        best, selected = float("inf"), 0
        for entry in saved["history"]:
            if entry["validation_unweighted_loss"] < best - 1e-6:
                best, selected = entry["validation_unweighted_loss"], entry["epoch"]
        stale = len(saved["history"]) - selected
        if saved["selected_epoch"] != selected or (len(saved["history"]) < cfg["max_epochs"] and stale != cfg["patience"]):
            raise ValueError("Inner epoch selection or early stopping mismatch")
        epochs.append(selected)
        rows = oof[oof["subject"] == person]
        if rows["sample_index"].tolist() != val.tolist() or rows["label"].tolist() != data["labels"][val].tolist() \
                or rows["video_id"].tolist() != data["video_ids"][val].astype(str).tolist():
            raise ValueError("Inner OOF rows misaligned")
        if not np.isclose(validation_loss(data["labels"][val], rows["score"].to_numpy()),
                          saved["history"][selected - 1]["validation_unweighted_loss"], rtol=1e-6, atol=1e-9):
            raise ValueError("Inner OOF scores do not reproduce the selected-epoch validation loss")
    if len(oof) != len(train) or not np.isfinite(oof["score"]).all() or ((oof["score"] < 0) | (oof["score"] > 1)).any():
        raise ValueError("Invalid inner OOF")
    epoch = int(np.median(epochs))
    check_history(record["refit"], epoch, False)
    if len(record["refit"]) != epoch:
        raise ValueError("Incomplete refit history")
    metadata = fold_metadata(model_id, cfg, seed, outer, epoch, train, test)
    if json.loads(json.dumps(record["metadata"])) != json.loads(json.dumps(metadata)):
        raise ValueError("Fold metadata mismatch")
    choices = original.select_thresholds(oof["label"].to_numpy(), oof["score"].to_numpy(), cfg)
    if json.loads(json.dumps(choices)) != record["choices"]:
        raise ValueError("Threshold selection mismatch")
    path = folder / "model.pt"
    checksum = sha256(path)
    if checksum != record["checkpoint_sha256"]:
        raise ValueError("Checkpoint SHA-256 mismatch")
    scores = scores_for(load_model(path, device, metadata), data, test, cfg, device)
    rows = record["predictions"]
    if [(r["policy"], r["sample_index"]) for r in rows] != [(p, int(i)) for p in POLICIES for i in test]:
        raise ValueError("Fold prediction coverage mismatch")
    for row, score in zip(rows, np.tile(scores, len(POLICIES))):
        idx, threshold = row["sample_index"], choices[row["policy"]]["threshold"]
        expected = (model_id, seed, outer, str(data["video_ids"][idx]), int(data["labels"][idx]), threshold)
        actual = tuple(row[k] for k in ("variant", "seed", "subject", "video_id", "label", "threshold"))
        if actual != expected or not 0 <= row["score"] <= 1 or not np.isclose(row["score"], score, **SCORE_TOLERANCE) \
                or row["prediction"] != int(row["score"] >= threshold):
            raise ValueError("Fold predictions do not match checkpoint or dataset")
    inner_rows = [dict(row, variant=model_id, seed=seed, outer_subject=outer) for row in record["inner_oof"]]
    histories = [dict(entry, variant=model_id, seed=seed, outer_subject=outer, fit=f"inner_{saved['subject']}")
                 for saved in record["inner"] for entry in saved["history"]]
    histories += [dict(entry, variant=model_id, seed=seed, outer_subject=outer, fit="refit") for entry in record["refit"]]
    artifact = dict(metadata, file=f"folds/{model_id}_seed{seed}_subject{outer}/model.pt", sha256=checksum,
                    bytes=path.stat().st_size,
                    parameters=record["parameters"], seconds=record["seconds"])
    artifact.pop("train"), artifact.pop("test")
    return dict(predictions=rows, inner=inner_rows, histories=histories, artifact=artifact)


def metric_table(frame, keys):
    return pd.DataFrame([dict(zip(keys, key if isinstance(key, tuple) else (key,)),
                              **original.metrics(group["label"].to_numpy(), group["prediction"].to_numpy()))
                         for key, group in frame.groupby(keys)])


def summarize(rows, data, model_id, seeds, outers):
    frame = pd.DataFrame(rows)[PREDICTION_COLUMNS]
    if frame.duplicated(["variant", "seed", "policy", "sample_index"]).any():
        raise ValueError("Duplicate prediction key")
    groups = frame.groupby(["variant", "seed", "policy"])
    if set(groups.groups) != {(model_id, s, p) for s in seeds for p in POLICIES}:
        raise ValueError("Incorrect variant/seed/policy coverage")
    target = np.flatnonzero(np.isin(data["subjects"], outers))
    for _, group in groups:
        indices = group["sample_index"].to_numpy()
        if not np.array_equal(np.sort(indices), target):
            raise ValueError("Incorrect sample coverage")
        for column, source in (("video_id", "video_ids"), ("label", "labels"), ("subject", "subjects")):
            if not np.array_equal(group[column].astype(str if column == "video_id" else int).to_numpy(),
                                  data[source][indices].astype(str if column == "video_id" else int)):
                raise ValueError(f"Prediction alignment mismatch: {column}")
        if not np.isfinite(group[["score", "threshold"]].to_numpy(float)).all():
            raise ValueError("Nonfinite prediction or threshold")
        if not np.array_equal(group["prediction"].to_numpy(), (group["score"] >= group["threshold"]).astype(int).to_numpy()):
            raise ValueError("Prediction does not match threshold")
    if (frame.groupby(["variant", "seed", "sample_index"])["score"].nunique() != 1).any():
        raise ValueError("Policies must reuse the same score")
    folds = metric_table(frame, ["variant", "seed", "policy", "subject"])
    pooled = metric_table(frame, ["variant", "seed", "policy"])
    return frame, folds, pooled


def seed_mean_std(pooled):
    rows = []
    for (variant, policy), group in pooled.groupby(["variant", "policy"]):
        row = dict(variant=variant, policy=policy, n_seeds=len(group))
        for column in RATES + COUNTS:  # undefined metric or a single seed -> NaN (empty cell), never 0
            values = group[column].astype(float)
            row[f"{column}_mean"] = values.mean() if values.notna().all() else np.nan
            row[f"{column}_std"] = values.std(ddof=1) if values.notna().all() and len(values) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def environment(device, threads):
    return dict(python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__,
                pandas=pd.__version__, sklearn=sklearn.__version__, platform=platform.platform(),
                processor=platform.processor(), device=str(device), threads=threads, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                precision="float32, no AMP", tf32_matmul=bool(torch.backends.cuda.matmul.allow_tf32),
                cudnn="deterministic=True, benchmark=False (set by set_seed before every fit)")


def code_hashes():
    sources = sorted([*PACKAGE.glob("*.py"), *(ROOT / "training/component_ablation").glob("*.py")])
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for p in sources}


def protocol_lock(data, model_id, mode, device, threads, synthetic, smoke_signature=None):
    cfg = original.default_config(mode)
    cfg["variants"] = [model_id]
    probe = build_model(model_id)
    lock = dict(format_version=FORMAT_VERSION, decision="GB-20261124-001", model_id=model_id,
                architecture=ARCHITECTURE[model_id], upstream=UPSTREAM.get(model_id, "training/component_ablation (07)"),
                parameters=sum(p.numel() for p in probe.parameters()), config=cfg,
                outer_subjects=[1] if mode == "smoke" else [1, 2, 3, 4], policies=list(POLICIES),
                primary=dict(policy="fixed", threshold=0.5, metrics=["f1 (Fall class)", "accuracy"],
                             aggregation="pooled 160 clips per seed, mean and sample SD (ddof=1) over seeds"),
                loader_recipe=RECIPE, synthetic=synthetic,
                inputs={k: v["sha256"] for k, v in data.get("provenance", {}).items()}, code=code_hashes(),
                environment=environment(device, threads), smoke_signature=smoke_signature)
    if synthetic:
        lock["fixture_hash"] = {k: hashlib.sha256(np.asarray(data[k]).tobytes()).hexdigest()
                                for k in ("poses", "masks", "durations", "labels", "subjects", "video_ids")}
    return json.loads(json.dumps(lock))


def signature_for(lock):
    return hashlib.sha256(json.dumps(lock, sort_keys=True).encode()).hexdigest()


def fold_inventory(lock):
    return [(seed, outer, f"{lock['model_id']}_seed{seed}_subject{outer}")
            for seed in lock["config"]["seeds"] for outer in lock["outer_subjects"]]


def collect(data, lock, device, output, allow_missing=False):
    root = output / "folds"
    expected = fold_inventory(lock)
    actual = {p.name for p in root.iterdir() if not p.name.startswith(".pending-")} if root.exists() else set()
    names = {tag for _, _, tag in expected}
    if actual - names or (not allow_missing and actual != names):
        raise ValueError("Unexpected or missing seed/fold in run")
    return {tag: validate_fold(root / tag, lock["model_id"], data, outer, seed, lock["config"], device,
                               signature_for(lock))
            for seed, outer, tag in expected if tag in actual}


def write_outputs(output, data, lock, verified):
    cfg = lock["config"]
    rows = [row for tag in sorted(verified) for row in verified[tag]["predictions"]]
    frame, folds, pooled = summarize(rows, data, lock["model_id"], cfg["seeds"], lock["outer_subjects"])
    frame = frame.sort_values(["variant", "seed", "policy", "sample_index"]).reset_index(drop=True)
    inner = pd.DataFrame([row for tag in sorted(verified) for row in verified[tag]["inner"]])
    histories = pd.DataFrame([row for tag in sorted(verified) for row in verified[tag]["histories"]])
    manifest = [verified[tag]["artifact"] for _, _, tag in fold_inventory(lock)]
    (output / "models").mkdir(exist_ok=True)
    original.write_json(output / "models/manifest.json", manifest)
    tables = dict(predictions=frame, fold_metrics=folds, pooled_by_seed=pooled,
                  errors=frame[frame["label"] != frame["prediction"]], inner_oof=inner, histories=histories,
                  seed_mean_std=seed_mean_std(pooled))
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    return frame, manifest


def completion_record(lock, manifest, frame, synthetic, targets):
    outer = len(fold_inventory(lock))
    mode = lock["config"]["mode"]
    return dict(mode=mode, complete=True, synthetic=synthetic, scientific_result=mode == "full" and not synthetic,
                model_id=lock["model_id"], signature=signature_for(lock), seeds=lock["config"]["seeds"],
                outer_subjects=lock["outer_subjects"], policies=list(POLICIES),
                outer_runs_expected=outer, outer_runs=len(manifest), inner_fits_expected=3 * outer,
                inner_fits=3 * len(manifest), refits_expected=outer, refits=len(manifest),
                fits_expected=4 * outer, fits=4 * len(manifest),
                prediction_rows_expected=len(POLICIES) * len(lock["config"]["seeds"]) * targets,
                prediction_rows=len(frame), reload="PASS", coverage="PASS",
                train_seconds=float(sum(item["seconds"] for item in manifest)))


def target_count(data, lock):
    return int(np.isin(data["subjects"], lock["outer_subjects"]).sum())


def validate_summary(data, lock, output, verified):
    """Recompute every CSV from the validated folds; any hand edit or missing artifact fails."""
    cfg = lock["config"]
    rows = [row for tag in sorted(verified) for row in verified[tag]["predictions"]]
    frame, folds, pooled = summarize(rows, data, lock["model_id"], cfg["seeds"], lock["outer_subjects"])
    frame = frame.sort_values(["variant", "seed", "policy", "sample_index"]).reset_index(drop=True)
    saved = pd.read_csv(output / "predictions.csv", dtype={"video_id": str}, float_precision="round_trip")
    pd.testing.assert_frame_equal(saved, frame, check_exact=False, rtol=1e-12, atol=0, check_dtype=False)
    for name, table in (("pooled_by_seed", pooled), ("fold_metrics", folds), ("seed_mean_std", seed_mean_std(pooled))):
        stored = pd.read_csv(output / f"{name}.csv")
        pd.testing.assert_frame_equal(stored.reset_index(drop=True), table.reset_index(drop=True), check_exact=False,
                                      rtol=1e-9, atol=1e-12, check_dtype=False)
    manifest = [verified[tag]["artifact"] for _, _, tag in fold_inventory(lock)]
    if read_json(output / "models/manifest.json") != json.loads(json.dumps(manifest)):
        raise ValueError("Checkpoint manifest mismatch")
    done = read_json(output / "completion.json")
    expected = completion_record(lock, manifest, frame, lock["synthetic"], target_count(data, lock))
    for key in ("mode", "complete", "synthetic", "scientific_result", "model_id", "signature", "outer_runs", "fits",
                "inner_fits", "refits", "prediction_rows", "reload", "coverage"):
        if done.get(key) != expected[key]:
            raise ValueError(f"Completion metadata mismatch: {key}")
    if done["outer_runs"] != done["outer_runs_expected"] or done["fits"] != done["fits_expected"]             or done["prediction_rows"] != done["prediction_rows_expected"]:
        raise ValueError("Completion counts differ from expectation")
    return done


def verify_smoke(data, model_id, device, threads, smoke_dir):
    if {k: v["sha256"] for k, v in data.get("provenance", {}).items()} != expected_inputs():
        raise ValueError("Smoke evidence requires the four verified real inputs")
    lock = protocol_lock(data, model_id, "smoke", device, threads, False)
    if read_json(smoke_dir / "locked_protocol.json") != lock:
        raise ValueError("Smoke data/code/config/environment/model mismatch; rerun smoke with this code")
    verified = collect(data, lock, device, smoke_dir)
    done = validate_summary(data, lock, smoke_dir, verified)
    if done["mode"] != "smoke" or done["synthetic"] or done["scientific_result"]:
        raise ValueError("Smoke evidence must be a real, non-scientific smoke run")
    return signature_for(lock)


def verify_test_report_dict(report):
    code = code_hashes()
    digest = hashlib.sha256(json.dumps(code, sort_keys=True).encode()).hexdigest()
    if report.get("status") != "PASS" or report.get("code") != code or report.get("failures") or report.get("errors") \
            or report.get("code_sha256") != digest:
        raise ValueError("Test report is missing, failing or for different code")
    return digest


def verify_test_report(path):
    return verify_test_report_dict(read_json(path))


class WriterLock:
    def __init__(self, output):
        self.path = output / ".writer.lock"

    def __enter__(self):
        descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(descriptor, "w") as stream:
            stream.write(str(os.getpid()))
        return self

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)


def run(data, model_id, mode, device, output, threads, synthetic=False, resume=False, smoke_evidence=None,
        test_report=None):
    if model_id not in MODEL_IDS:
        raise ValueError(f"Unknown model id {model_id!r}")
    build_model(model_id)  # raises BlockedModel for DS-GCN before anything is written
    if synthetic and mode != "smoke":
        raise ValueError("Synthetic inputs are limited to software tests")
    if not synthetic and {k: v["sha256"] for k, v in data.get("provenance", {}).items()} != expected_inputs():
        raise ValueError("Run requires the four verified real inputs")
    smoke_signature = None
    if mode == "full":
        if smoke_evidence is None or test_report is None:
            raise ValueError("Full requires --smoke-evidence (real smoke) and --test-report (PASS for this code)")
        verify_test_report(test_report)
        smoke_signature = verify_smoke(data, model_id, device, threads, Path(smoke_evidence))
    lock = protocol_lock(data, model_id, mode, device, threads, synthetic, smoke_signature)
    signature = signature_for(lock)
    output = Path(output)
    if resume:
        if read_json(output / "locked_protocol.json") != lock:
            raise ValueError("Resume data/code/config/environment mismatch; use a new output directory")
    else:
        output.mkdir(parents=True, exist_ok=False)
        original.write_json(output / "locked_protocol.json", lock)
    with WriterLock(output):
        (output / "folds").mkdir(exist_ok=True)
        for stale in (output / "folds").glob(".pending-*"):
            shutil.rmtree(stale)
        finalized = (output / "completion.json").exists()
        verified = collect(data, lock, device, output, allow_missing=not finalized)
        if finalized:
            return validate_summary(data, lock, output, verified)
        for seed, outer, tag in fold_inventory(lock):
            if tag in verified:
                print("Resume validated:", tag, flush=True)
                continue
            staged = output / "folds" / f".pending-{tag}"
            staged.mkdir()
            run_fold(model_id, data, outer, seed, lock["config"], device, staged, signature)
            verified[tag] = validate_fold(staged, model_id, data, outer, seed, lock["config"], device, signature)
            staged.rename(output / "folds" / tag)
            print(f"Completed {tag} epochs {verified[tag]['artifact']['epochs']} "
                  f"seconds {verified[tag]['artifact']['seconds']:.0f}", flush=True)
        frame, manifest = write_outputs(output, data, lock, verified)
        original.write_json(output / "completion.json",
                            completion_record(lock, manifest, frame, synthetic, target_count(data, lock)))
        return validate_summary(data, lock, output, collect(data, lock, device, output))


def main():
    parser = argparse.ArgumentParser(description="Duc modern benchmark: nested LOSO for one model")
    parser.add_argument("--model", required=True, help=f"one of {', '.join(MODEL_IDS)}")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=None, help="CPU threads (recorded in the lock)")
    parser.add_argument("--smoke-evidence", type=Path, help="completed real smoke directory of the same model")
    parser.add_argument("--test-report", type=Path, help="PASS report of test_modern for the same code")
    parser.add_argument("--resume", action="store_true", help="continue a run with the identical lock")
    args = parser.parse_args()
    if args.model not in MODEL_IDS:
        parser.error(f"unknown model {args.model!r}; allowed: {', '.join(MODEL_IDS)}")
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable for this Python environment")
    try:
        result = run(load_verified(args.input_root), args.model, args.mode, torch.device(args.device), args.output,
                     torch.get_num_threads(), resume=args.resume, smoke_evidence=args.smoke_evidence,
                     test_report=args.test_report)
    except BlockedModel as error:
        raise SystemExit(f"BLOCKED: {error}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
