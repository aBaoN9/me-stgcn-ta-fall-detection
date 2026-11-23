"""Supplemental fork of the accepted runner; preserve its source hashes."""

import argparse
import hashlib
import json
import platform
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from torch import nn

from training.component_ablation import training as original
from training.component_ablation.features import FEATURE_VERSION, sha256
from training.component_ablation.models import make_loader, set_seed

from ..models import A3
from ..runner import expected_inputs, load_verified, read_json, scores_for, signature_for, splits
from .models import SKATE, build_model

ROOT = Path(__file__).resolve().parents[3]
POLICIES = ("fixed", "balanced", "sensitivity")
VARIANTS = (SKATE,)


def fit(variant, data, train, val, cfg, seed, device, epochs=None):
    recipe = original.VARIANTS[A3]

    set_seed(seed)
    model = build_model(variant, cfg["dropout"]).to(device)
    loader = make_loader(data, train, recipe, cfg, True, seed)
    counts = np.bincount(data["labels"][train], minlength=2)
    weights = len(train) / (2 * counts.astype(np.float32))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                                  weight_decay=cfg["weight_decay"])
    history, best, best_epoch, stale = [], float("inf"), 0, 0
    state = None
    for epoch in range(1, (epochs or cfg["max_epochs"]) + 1):
        model.train()
        total = 0.0
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x.to(device)), y.to(device))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise ValueError("Nonfinite gradient")
            optimizer.step()
            total += loss.item() * len(y)
        record = {"epoch": epoch, "train_weighted_loss": total / len(train)}
        history.append(record)
        if val is None:
            continue
        scores = np.clip(scores_for(model, data, val, cfg, device), 1e-7, 1 - 1e-7)
        labels = data["labels"][val]
        value = float(-(labels * np.log(scores) + (1 - labels) * np.log(1 - scores)).mean())
        record["validation_unweighted_loss"] = value
        if value < best - 1e-6:
            best, best_epoch, stale = value, epoch, 0
            state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
        else:
            stale += 1
        if stale >= cfg["patience"]:
            break
    if state is not None:
        model.load_state_dict(state)
    return model, best_epoch if val is not None else epochs, history


def load_model(path, device, metadata=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["feature_version"] != FEATURE_VERSION:
        raise ValueError("Checkpoint feature version mismatch")
    if metadata is not None and {k: v for k, v in checkpoint.items() if k != "state"} != metadata:
        raise ValueError("Checkpoint metadata mismatch")
    model = build_model(checkpoint["variant"], checkpoint["dropout"])
    model.load_state_dict(checkpoint["state"])
    return model.to(device).eval()


def run_fold(variant, data, outer, seed, cfg, device, output, signature):
    train, test, inner = splits(data, outer)
    val_labels, val_scores, epochs, histories = [], [], [], []
    for person, fit_indices, val in inner:
        model, epoch, history = fit(variant, data, fit_indices, val, cfg,
                                    seed + outer * 1000 + person * 10, device)
        epochs.append(epoch)
        val_labels.extend(data["labels"][val].tolist())
        val_scores.extend(scores_for(model, data, val, cfg, device).tolist())
        histories.append({"subject": person, "fit": fit_indices.tolist(), "val": val.tolist(),
                          "selected_epoch": epoch, "history": history})
    choices = original.select_thresholds(np.asarray(val_labels), np.asarray(val_scores), cfg)
    refit_seed = seed + outer * 1000 + 999
    selected = int(np.median(epochs))
    model, _, refit = fit(variant, data, train, None, cfg, refit_seed, device, selected)
    path = output / "model.pt"
    metadata = dict(variant=variant, dropout=cfg["dropout"], feature_version=FEATURE_VERSION,
                    seed=seed, subject=outer, refit_seed=refit_seed, epochs=selected,
                    train=train.tolist(), test=test.tolist())
    torch.save(dict(metadata, state={k: v.cpu() for k, v in model.state_dict().items()}), path)
    scores = scores_for(model, data, test, cfg, device)
    np.testing.assert_allclose(scores_for(load_model(path, device), data, test, cfg, device),
                               scores, rtol=1e-6, atol=1e-7)
    rows = []
    for policy, choice in choices.items():
        threshold = choice["threshold"]
        for idx, score in zip(test, scores):
            rows.append(dict(sample_index=int(idx), video_id=str(data["video_ids"][idx]),
                             subject=outer, label=int(data["labels"][idx]), score=float(score),
                             prediction=int(score >= threshold), threshold=threshold,
                             variant=variant, seed=seed, policy=policy))
    original.write_json(output / "fold.json", dict(
        signature=signature, metadata=metadata, inner=histories, refit=refit,
        choices=choices, inner_labels=val_labels, inner_scores=val_scores,
        predictions=rows, checkpoint_sha256=sha256(path)))


def validate_fold(folder, variant, data, outer, seed, cfg, device, signature):
    record = read_json(folder / "fold.json")
    train, test, inner = splits(data, outer)
    history = record["inner"]
    if record["signature"] != signature or len(history) != len(inner):
        raise ValueError("Fold signature or inner-fit count mismatch")

    def check_history(entries, limit):
        if not 1 <= len(entries) <= limit or [e["epoch"] for e in entries] != list(range(1, len(entries) + 1)):
            raise ValueError("Invalid epoch history")
        if any(not np.isfinite(list(e.values())).all() for e in entries):
            raise ValueError("Nonfinite epoch history")

    epochs, labels = [], []
    for saved, (person, fit_indices, val) in zip(history, inner):
        if (saved["subject"], saved["fit"], saved["val"]) != (person, fit_indices.tolist(), val.tolist()):
            raise ValueError("Inner split mismatch")
        entries = saved["history"]
        check_history(entries, cfg["max_epochs"])
        best, selected = float("inf"), 0
        for entry in entries:
            if entry["validation_unweighted_loss"] < best - 1e-6:
                best, selected = entry["validation_unweighted_loss"], entry["epoch"]
        if saved["selected_epoch"] != selected:
            raise ValueError("Inner epoch selection mismatch")
        epochs.append(selected)
        labels.extend(data["labels"][val].tolist())
    epoch = int(np.median(epochs))
    check_history(record["refit"], epoch)
    if len(record["refit"]) != epoch:
        raise ValueError("Incomplete refit history")
    metadata = dict(variant=variant, dropout=cfg["dropout"], feature_version=FEATURE_VERSION,
                    seed=seed, subject=outer, refit_seed=seed + outer * 1000 + 999,
                    epochs=epoch, train=train.tolist(), test=test.tolist())
    if record["metadata"] != metadata:
        raise ValueError("Fold metadata mismatch")
    scores = np.asarray(record["inner_scores"])
    if (labels != record["inner_labels"] or scores.shape != (len(labels),)
            or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1))):
        raise ValueError("Invalid inner predictions")
    choices = original.select_thresholds(np.asarray(labels), scores, cfg)
    if json.loads(json.dumps(choices)) != record["choices"]:
        raise ValueError("Threshold selection mismatch")
    path = folder / "model.pt"
    checksum = sha256(path)
    if checksum != record["checkpoint_sha256"]:
        raise ValueError("Checkpoint SHA-256 mismatch")
    scores = scores_for(load_model(path, device, metadata), data, test, cfg, device)
    rows = record["predictions"]
    if [(r["policy"], r["sample_index"]) for r in rows] != [(p, int(i)) for p in choices for i in test]:
        raise ValueError("Fold prediction coverage mismatch")
    for row, score in zip(rows, np.tile(scores, len(choices))):
        idx = row["sample_index"]
        threshold = choices[row["policy"]]["threshold"]
        expected = (variant, seed, outer, str(data["video_ids"][idx]), int(data["labels"][idx]), threshold)
        actual = tuple(row[k] for k in ("variant", "seed", "subject", "video_id", "label", "threshold"))
        if (actual != expected or not 0 <= row["score"] <= 1
                or not np.isclose(row["score"], score, rtol=1e-6, atol=1e-7)
                or row["prediction"] != int(row["score"] >= threshold)
                or row["prediction"] != int(score >= threshold)):
            raise ValueError("Fold predictions do not match checkpoint or dataset")
    return rows, dict(metadata, file=f"folds/{folder.name}/model.pt",
                      sha256=checksum, bytes=path.stat().st_size)



def summarize(rows, data, seeds, outers):
    frame = pd.DataFrame(rows)
    keys = ["variant", "seed", "policy"]
    expected = {(v, s, p) for v in VARIANTS for s in seeds for p in POLICIES}
    groups = frame.groupby(keys)
    if set(groups.groups) != expected:
        raise ValueError("Incorrect variant/seed/policy coverage")
    target = np.flatnonzero(np.isin(data["subjects"], outers))
    for _, group in groups:
        indices = group.sample_index.to_numpy()
        if not np.array_equal(np.sort(indices), target):
            raise ValueError("Incorrect sample coverage")
        for column, source in (("video_id", "video_ids"), ("label", "labels"), ("subject", "subjects")):
            if not np.array_equal(group[column].to_numpy(), data[source][indices]):
                raise ValueError(f"Prediction alignment mismatch: {column}")
        if not np.isfinite(group[["score", "threshold"]]).all().all():
            raise ValueError("Nonfinite prediction or threshold")
        if not np.array_equal(group.prediction, (group.score >= group.threshold).astype(int)):
            raise ValueError("Prediction does not match threshold")
    for _, group in frame.groupby(["variant", "seed", "sample_index"]):
        if group.score.nunique() != 1:
            raise ValueError("Policies must reuse the same score")

    def table(columns):
        return pd.DataFrame([dict(zip(columns, key), **original.metrics(group.label, group.prediction))
                             for key, group in frame.groupby(columns)])

    return frame, table(keys + ["subject"]), table(keys)


def protocol_lock(data, mode, device, synthetic):
    cfg = original.default_config(mode)
    cfg["variants"] = list(VARIANTS)
    if mode == "smoke":
        cfg["patience"] = 10
    seeds = cfg["seeds"]
    outers = [1] if mode == "smoke" else [1, 2, 3, 4]
    sources = [*Path(__file__).parent.glob("*.py"),
               *(ROOT / "training/component_ablation").glob("*.py"),
               *(Path(__file__).parent.parent).glob("*.py"),
               ROOT / "docs/research/skateformer-addendum.md"]
    lock = dict(config=cfg, outer_subjects=outers, primary_policy="fixed",
                primary_metric="accuracy", synthetic=synthetic,
                inputs={k: v["sha256"] for k, v in data.get("provenance", {}).items()},
                code={p.relative_to(ROOT).as_posix(): hashlib.sha256(
                    p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in sources},
                device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                python=platform.python_version(), torch=str(torch.__version__),
                numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__)
    lock["format_version"] = 2
    lock["cuda"] = torch.version.cuda
    if synthetic:
        lock["fixture_hash"] = {
            k: hashlib.sha256(np.asarray(data[k]).tobytes()).hexdigest()
            for k in ("poses", "masks", "durations", "labels", "subjects", "video_ids")
        }
    return lock


def fold_inventory(lock):
    return [(v, seed, outer, f"{v}_seed{seed}_subject{outer}")
            for v in VARIANTS for seed in lock["config"]["seeds"]
            for outer in lock["outer_subjects"]]


def collect(data, lock, device, output, allow_missing=False):
    expected = fold_inventory(lock)
    root = output / "folds"
    actual = {p.name for p in root.iterdir() if not p.name.startswith(".pending-")}
    names = {tag for _, _, _, tag in expected}
    if actual - names or (not allow_missing and actual != names):
        raise ValueError("Unexpected or missing seed/fold in run")
    verified = {}
    for variant, seed, outer, tag in expected:
        if tag in actual:
            verified[tag] = validate_fold(root / tag, variant, data, outer, seed,
                                          lock["config"], device, signature_for(lock))
    return verified


def verify_smoke(data, device, output):
    if device.type != "cuda":
        raise ValueError("Full requires matching CUDA smoke evidence")
    if {k: v["sha256"] for k, v in data.get("provenance", {}).items()} != expected_inputs():
        raise ValueError("Smoke evidence requires the four verified real inputs")
    lock = protocol_lock(data, "smoke", device, False)
    if read_json(output / "locked_protocol.json") != lock:
        raise ValueError("Smoke data/code/config/runtime mismatch")
    folds = collect(data, lock, device, output)
    validate_summary(data, lock, output, folds)


def validate_summary(data, lock, output, folds):
    rows = [row for predictions, _ in folds.values() for row in predictions]
    cfg = lock["config"]
    frame, _, _ = summarize(rows, data, cfg["seeds"], lock["outer_subjects"])
    saved = pd.read_csv(output / "predictions.csv", dtype={"video_id": str}, float_precision="round_trip")
    pd.testing.assert_frame_equal(saved, frame, check_exact=True)
    if read_json(output / "models/manifest.json") != [artifact for _, artifact in folds.values()]:
        raise ValueError("Smoke checkpoint manifest mismatch")
    done = read_json(output / "completion.json")
    count = len(folds)
    mode, synthetic = cfg["mode"], lock["synthetic"]
    expected = dict(mode=mode, complete=True, synthetic=synthetic, scientific_result=mode == "full" and not synthetic,
                    outer_runs=count, inner_fits=3 * count, refits=count, expected_fits=4 * count, actual_fits=4 * count,
                    reload="PASS", coverage="PASS", signature=signature_for(lock))
    if any(done.get(k) != v for k, v in expected.items()):
        raise ValueError("Completion metadata mismatch")
    return done


def run(data, mode, device, output, synthetic=False, resume=False, smoke_evidence=None):
    if synthetic and mode != "smoke":
        raise ValueError("Synthetic inputs are limited to software tests")
    if not synthetic and {k: v["sha256"] for k, v in data.get("provenance", {}).items()} != expected_inputs():
        raise ValueError("Run requires the four verified real inputs")
    if mode == "full":
        if smoke_evidence is None:
            raise ValueError("Full requires completed real smoke evidence")
        verify_smoke(data, device, smoke_evidence)
    lock = protocol_lock(data, mode, device, synthetic)
    signature = signature_for(lock)
    cfg, outers = lock["config"], lock["outer_subjects"]
    seeds = cfg["seeds"]
    if resume:
        if read_json(output / "locked_protocol.json") != lock:
            raise ValueError("Resume data/code/config/runtime mismatch")
    else:
        output.mkdir(parents=True, exist_ok=False)
        original.write_json(output / "locked_protocol.json", lock)
    (output / "folds").mkdir(exist_ok=True)
    finalized = (output / "completion.json").exists()
    verified = collect(data, lock, device, output, allow_missing=not finalized)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    rows, manifest = [], []
    for variant, seed, outer, tag in fold_inventory(lock):
        target = output / "folds" / tag
        if tag not in verified:
            with tempfile.TemporaryDirectory(prefix=".pending-", dir=target.parent) as tmp:
                staged = Path(tmp) / tag
                staged.mkdir()
                run_fold(variant, data, outer, seed, cfg, device, staged, signature)
                verified[tag] = validate_fold(staged, variant, data, outer, seed, cfg, device, signature)
                staged.rename(target)
        predictions, artifact = verified[tag]
        rows.extend(predictions)
        manifest.append(artifact)
    if resume and finalized:
        return validate_summary(data, lock, output, verified)
    (output / "models").mkdir(exist_ok=True)
    frame, folds, pooled = summarize(rows, data, seeds, outers)
    expected_outer = len(VARIANTS) * len(seeds) * len(outers)
    if len(manifest) != expected_outer:
        raise ValueError("Incomplete checkpoint coverage")
    original.write_json(output / "models/manifest.json", manifest)
    for name, table in (("predictions", frame), ("fold_metrics", folds),
                        ("pooled_by_seed", pooled), ("errors", frame[frame.label != frame.prediction])):
        table.to_csv(output / f"{name}.csv", index=False)
    if mode == "full":
        metrics = [c for c in pooled if c not in {"variant", "seed", "policy"}]
        grouped = pooled.groupby(["variant", "policy"])[metrics]
        grouped.agg(["mean", "std", "count"]).to_csv(output / "seed_mean_std.csv")
    completion = dict(mode=mode, complete=True, synthetic=synthetic,
                      scientific_result=mode == "full" and not synthetic,
                      signature=signature, outer_runs=len(manifest), inner_fits=3 * len(manifest),
                      refits=len(manifest), expected_fits=4 * expected_outer,
                      actual_fits=4 * len(manifest), reload="PASS", coverage="PASS",
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(device)
                      if device.type == "cuda" else None)
    pending = output / ".completion.pending.json"
    original.write_json(pending, completion)
    pending.replace(output / "completion.json")
    return validate_summary(data, lock, output, verified)


def main():
    parser = argparse.ArgumentParser(description="SkateFormer supplemental nested LOSO comparison")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--smoke-evidence", type=Path, help="Completed real smoke directory for full mode")
    parser.add_argument("--resume", action="store_true", help="Resume a run with the same lock")
    args = parser.parse_args()
    data = load_verified(args.input_root)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable for this Python environment")
    print(run(data, args.mode, torch.device(args.device), args.output,
              resume=args.resume, smoke_evidence=args.smoke_evidence))


if __name__ == "__main__":
    main()
