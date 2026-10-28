"""Software checks for DUC-02/03 on synthetic tensors: shape after stride, prior-off parity, all-missing fallback,
mask/augmentation consistency, gradients, and an end-to-end runner smoke on fake data with OOF/metric recompute.
Synthetic results are not experimental evidence. Report: .runtime/contribution_study/quality_checks.json
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from training.component_ablation.features import augment_pose, make_features_v2
from training.component_ablation.models import SkeletonModel, set_seed
from training.contribution_study.audit_existing import metrics as audit_metrics
from training.contribution_study.quality_pooling import EPSILON, QualityAwareSkeletonModel, frame_quality, temporal_quality
from training.contribution_study.runner import make_config, run_experiment

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / ".runtime/contribution_study"


def synthetic_pose(rng, missing=()):
    pose = rng.uniform(0.2, 0.8, (32, 33, 4)).astype(np.float32)
    pose[..., 3] = rng.uniform(0.3, 1.0, (32, 33))
    detected = np.ones(32, dtype=bool)
    detected[list(missing)] = False
    return pose, detected


def synthetic_data(rng):
    subjects = np.repeat([1, 2, 3, 4], 40)
    labels = np.zeros(160, dtype=int)
    for subject in range(4):
        labels[subject * 40: subject * 40 + (20 if subject else 19)] = 1
    poses, masks = zip(*[synthetic_pose(rng, rng.choice(32, int(rng.integers(0, 6)), replace=False)) for _ in range(160)])
    poses = np.stack(poses)
    poses[labels == 1, 16:, :, 1] += 0.15
    return {"poses": poses, "masks": np.stack(masks), "durations": rng.uniform(1.5, 4.0, 160).astype(np.float32),
            "labels": labels, "subjects": subjects, "video_ids": np.array([f"SYN_{i:03d}" for i in range(160)]),
            "provenance": {"synthetic": {"path": "generated", "sha256": "synthetic-seed-7"}}, "synthetic": True}


def main():
    torch.set_num_threads(2)
    rng = np.random.default_rng(7)
    checks = {}
    batch = torch.from_numpy(np.stack([make_features_v2(*synthetic_pose(rng, [3, 4, 5]), 2.0) for _ in range(4)]))

    set_seed(0)
    base = SkeletonModel(8, True).eval()
    set_seed(0)
    q0 = QualityAwareSkeletonModel(8, True, quality_prior=False).eval()
    set_seed(0)
    q1 = QualityAwareSkeletonModel(8, True, quality_prior=True).eval()
    same_init = all(torch.equal(a, b) for a, b in zip(base.state_dict().values(), q1.state_dict().values()))
    checks["same_init_and_parameter_count"] = same_init and sum(p.numel() for p in base.parameters()) == sum(p.numel() for p in q1.parameters())
    with torch.no_grad():
        logits_base, weights_base = base(batch)
        logits_q0, weights_q0 = q0(batch)
        logits_q1, weights_q1 = q1(batch)
    checks["prior_off_equals_SkeletonModel"] = torch.equal(logits_base, logits_q0) and torch.equal(weights_base, weights_q0)
    checks["shape_after_stride"] = tuple(logits_q1.shape) == (4, 2) and tuple(weights_q1.shape) == (4, 16) \
        and torch.allclose(weights_q1.sum(1), torch.ones(4), atol=1e-6)

    quality = frame_quality(batch)
    checks["frame_quality_zero_on_undetected"] = quality.shape == (4, 32) and bool((quality[:, [3, 4, 5]] == 0).all()) \
        and bool((quality[:, [0, 1, 2, 6]] > 0).all())
    mapped = temporal_quality(torch.arange(32, dtype=torch.float32)[None], 16)[0]
    centred = torch.tensor([0.5] + [float(2 * t) for t in range(1, 16)])
    checks["stride_mapping_centred"] = torch.allclose(mapped, centred)
    prior = EPSILON + temporal_quality(quality, 16)
    reweighted = weights_q0 * prior
    checks["weights_equal_normalised_prior_product"] = torch.allclose(weights_q1, reweighted / reweighted.sum(1, keepdim=True), atol=1e-6)

    constant = batch.clone()
    constant[:, 2] = 0.6
    missing = batch.clone()
    missing[:, 2] = 0
    with torch.no_grad():
        checks["constant_quality_equals_q0"] = torch.allclose(q1(constant)[1], q0(constant)[1], atol=1e-6)
        all_missing_logits, all_missing_weights = q1(missing)
        checks["all_missing_finite_and_falls_back_to_attention"] = bool(torch.isfinite(all_missing_logits).all()) \
            and torch.allclose(all_missing_weights, q0(missing)[1], atol=1e-6)

    augmented_ok = True
    for _ in range(200):
        pose, detected = synthetic_pose(rng, rng.choice(32, int(rng.integers(0, 8)), replace=False))
        pose, detected, duration, dropped = augment_pose(pose, detected, 2.0, rng)
        features = make_features_v2(pose, detected, duration)
        features[:5, :, dropped] = 0
        q = frame_quality(torch.from_numpy(features)[None])[0].numpy()
        visibility = np.clip(pose[..., 3], 0, 1) * detected[:, None]
        visibility[:, dropped] = 0
        augmented_ok &= np.allclose(q, visibility.mean(1), atol=1e-6) and np.all(q[~detected] == 0)
    checks["quality_follows_augmented_mask_and_dropped_joints"] = bool(augmented_ok)

    train_model = QualityAwareSkeletonModel(8, True).train()
    logits, _ = train_model(batch)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 0, 1])).backward()
    grads = [p.grad for p in train_model.parameters()]
    checks["gradients_finite_and_reach_attention"] = all(g is not None and torch.isfinite(g).all() for g in grads) \
        and float(train_model.temporal_attention[0].weight.grad.abs().sum()) > 0

    smoke_dir = OUTPUT / "synthetic_smoke"
    shutil.rmtree(smoke_dir, ignore_errors=True)
    data = synthetic_data(np.random.default_rng(11))
    config = make_config("smoke")
    run_experiment(data, config, smoke_dir)
    completion = json.loads((smoke_dir / "completion.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(smoke_dir / "predictions.csv")
    pooled = pd.read_csv(smoke_dir / "pooled_by_seed.csv")
    recomputed = predictions.groupby(["variant", "seed", "policy"]).apply(
        lambda g: pd.Series(audit_metrics(g.label, g.prediction)), include_groups=False).reset_index()
    merged = pooled.merge(recomputed, on=["variant", "seed", "policy"], suffixes=("", "_re"))
    checks["runner_smoke_coverage_and_counts"] = completion["outer_runs"] == completion["outer_runs_expected"] == 8 \
        and completion["fits"] == completion["fits_expected"] == 32 and completion["scientific_result"] is False \
        and not predictions.duplicated(["variant", "seed", "policy", "sample_index"]).any() \
        and (predictions.groupby(["variant", "seed", "policy"]).sample_index.nunique() == 160).all()
    checks["runner_smoke_metrics_recomputed"] = len(merged) == len(pooled) and all(
        np.allclose(merged[c], merged[f"{c}_re"]) for c in ["accuracy", "sensitivity", "specificity", "tp", "tn", "fp", "fn"])
    reports = [json.loads(p.read_text(encoding="utf-8")) for p in smoke_dir.glob("Q*_seed*_subject*.json")]
    checks["runner_no_subject_leakage"] = all(
        not set(s["train_indices"]) & set(s["val_indices"]) for r in reports for s in r["inner_splits"]) \
        and all(r["predictions"][0]["subject"] not in {data["subjects"][i] for s in r["inner_splits"] for i in s["train_indices"]} for r in reports)

    checks = {key: bool(value) for key, value in checks.items()}
    status = "PASS" if all(checks.values()) else "FAIL"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = {"status": status, "checks": checks, "epsilon": EPSILON, "torch": torch.__version__,
              "note": "Synthetic tensors and fake labels; software check only, not a result."}
    (OUTPUT / "quality_checks.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for key, value in checks.items():
        print("PASS" if value else "FAIL", key)
    print("OVERALL", status)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
