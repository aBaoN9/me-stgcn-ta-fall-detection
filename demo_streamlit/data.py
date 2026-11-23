import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 142, 242)
SKATE = "SkateFormer_adapted_small"
SKATE_COMMIT = "20c67f57c87d1a2cfc0489bf23da45522c4c8d2c"
MODEL_NAMES = {
    "A3_motion_attention": "ME-STGCN-TA · A3",
    "TCN_v2_small": "TCN · Small v2",
    "GRU_v2_small": "GRU · Small v2",
    "CTR_GCN_adapted": "CTR-GCN · Adapted",
    SKATE: "SkateFormer · Adapted small",
}
BASELINE_NAMES = {"Motion_SVM_base": "Motion-SVM · Base", "ST_GCN_base": "ST-GCN · Base (3 kênh)"}
MODEL_CHOICES = {**MODEL_NAMES, **BASELINE_NAMES}
METRICS = ("accuracy", "sensitivity", "specificity", "f1")
BASE = ROOT / "results/backbone_benchmark"
PRIMARY = BASE / "tung-m2-rtx3050-20261120-165523"
SUPPLEMENT = BASE / "tung-gru-ctr-rtx3050-20261120-233932"
SKATE_RUN = BASE / "tung-skate-rtx3050-20261122-205954"
TUNING = ROOT / "results/a3_tuning/giabao-t4x2-20261121-partial"


@dataclass(frozen=True)
class Run:
    key: str
    label: str
    variants: tuple
    checkpoints: bool
    provenance: str


RUNS = {
    "accepted": Run("accepted", "Benchmark da nghiem thu", tuple(MODEL_NAMES), True,
                    "A3/TCN/GRU/CTR-GCN/SkateFormer: accepted RTX3050 runs. SkateFormer: PR #4 merged "
                    "23 November 2026, source commit 20c67f5. Fixed threshold 0.5. No new training."),
    "tuning": Run("tuning", "Gia Bao / tuning T4 - archive PARTIAL", tuple(MODEL_NAMES)[:3], False,
                  "Recomputed from notebook confusion matrices. Weights and raw OOF missing."),
}


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def verified_file(folder, relative, manifest_name):
    manifest = json.loads((folder / manifest_name).read_text(encoding="utf-8"))
    entries = manifest.get("files", manifest)
    entry = entries.get(relative)
    path = folder / relative
    if not isinstance(entry, dict) or not path.is_file():
        raise FileNotFoundError(f"Missing trusted artifact: {relative}")
    if digest(path) != entry["sha256"] or path.stat().st_size != entry["bytes"]:
        raise ValueError(f"Artifact SHA-256/size mismatch: {relative}")
    return path


def source_for(variant):
    if variant not in MODEL_NAMES:
        raise ValueError("Unknown model")
    if variant in ("A3_motion_attention", "TCN_v2_small"):
        return PRIMARY, "m3_files_sha256.json"
    if variant == SKATE:
        return SKATE_RUN, "files_sha256.json"
    return SUPPLEMENT, "files_sha256.json"


def metrics_from_counts(true_positive, true_negative, false_positive, false_negative):
    counts = np.asarray([true_positive, true_negative, false_positive, false_negative], dtype=float)
    if not np.isfinite(counts).all() or np.any(counts < 0) or np.any(counts != counts.astype(int)):
        raise ValueError("Invalid confusion counts")
    if true_positive + false_negative != 79 or true_negative + false_positive != 81:
        raise ValueError("Expected 79 Fall and 81 ADL per seed")
    return {
        "accuracy": (true_positive + true_negative) / 160,
        "sensitivity": true_positive / 79,
        "specificity": true_negative / 81,
        "f1": 2 * true_positive / (2 * true_positive + false_positive + false_negative),
    }


def load_predictions(variant, seed):
    if seed not in SEEDS:
        raise ValueError("Unknown seed")
    folder, manifest = source_for(variant)
    path = verified_file(folder, "predictions.csv", manifest)
    predictions = pd.read_csv(path)
    selected = predictions.loc[(predictions.variant == variant) & (predictions.seed == seed)
                               & (predictions.policy == "fixed")].copy()
    if len(selected) != 160 or selected.video_id.nunique() != 160:
        raise ValueError("Incomplete OOF coverage")
    if set(selected.subject) != {1, 2, 3, 4} or not (selected.threshold == 0.5).all():
        raise ValueError("Unexpected subjects or threshold")
    if not np.isfinite(selected.score).all() or not selected.score.between(0, 1).all():
        raise ValueError("Invalid archived scores")
    if not np.array_equal(selected.prediction.to_numpy(), (selected.score >= 0.5).astype(int)):
        raise ValueError("Decision/score mismatch")
    return selected.sort_values("sample_index").reset_index(drop=True)


def load_metrics(run_key):
    run = RUNS[run_key]
    if run.checkpoints:
        records = []
        for variant in run.variants:
            for seed in SEEDS:
                predictions = load_predictions(variant, seed)
                label, decision = predictions.label, predictions.prediction
                counts = dict(tp=int(((label == 1) & (decision == 1)).sum()),
                              tn=int(((label == 0) & (decision == 0)).sum()),
                              fp=int(((label == 0) & (decision == 1)).sum()),
                              fn=int(((label == 1) & (decision == 0)).sum()))
                records.append(dict(variant=variant, seed=seed, **counts))
        table = pd.DataFrame(records)
    else:
        table = pd.read_csv(verified_file(TUNING, "metrics/confusion_matrices.csv", "files_sha256.json"))
    identities = {(variant, seed) for variant in run.variants for seed in SEEDS}
    if set(zip(table.variant, table.seed)) != identities or len(table) != len(identities):
        raise ValueError("Invalid model/seed inventory")
    computed = [metrics_from_counts(row.tp, row.tn, row.fp, row.fn) for row in table.itertuples()]
    return pd.concat([table.reset_index(drop=True).drop(columns=list(METRICS), errors="ignore"),
                      pd.DataFrame(computed)], axis=1)


def load_canonical_data():
    from training.component_ablation.features import load_inputs

    folder = ROOT / "data/gmdcsa24"
    manifest = json.loads((folder / "input_manifest.json").read_text(encoding="utf-8"))
    expected = json.loads((ROOT / "results/component_ablation/locked_protocol.json")
                          .read_text(encoding="utf-8"))["inputs"]
    for entry in manifest["files"]:
        path = folder / entry["filename"]
        if digest(path) != expected[entry["role"]]:
            raise ValueError(f"Canonical input hash mismatch: {entry['role']}")
    return load_inputs(folder)


def checkpoint_for(run_key, variant, seed, subject):
    if not RUNS[run_key].checkpoints:
        raise FileNotFoundError("Tuning checkpoints were not recovered; inference is unavailable")
    if variant not in RUNS[run_key].variants or seed not in SEEDS or subject not in (1, 2, 3, 4):
        raise ValueError("Unknown checkpoint identity")
    folder, manifest = source_for(variant)
    return verified_file(folder, f"folds/{variant}_seed{seed}_subject{subject}/model.pt", manifest)


def load_checkpoint_model(run_key, variant, seed, subject, data=None):
    import torch

    from training.backbone_benchmark.models import build_model
    from training.backbone_benchmark.supplement.models import build_model as build_supplement
    from training.backbone_benchmark.skate.models import build_model as build_skate
    from training.component_ablation.features import FEATURE_VERSION

    path = checkpoint_for(run_key, variant, seed, subject)
    if data is None:
        data = load_canonical_data()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    expected_train = np.flatnonzero(data["subjects"] != subject).tolist()
    expected_test = np.flatnonzero(data["subjects"] == subject).tolist()
    if (checkpoint["variant"] != variant or checkpoint["seed"] != seed
            or checkpoint["subject"] != subject or checkpoint["feature_version"] != FEATURE_VERSION
            or checkpoint["train"] != expected_train or checkpoint["test"] != expected_test):
        raise ValueError("Checkpoint identity/held-out split mismatch")
    torch.set_num_threads(2)
    if variant in ("A3_motion_attention", "TCN_v2_small"):
        builder = build_model
    else:
        builder = build_skate if variant == SKATE else build_supplement
    model = builder(variant, checkpoint["dropout"])
    model.load_state_dict(checkpoint["state"], strict=True)
    model.eval()
    identity = {"subject": subject, "seed": seed, "variant": variant,
                "checkpoint": str(path.relative_to(ROOT)), "sha256": digest(path)}
    if variant == SKATE:
        identity.update(source_commit=SKATE_COMMIT, review_status="accepted_pr4_merged")
    return model, identity


def replay(run_key, variant, seed, sample_index, data):
    import torch

    from training.backbone_benchmark.models import logits_of
    from training.component_ablation.features import make_features_v2

    if not isinstance(sample_index, (int, np.integer)) or not 0 <= sample_index < 160:
        raise ValueError("Invalid sample index")
    subject = int(data["subjects"][sample_index])
    model, identity = load_checkpoint_model(run_key, variant, seed, subject, data)
    features = make_features_v2(data["poses"][sample_index], data["masks"][sample_index],
                                data["durations"][sample_index])
    with torch.inference_mode():
        score = float(logits_of(model, torch.from_numpy(features[None])).softmax(1)[0, 1])
    if not np.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Invalid model output")
    return {"score": score, "prediction": int(score >= 0.5), **identity}
