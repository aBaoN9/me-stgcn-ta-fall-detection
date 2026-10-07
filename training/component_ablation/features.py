import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


CHANNELS = ["relative_x", "relative_y", "visibility", "joint_velocity_x", "joint_velocity_y",
            "torso_angle", "hip_displacement_x", "hip_displacement_y"]
PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
         (17, 18), (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]
FEATURE_VERSION = "v2-mask-visibility-raw-pose-augmentation"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_unique(root, names):
    matches = sorted({path for name in names for path in Path(root).rglob(name)})
    if not matches:
        raise FileNotFoundError(f"Missing input: {names}; attach notebook 01 outputs")
    hashes = {sha256(path) for path in matches}
    if len(hashes) != 1:
        raise ValueError(f"Conflicting copies; remove ambiguous inputs: {matches}")
    return matches[0]


def load_inputs(root):
    paths = {
        "pose": find_unique(root, ["gmdcsa24_pose_32.npz", "gmdcsa24_pose_32.zip"]),
        "features": find_unique(root, ["gmdcsa24_features_32.npz", "gmdcsa24_features_32.zip"]),
        "manifest": find_unique(root, ["gmdcsa24_pose_manifest.csv", "pose_mnf.txt"]),
        "folds": find_unique(root, ["gmdcsa24_cross_subject_folds.csv", "cross.txt"]),
    }
    with np.load(paths["pose"], allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    with np.load(paths["features"], allow_pickle=False) as archive:
        features = {key: archive[key].copy() for key in archive.files}
    for key in ["video_ids", "subjects", "labels", "masks"]:
        if not np.array_equal(data[key], features[key]):
            raise ValueError(f"Pose/feature alignment failed: {key}")
    manifest = pd.read_csv(paths["manifest"])
    folds = pd.read_csv(paths["folds"])
    for column, key in [("video_id", "video_ids"), ("subject", "subjects"), ("label_id", "labels")]:
        if not np.array_equal(manifest[column].to_numpy(), data[key]):
            raise ValueError(f"Manifest order mismatch: {column}")
    count = len(data["labels"])
    if count != 160 or data["poses"].shape != (count, 32, 33, 4):
        raise ValueError("This preregistered experiment expects 160 poses of shape (32,33,4)")
    if len(np.unique(data["video_ids"])) != count or set(np.unique(data["subjects"])) != {1, 2, 3, 4}:
        raise ValueError("Duplicate IDs or unexpected subjects")
    if set(np.unique(data["labels"])) != {0, 1} or data["masks"].shape != (count, 32):
        raise ValueError("Invalid labels or masks")
    if not np.isfinite(data["poses"]).all() or not np.isfinite(features["durations"]).all() or np.any(features["durations"] <= 0):
        raise ValueError("Nonfinite pose or invalid duration")
    if len(folds) != count * 4 or folds["fold"].nunique() != 4:
        raise ValueError("Saved split file does not have four folds")
    for _, group in folds.groupby("fold"):
        indices = group["sample_index"].to_numpy(dtype=int)
        if not np.array_equal(np.sort(indices), np.arange(count)):
            raise ValueError("Invalid saved sample indices")
        if not np.array_equal(group["video_id"].to_numpy(), data["video_ids"][indices]):
            raise ValueError("Saved fold IDs do not match pose order")
        if not np.array_equal(group["subject"].to_numpy(), data["subjects"][indices]) or not np.array_equal(group["label_id"].to_numpy(), data["labels"][indices]):
            raise ValueError("Saved fold subjects or labels mismatch")
        partitions = [set(group.loc[group["split"] == split, "subject"]) for split in ["train", "val", "test"]]
        if any(partitions[left] & partitions[right] for left, right in [(0, 1), (0, 2), (1, 2)]):
            raise ValueError("Subject overlap in saved folds")
    data.update(durations=features["durations"].astype(np.float32), manifest=manifest,
                provenance={key: {"path": str(path), "sha256": sha256(path)} for key, path in paths.items()})
    return data


def make_features_v2(pose, detected, duration, mask_visibility=True):
    pose = np.asarray(pose, dtype=np.float32)
    detected = np.asarray(detected, dtype=bool)
    if pose.shape != (32, 33, 4) or detected.shape != (32,) or not np.isfinite(pose).all():
        raise ValueError("Invalid finite pose/mask shape")
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("Invalid duration")
    hips = pose[:, [23, 24], :2].mean(axis=1)
    shoulders = pose[:, [11, 12], :2].mean(axis=1)
    length = np.linalg.norm(shoulders - hips, axis=-1)
    usable = length[length > 1e-3]
    scale = max(float(np.median(usable)) if len(usable) else 1.0, 1e-3)
    relative = (pose[..., :2] - hips[:, None, :]) / scale
    velocity = np.zeros_like(relative)
    velocity[1:] = np.diff(relative, axis=0) / max(float(duration) / 31, 1e-6)
    visibility = np.clip(pose[..., 3], 0, 1).copy()
    if mask_visibility:
        visibility[~detected] = 0
    torso = shoulders - hips
    angle = np.arctan2(torso[:, 0], -torso[:, 1]) / np.pi
    displacement = hips - hips[:1]
    local = np.concatenate([relative, visibility[..., None], velocity], axis=-1)
    global_values = np.concatenate([angle[:, None], displacement], axis=-1)
    output = np.concatenate([local, np.repeat(global_values[:, None], 33, axis=1)], axis=-1)
    return output.transpose(2, 0, 1).astype(np.float32)


def augment_pose(pose, detected, duration, rng, crop=True, flip=True, noise=True, dropout=True):
    pose, detected = pose.copy(), detected.copy()
    if crop and rng.random() < 0.5:
        length = int(rng.integers(26, 33))
        start = int(rng.integers(0, 33 - length))
        positions = np.linspace(start, start + length - 1, 32)
        lower, upper = np.floor(positions).astype(int), np.ceil(positions).astype(int)
        fraction = (positions - lower).astype(np.float32)[:, None, None]
        pose = pose[lower] * (1 - fraction) + pose[upper] * fraction
        detected = detected[lower] & detected[upper]
        duration *= (length - 1) / 31
    if flip and rng.random() < 0.5:
        pose[..., 0] = 1 - pose[..., 0]
        original = pose.copy()
        for left, right in PAIRS:
            pose[:, left], pose[:, right] = original[:, right], original[:, left]
    if noise and rng.random() < 0.35:
        torso_length = np.linalg.norm(pose[:, [11, 12], :2].mean(axis=1) - pose[:, [23, 24], :2].mean(axis=1), axis=-1)
        amplitude = max(float(np.median(torso_length)), 1e-3) * 0.01
        pose[..., :2] += rng.normal(0, amplitude, pose[..., :2].shape).astype(np.float32)
    dropped = np.empty(0, dtype=int)
    if dropout and rng.random() < 0.2:
        dropped = rng.choice(33, int(rng.integers(1, 4)), replace=False)
    return pose, detected, float(duration), dropped


def svm_features(pose, detected, duration):
    sequence = make_features_v2(pose, detected, duration)
    speed = np.linalg.norm(sequence[3:5], axis=0)
    height = np.ptp(sequence[1], axis=1)
    height_velocity = np.diff(height, prepend=height[0]) / (float(duration) / 31)
    per_frame = np.stack([speed.mean(axis=1), np.quantile(speed, 0.9, axis=1), sequence[5, :, 0],
                          sequence[6, :, 0], sequence[7, :, 0], height, height_velocity], axis=1)
    return np.concatenate([per_frame.mean(axis=0), per_frame.std(axis=0), per_frame.min(axis=0),
                           per_frame.max(axis=0), *np.quantile(per_frame, [0.1, 0.5, 0.9], axis=0)]).astype(np.float32)
