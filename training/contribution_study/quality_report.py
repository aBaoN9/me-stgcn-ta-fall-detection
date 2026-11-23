"""Pre-full data check for Q1: distribution of the visibility quality q on the real, hash-locked inputs.

Run after the input guard and before any outer-test result exists. It only describes inputs (no model,
no predictions, no outer scores) and does not change the locked M0 (epsilon = 0.1 is fixed). If q' is
nearly constant within most clips, Q1 is expected to stay close to Q0; that is reported, not tuned away.
Labels are used only to describe how missing pose is spread over Fall/ADL, never to pick a setting.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from training.component_ablation.features import load_inputs, make_features_v2
from training.component_ablation.training import write_json
from training.contribution_study.quality_pooling import EPSILON, frame_quality, temporal_quality
from training.contribution_study.runner import make_config, verify_inputs

STEPS = 16  # attention steps after the stride-2 block of A3
FLAT = 0.01  # within-clip std of q' below this means the prior is effectively constant for that clip


def quality_table(data):
    features = np.stack([make_features_v2(pose, mask, duration) for pose, mask, duration
                         in zip(data["poses"], data["masks"], data["durations"])])
    quality = frame_quality(torch.from_numpy(features))
    steps = temporal_quality(quality, STEPS).numpy()
    prior = np.log(EPSILON + steps)
    weight = np.exp(prior - prior.max(axis=1, keepdims=True))  # relative multiplier the prior applies per step
    return pd.DataFrame({
        "sample_index": np.arange(len(data["labels"])), "video_id": data["video_ids"].astype(str),
        "subject": data["subjects"], "label": data["labels"],
        "missing_frames": (~data["masks"]).sum(axis=1),
        "q_frame_mean": quality.numpy().mean(axis=1), "q_frame_min": quality.numpy().min(axis=1),
        "q_step_min": steps.min(axis=1), "q_step_max": steps.max(axis=1), "q_step_std": steps.std(axis=1),
        "prior_min_relative_weight": weight.min(axis=1),
    })


def summarize(table):
    describe = lambda column: {key: float(value) for key, value in
                               table[column].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).items()}
    return {
        "clips": int(len(table)), "epsilon": EPSILON, "flat_threshold_std": FLAT,
        "clips_with_missing_frames": int((table["missing_frames"] > 0).sum()),
        "missing_frames_total": int(table["missing_frames"].sum()),
        "clips_flat_prior": int((table["q_step_std"] < FLAT).sum()),
        "q_frame_mean": describe("q_frame_mean"), "q_step_std": describe("q_step_std"),
        "prior_min_relative_weight": describe("prior_min_relative_weight"),
        "by_label": table.groupby("label")[["missing_frames", "q_frame_mean", "q_step_std"]].mean().to_dict(),
        "by_subject": table.groupby("subject")[["missing_frames", "q_frame_mean", "q_step_std"]].mean().to_dict(),
        "note": "Input description only; no model output. Does not change epsilon or the locked M0.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = load_inputs(args.input_root)
    guard = verify_inputs(data, make_config("full"))
    table = quality_table(data)
    args.output.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output / "quality_by_clip.csv", index=False)
    summary = {"input_guard": guard, "inputs": {k: v["sha256"] for k, v in data["provenance"].items()}, **summarize(table)}
    write_json(args.output / "quality_summary.json", summary)
    print({key: summary[key] for key in ["input_guard", "clips_with_missing_frames", "clips_flat_prior"]})
