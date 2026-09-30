import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from research07.features import CHANNELS, augment_pose, load_inputs, make_features_v2
from research07.models import PoseDataset, SkeletonModel
from research07.training import VARIANTS, default_config, metrics, select_thresholds, write_json


ROOT = Path(__file__).resolve().parents[1]


def validate(input_root):
    data = load_inputs(input_root)
    with np.load(data["provenance"]["features"]["path"], allow_pickle=False) as archive:
        joint = np.stack([archive["skeleton_features"][..., 0], archive["skeleton_features"][..., 1], archive["skeleton_features"][..., 3], archive["motion_features"][..., 0], archive["motion_features"][..., 1]], axis=-1)
        globals_array = archive["global_features"]
        hip = globals_array[..., :2] - globals_array[:, :1, :2]
        body = np.concatenate([globals_array[..., 9:10], hip], axis=-1)
        legacy = np.concatenate([joint, np.repeat(body[:, :, None], 33, axis=2)], axis=-1).transpose(0, 3, 1, 2)
    rebuilt = np.stack([make_features_v2(pose, mask, duration, False) for pose, mask, duration in zip(data["poses"], data["masks"], data["durations"])])
    parity = float(np.max(np.abs(rebuilt - legacy)))
    assert parity < 2e-5, parity
    corrected = np.stack([make_features_v2(pose, mask, duration) for pose, mask, duration in zip(data["poses"], data["masks"], data["durations"])])
    assert np.array_equal(corrected[:, [0, 1, 3, 4, 5, 6, 7]], rebuilt[:, [0, 1, 3, 4, 5, 6, 7]])
    assert np.all(corrected[:, 2][~data["masks"]] == 0)
    assert np.array_equal(corrected[:, 2][data["masks"]], rebuilt[:, 2][data["masks"]])
    rng = np.random.default_rng(42)
    for iteration in range(100):
        pose, mask, duration, dropped = augment_pose(data["poses"][iteration], data["masks"][iteration], float(data["durations"][iteration]), rng)
        features = make_features_v2(pose, mask, duration)
        expected = np.diff(features[:2], axis=1) / (duration / 31)
        assert np.allclose(features[3:5, 1:], expected, atol=2e-5)
        assert np.all(features[3:5, 0] == 0)
        assert np.all(features[2][~mask] == 0)
        assert np.allclose(features[5:], features[5:, :, :1])
        assert np.isfinite(features).all()
        features[:5, :, dropped] = 0
        assert np.allclose(features[5:], features[5:, :, :1])
    translated = data["poses"][0].copy()
    translated[..., :2] += np.array([0.03, -0.04], dtype=np.float32)
    translated_features = make_features_v2(translated, data["masks"][0], data["durations"][0])
    assert np.allclose(translated_features, corrected[0], atol=2e-5)
    tensor = torch.from_numpy(corrected[:2])
    for name, variant in VARIANTS.items():
        model = SkeletonModel(variant["channels"], variant["attention"])
        model.eval()
        with torch.no_grad():
            logits, attention = model(tensor[:, :variant["channels"]])
        assert logits.shape == (2, 2) and attention.shape == (2, 16)
        assert torch.allclose(attention.sum(dim=1), torch.ones(2), atol=1e-6)
    assert metrics([0, 0, 1, 1], [0, 1, 0, 1])["accuracy"] == 0.5
    selection = select_thresholds(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.3, 0.9]), default_config())
    assert "sensitivity" in selection and selection["sensitivity"]["feasible"] is False
    affected = ~data["masks"]
    report = {"passed": True, "samples": len(data["poses"]), "legacy_parity_max_error": parity,
              "missing_sampled_frames": int(affected.sum()), "affected_videos": int(affected.any(axis=1).sum()),
              "old_mean_visibility_on_missing": float(rebuilt[:, 2][affected].mean()),
              "v2_mean_visibility_on_missing": float(corrected[:, 2][affected].mean()),
              "augmentation_consistency_cases": 100, "model_variants_shape_checked": list(VARIANTS),
              "note": "Legacy source has already interpolated 32 poses; v2 cannot restore original raw missing spans. No sample is dropped from LOSO."}
    output = ROOT / ".runtime/validation"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "pipeline_validation.json", report)
    print(json.dumps(report, indent=2))
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=ROOT / "data/private")
    torch.set_num_threads(2)
    validate(parser.parse_args().input_root)
