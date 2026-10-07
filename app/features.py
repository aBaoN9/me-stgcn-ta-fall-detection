import numpy as np


CHANNELS = [
    "relative_x", "relative_y", "visibility", "joint_velocity_x",
    "joint_velocity_y", "torso_angle", "hip_displacement_x", "hip_displacement_y",
]
EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]


def make_features(poses, duration):
    poses = np.asarray(poses, dtype=np.float32)
    if poses.shape != (32, 33, 4) or not np.isfinite(poses).all():
        raise ValueError("Pose phải hữu hạn và có shape (32,33,4).")
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("Thời lượng không hợp lệ.")
    xyz = poses[..., :3].copy()
    hips = (xyz[:, 23] + xyz[:, 24]) / 2.0
    shoulders = (xyz[:, 11] + xyz[:, 12]) / 2.0
    lengths = np.linalg.norm(shoulders[:, :2] - hips[:, :2], axis=1)
    valid_lengths = lengths[lengths > 1e-3]
    scale = max(float(np.median(valid_lengths)) if len(valid_lengths) else 1.0, 1e-3)
    relative = (xyz - hips[:, None, :]) / scale
    visibility = np.clip(poses[..., 3], 0, 1)
    delta = max(np.float32(duration) / np.float32(31), np.float32(1e-6))
    velocity = np.zeros_like(relative)
    velocity[1:] = np.diff(relative, axis=0) / delta
    torso = shoulders - hips
    angle = np.arctan2(torso[:, 0], -torso[:, 1]) / np.pi
    displacement = hips[:, :2] - hips[:1, :2]
    joint = np.stack([relative[..., 0], relative[..., 1], visibility,
                      velocity[..., 0], velocity[..., 1]], axis=-1)
    body = np.concatenate([angle[:, None], displacement], axis=-1)
    body = np.repeat(body[:, None, :], 33, axis=1)
    features = np.concatenate([joint, body], axis=-1).transpose(2, 0, 1)[None].astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Đặc trưng có giá trị không hữu hạn.")
    return features, scale, bool(len(valid_lengths))


def sample_poses(raw, detected):
    raw = np.asarray(raw, dtype=np.float32)
    detected = np.asarray(detected, dtype=bool)
    if len(raw) < 2 or not detected.any():
        return None, np.zeros(32, dtype=bool)
    valid = np.flatnonzero(detected)
    filled = raw.copy()
    for joint in range(33):
        for feature in range(4):
            filled[:, joint, feature] = np.interp(np.arange(len(raw)), valid, raw[valid, joint, feature])
    selected = np.linspace(0, len(raw) - 1, 32).round().astype(int)
    return filled[selected], detected[selected]
