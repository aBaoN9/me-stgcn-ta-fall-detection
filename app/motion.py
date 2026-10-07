from collections import deque
import math

import numpy as np

from features import make_features, sample_poses


STEP_SECONDS = 0.5
CONFIRM_WINDOWS = 2


def motion_evidence(features, sample_mask, frame_count, fps, start_seconds):
    channels = features[0]
    speeds = np.sqrt(channels[3] ** 2 + channels[4] ** 2).mean(axis=1)
    angles = channels[5, :, 0] * 180.0
    hip_x = channels[6, :, 0]
    hip_y = channels[7, :, 0]
    return {
        "times": (start_seconds + np.linspace(0, frame_count - 1, 32).round() / fps).tolist(),
        "mean_joint_speed": speeds.tolist(),
        "torso_angle_degrees": angles.tolist(),
        "hip_dx": hip_x.tolist(), "hip_dy": hip_y.tolist(),
        "direct_mask": sample_mask.tolist(),
        "mean_visibility": channels[2].mean(axis=1).tolist(),
        "peak_joint_speed": float(speeds.max()),
        "peak_abs_torso_angle": float(np.abs(angles).max()),
        "final_hip_dy": float(hip_y[-1]),
        "units": {"mean_joint_speed": "mean over 33 joints, normalized torso lengths / second",
                  "torso_angle_degrees": "degrees in normalized x/width, y/height coordinates; not a 3D angle",
                  "hip_dx": "fraction of image width", "hip_dy": "fraction of image height, positive down"},
        "interpretation": "Descriptive input measurements, not causal feature attribution or clinical evidence.",
    }


def evaluate_sequence(raw, detected, fps, threshold, predict, start_seconds=0.0, require_current=False):
    duration = len(raw) / fps
    sampled, mask = sample_poses(raw, detected)
    rate = float(np.mean(detected))
    sufficient = sampled is not None and rate >= 0.5 and int(mask.sum()) >= 16
    reason = None if sufficient else "low_pose_quality"
    if require_current and not detected[-1]:
        sufficient, reason = False, "current_pose_missing"
    score, scale, evidence = None, None, None
    if sufficient:
        features, scale, valid = make_features(sampled, duration)
        if valid:
            score = float(predict(features))
            if not np.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Model trả điểm không hợp lệ; chưa thể kết luận.")
            evidence = motion_evidence(features, mask, len(raw), fps, start_seconds)
        else:
            reason = "invalid_body_scale"
    label = "insufficient" if score is None else ("fall" if score >= threshold else "adl")
    return {
        "label": label, "score": score, "threshold": threshold,
        "detection_rate": rate, "sampled_detected": int(mask.sum()),
        "sampled_mask": mask.tolist(), "body_scale": scale,
        "evidence": evidence, "quality_reason": reason,
        "feature_duration_seconds": duration,
    }


class AlertPolicy:
    def __init__(self):
        self.high = 0
        self.low = 0
        self.active = False
        self.candidate_since = None
        self.events = []

    def update(self, label, timestamp):
        if label == "insufficient":
            self._close(timestamp, "pose_unavailable")
            self.high = self.low = 0
            self.candidate_since = None
            return "insufficient"
        if label == "fall":
            self.high += 1
            self.low = 0
            if self.candidate_since is None:
                self.candidate_since = timestamp
            if self.high >= CONFIRM_WINDOWS and not self.active:
                self.active = True
                self.events.append({"candidate_at": self.candidate_since, "alert_at": timestamp,
                                    "ended_at": None, "end_reason": None})
            return "alert" if self.active else "suspected"
        self.high = 0
        self.candidate_since = None
        self.low += 1
        if self.active and self.low < CONFIRM_WINDOWS:
            return "recovering"
        self._close(timestamp, "two_below_threshold")
        return "below"

    def _close(self, timestamp, reason):
        if self.active:
            self.events[-1].update(ended_at=timestamp, end_reason=reason)
        self.active = False


class RollingAnalysis:
    def __init__(self, fps, seconds, threshold, predict):
        if not np.isfinite(seconds) or not 2 <= seconds <= 6:
            raise ValueError("Cửa sổ phải từ 2 đến 6 giây.")
        self.fps, self.seconds, self.threshold, self.predict = fps, seconds, threshold, predict
        self.capacity = math.ceil(seconds * fps) + 1
        self.raw = deque(maxlen=self.capacity)
        self.detected = deque(maxlen=self.capacity)
        self.next_index = math.ceil(seconds * fps)
        self.policy = AlertPolicy()
        self.records = []
        self.frame_index = -1

    def push(self, pose, found):
        self.frame_index += 1
        self.raw.append(pose)
        self.detected.append(found)
        if self.frame_index < self.next_index:
            return None
        start_index = self.frame_index - len(self.raw) + 1
        record = evaluate_sequence(list(self.raw), list(self.detected), self.fps,
                                   self.threshold, self.predict, start_index / self.fps, True)
        timestamp = self.frame_index / self.fps
        record.update(start_frame=start_index, end_frame=self.frame_index,
                      start_seconds=start_index / self.fps, end_seconds=timestamp,
                      state=self.policy.update(record["label"], timestamp),
                      high_streak=self.policy.high, low_streak=self.policy.low,
                      alert_active=self.policy.active)
        self.records.append(record)
        self.next_index = math.ceil((self.seconds + len(self.records) * STEP_SECONDS) * self.fps)
        return record

    def report(self):
        return {
            "window_seconds": self.seconds, "step_seconds": STEP_SECONDS,
            "confirm_windows": CONFIRM_WINDOWS, "records": self.records,
            "events": self.policy.events, "causal": True,
            "threshold_validated_for_windows": False,
            "clock": "frame_index / nominal_fps; use constant-frame-rate video",
            "policy": "Two consecutive above-threshold windows trigger; two below clear. Missing pose interrupts, not an all-clear.",
        }
