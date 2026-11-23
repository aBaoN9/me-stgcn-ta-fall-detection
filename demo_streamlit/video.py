import base64
import json
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from demo_streamlit.prepare_assets import POSE_SHA256, verify_pose_asset
from training.component_ablation.features import make_features_v2


MAX_BYTES = 50 * 1024 * 1024
MAX_SECONDS = 60
MIN_POSE_RATE = 0.5
UPDATE_SECONDS = 0.25
PIPELINE_VERSION = "causal-window-v1"


class AnalysisCancelled(Exception):
    pass


def check_cancelled(check):
    if check is not None:
        check()


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    frames: int
    width: int
    height: int
    duration: float


def inspect_video(path):
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("Khong doc duoc video; thu MP4/H.264")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if not np.isfinite(fps) or not 1 <= fps <= 60 or frames < 2 or width <= 0 or height <= 0:
        raise ValueError("Video metadata khong hop le; can 1-60 FPS va it nhat 2 frame")
    if frames > 3600 or frames / fps > MAX_SECONDS or width * height > 1920 * 1080:
        raise ValueError("Ban thu gioi han 60 giay, 3600 frame va do phan giai Full HD")
    return VideoInfo(fps, frames, width, height, frames / fps)


def extract_track(path, progress=None, check=None):
    import mediapipe as mp

    check_cancelled(check)
    info = inspect_video(path)
    asset = verify_pose_asset()
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(asset)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=.35, min_pose_presence_confidence=.35,
        min_tracking_confidence=.35)
    poses, masks = [], []
    started = time.perf_counter()
    capture = cv2.VideoCapture(str(path))
    try:
        with mp.tasks.vision.PoseLandmarker.create_from_options(options) as detector:
            timestamp = -1
            while True:
                check_cancelled(check)
                readable, frame = capture.read()
                if not readable:
                    break
                if len(poses) >= min(3600, int(MAX_SECONDS * info.fps)):
                    raise ValueError("Decoded video exceeds the duration/frame limit")
                frame_index = len(poses)
                decoded_seconds = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
                if not np.isfinite(decoded_seconds) or abs(decoded_seconds - frame_index / info.fps) > max(.05, 2 / info.fps):
                    raise ValueError("Video co timestamp bien thien; hay chuyen sang MP4 constant FPS")
                timestamp = max(timestamp + 1, round(frame_index * 1000 / info.fps))
                image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                 data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                result = detector.detect_for_video(image, timestamp)
                detected = bool(result.pose_landmarks)
                pose = np.full((33, 4), np.nan, dtype=np.float32)
                if detected:
                    pose[:] = [[joint.x, joint.y, joint.z, joint.visibility]
                               for joint in result.pose_landmarks[0]]
                    detected = bool(np.isfinite(pose).all())
                poses.append(pose)
                masks.append(detected)
                if progress and (frame_index % 10 == 0):
                    progress(min(.8, (frame_index + 1) / info.frames * .8), "Đang trích xuất khung xương...")
    finally:
        capture.release()
    if len(poses) < 2 or abs(len(poses) - info.frames) > 2:
        raise ValueError("Video truncated or frame count does not match its metadata")
    info = VideoInfo(info.fps, len(poses), info.width, info.height, len(poses) / info.fps)
    return info, np.asarray(poses), np.asarray(masks, dtype=bool), time.perf_counter() - started


def window_features(poses, masks, fps, end_index, window_seconds):
    if not 1 <= window_seconds <= 6 or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid window duration or FPS")
    if poses.shape != (len(masks), 33, 4) or not 0 <= end_index < len(masks):
        raise ValueError("Invalid pose track shape/index")
    length = max(32, int(round(window_seconds * fps)))
    start_index = max(0, end_index - length + 1)
    detected = masks[start_index:end_index + 1]
    quality = float(detected.mean())
    if end_index + 1 < length or not masks[end_index] or quality < MIN_POSE_RATE:
        return None, quality
    sequence = poses[start_index:end_index + 1].copy()
    valid = np.flatnonzero(detected)
    if not np.isfinite(sequence[valid]).all():
        raise ValueError("Nonfinite detected pose")
    for joint_index in range(33):
        for channel in range(4):
            sequence[:, joint_index, channel] = np.interp(
                np.arange(length), valid, sequence[valid, joint_index, channel])
    indices = np.linspace(0, length - 1, 32).round().astype(int)
    return make_features_v2(sequence[indices], detected[indices], length / fps), quality


def trace_for_track(poses, masks, fps, window_seconds, predict, check=None):
    stride = max(1, round(fps * UPDATE_SECONDS))
    endpoints = sorted(set(range(0, len(masks), stride)) | {len(masks) - 1})
    rows = []
    for end_index in endpoints:
        check_cancelled(check)
        features, quality = window_features(poses, masks, fps, end_index, window_seconds)
        score, elapsed = None, None
        if features is not None:
            started = time.perf_counter()
            score = float(predict(features))
            elapsed = (time.perf_counter() - started) * 1000
            if not np.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Nonfinite or out-of-range model score")
        rows.append({"time": end_index / fps, "score": score, "quality": quality,
                     "model_ms": elapsed, "warmup": end_index + 1 < max(32, round(window_seconds * fps))})
    return rows


def events_for_trace(rows, duration):
    events, active = [], None
    for index, row in enumerate(rows):
        next_time = rows[index + 1]["time"] if index + 1 < len(rows) else duration
        high = row["score"] is not None and row["score"] >= .5
        if high:
            if active is None:
                active = {"start": row["time"], "end": next_time, "peak": row["score"]}
            else:
                active["end"] = next_time
                active["peak"] = max(active["peak"], row["score"])
        elif active is not None:
            events.append(active)
            active = None
    if active is not None:
        events.append(active)
    return events


def preview_video(source, destination, check=None):
    import imageio_ffmpeg

    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error", "-i", str(source),
               "-map", "0:v:0", "-an", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "25", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", "-y", str(destination)]
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    started = time.monotonic()
    try:
        while process.poll() is None:
            check_cancelled(check)
            if time.monotonic() - started > 120:
                raise ValueError("Tạo preview quá lâu. Hãy thử video ngắn hơn.")
            time.sleep(.1)
        if process.returncode:
            raise ValueError("Không tạo được preview H.264. Hãy thử video MP4 khác.")
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
    if destination.stat().st_size > MAX_BYTES:
        raise ValueError("Preview exceeds memory limit")
    return destination.read_bytes()


def event_thumbnails(path, events, check=None):
    capture = cv2.VideoCapture(str(path))
    try:
        for event in events:
            check_cancelled(check)
            capture.set(cv2.CAP_PROP_POS_MSEC, event["start"] * 1000)
            readable, frame = capture.read()
            if not readable:
                continue
            height, width = frame.shape[:2]
            scale = min(240 / width, 144 / height, 1)
            frame = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))))
            encoded, image = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
            if encoded:
                event["thumbnail"] = base64.b64encode(image).decode("ascii")
    finally:
        capture.release()


def analyze_video(video_bytes, model, identity, window_seconds=3, progress=None, check=None):
    import torch

    from training.backbone_benchmark.models import logits_of

    if not 0 < len(video_bytes) <= MAX_BYTES:
        raise ValueError("Video can nho hon hoac bang 50 MB")
    started = time.perf_counter()
    check_cancelled(check)
    with tempfile.TemporaryDirectory(prefix="fallscope-video-") as temporary:
        source = Path(temporary) / "source.video"
        source.write_bytes(video_bytes)
        if progress:
            progress(0, "Đang kiểm tra video và khởi tạo MediaPipe (lần đầu có thể chậm)...")
        info, poses, masks, pose_seconds = extract_track(source, progress, check)

        def predict(features):
            with torch.inference_mode():
                return float(logits_of(model, torch.from_numpy(features[None])).softmax(1)[0, 1])

        if progress:
            progress(.82, "Đang chạy mô hình trên cửa sổ quá khứ...")
        rows = trace_for_track(poses, masks, info.fps, window_seconds, predict, check)
        events = events_for_trace(rows, info.duration)
        event_thumbnails(source, events, check)
        if progress:
            progress(.94, "Đang tạo video xem lại...")
        preview = preview_video(source, Path(temporary) / "preview.mp4", check)
    overlay, angles = [], []
    for pose, detected in zip(poses, masks):
        check_cancelled(check)
        if detected:
            overlay.append(np.round(pose[:, [0, 1, 3]], 5).tolist())
            torso = pose[[11, 12], :2].mean(axis=0) - pose[[23, 24], :2].mean(axis=0)
            angles.append(float(abs(np.degrees(np.arctan2(torso[0], -torso[1])))))
        else:
            overlay.append(None)
            angles.append(None)
    result = {"info": asdict(info), "rows": rows, "poses": overlay, "angles": angles,
              "events": events, "identity": identity,
              "window_seconds": window_seconds, "minimum_pose_rate": MIN_POSE_RATE,
              "pipeline": PIPELINE_VERSION, "pose_asset_sha256": POSE_SHA256,
              "pose_seconds": pose_seconds, "processing_seconds": time.perf_counter() - started,
              "valid_windows": sum(row["score"] is not None for row in rows)}
    json.dumps(result, allow_nan=False)
    return preview, result
