from pathlib import Path
import hashlib
import json
import time

import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort

from features import CHANNELS
from motion import RollingAnalysis, evaluate_sequence
from prepare import POSE_MODEL


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT.parent / "models" / "deployment_06"
MAX_SECONDS = 120
MAX_FRAMES = 7200


class Engine:
    def __init__(self):
        if not POSE_MODEL.exists():
            raise RuntimeError("Chưa có Pose Landmarker. Chạy start.ps1 để tải model.")
        self.config = json.loads((MODEL_DIR / "model_config.json").read_text(encoding="utf-8"))
        assert self.config["channel_names"] == CHANNELS
        assert self.config["torso_angle_global_index"] == 9
        model_bytes = (MODEL_DIR / self.config["model_file"]).read_bytes()
        self.signature = hashlib.sha256(model_bytes).hexdigest()
        manifest = json.loads((MODEL_DIR / "artifact_manifest.json").read_text(encoding="utf-8"))
        if self.signature != manifest["files_sha256"][self.config["model_file"]]:
            raise RuntimeError("ONNX checksum does not match the packaged model manifest")
        self.session = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])

    def predict(self, features):
        output = self.session.run([self.config["output_name"]], {self.config["input_name"]: features})[0]
        return float(output[0])

    def analyze(self, path, filename, progress, mode="clip", window_seconds=3.0, observation=None):
        if mode not in {"clip", "timeline"}:
            raise ValueError("Chế độ không hợp lệ.")
        started = time.monotonic()
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise ValueError("Không đọc được video. Hãy dùng MP4 H.264 hoặc WebM hợp lệ.")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if not np.isfinite(fps) or fps <= 0 or fps > 120 or expected < 2:
                raise ValueError("Không đọc được FPS/số frame hợp lệ (tối đa 120 FPS).")
            if expected / fps > MAX_SECONDS or expected > MAX_FRAMES:
                raise ValueError("Video quá dài. Hãy chọn clip tối đa 120 giây / 7.200 frame.")
            if width * height > 3840 * 2160:
                raise ValueError("Video vượt quá 4K. Hãy xuất lại ở độ phân giải thấp hơn.")
            options = mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(POSE_MODEL)),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.35,
                min_pose_presence_confidence=0.35,
                min_tracking_confidence=0.35,
            )
            threshold = float(self.config["decision_threshold"])
            rolling = RollingAnalysis(fps, window_seconds, threshold, self.predict) if mode == "timeline" else None
            raw, detected, overlay = [], [], []
            previous_timestamp = -1
            with mp.tasks.vision.PoseLandmarker.create_from_options(options) as detector:
                while True:
                    success, frame = capture.read()
                    if not success:
                        break
                    index = len(raw)
                    if index >= MAX_FRAMES or index / fps > MAX_SECONDS:
                        raise ValueError("Video thực tế vượt giới hạn xử lý.")
                    timestamp = max(round(index * 1000 / fps), previous_timestamp + 1)
                    previous_timestamp = timestamp
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    result = detector.detect_for_video(image, timestamp)
                    values = np.full((33, 4), np.nan, dtype=np.float32)
                    found = bool(result.pose_landmarks)
                    if found:
                        values = np.array([[point.x, point.y, point.z, point.visibility]
                                           for point in result.pose_landmarks[0]], dtype=np.float32)
                        found = bool(np.isfinite(values).all())
                    raw.append(values)
                    detected.append(found)
                    overlay.append(values[:, [0, 1, 3]].round(5).tolist() if found else None)
                    if rolling:
                        record = rolling.push(values, found)
                        if record and observation:
                            observation({key: record[key] for key in ["end_seconds", "score", "state", "sampled_detected"]})
                    if index % 8 == 0:
                        progress(min(94, int((index + 1) / expected * 94)), f"Đọc pose · {index + 1}/{expected} frame")
        finally:
            capture.release()
        if len(raw) < 2:
            raise ValueError("Video không có đủ frame đọc được.")
        progress(96, "Tạo 8 kênh đặc trưng · 32 thời điểm")
        duration = len(raw) / fps
        rate = float(np.mean(detected))
        warnings = []
        if len(raw) != expected:
            warnings.append("Số frame đọc được khác metadata; thời lượng dùng số frame thực đọc.")
        if rate < 0.8:
            warnings.append("Pose bị mất ở nhiều frame; nội suy có thể làm thay đổi chuyển động.")
        progress(98, "Tổng hợp đặc trưng và kết quả")
        summary = evaluate_sequence(raw, detected, fps, threshold, self.predict)
        if summary["label"] == "insufficient":
            warnings.append("Chưa đủ dữ liệu pose để kết luận. Đây không phải kết quả ADL.")
        if rolling:
            warnings.append("Cửa sổ trượt là thử nghiệm: model/ngưỡng được xây cho toàn clip, chưa được kiểm định cho cảnh báo theo thời gian.")
            if not rolling.records:
                warnings.append("Clip ngắn hơn thời gian tích lũy cửa sổ; không có quyết định theo thời gian.")
        warnings.append("Mốc thời gian tính theo FPS danh định; dùng video FPS cố định. Vận tốc là tương đối, không phải m/s.")
        return {
            **summary, "filename": filename, "model": "ME-STGCN-TA", "analysis_mode": mode,
            "scope": "whole_clip", "duration": duration, "fps": fps,
            "frame_count": len(raw), "width": width, "height": height,
            "timeline": rolling.report() if rolling else None,
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "warnings": warnings, "overlay": overlay,
            "model_sha256": self.signature, "mediapipe_version": mp.__version__,
            "quality_policy": "Heuristic: >=50% detected and >=16/32 sampled frames; not clinically validated.",
        }
