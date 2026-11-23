import threading
import time
from datetime import datetime, timezone

from demo_streamlit.data import load_checkpoint_model
from demo_streamlit.video import AnalysisCancelled, analyze_video


class AnalysisJob:
    def __init__(self, video_bytes, signature, video_name, loader=None, analyzer=None):
        self.signature = signature
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self._status = "running"
        self._fraction = 0.0
        self._message = "Đang kiểm tra và nạp mô hình..."
        self._output = None
        self._started = time.monotonic()
        self._thread = threading.Thread(
            target=self._run, args=(video_bytes, video_name, loader or load_checkpoint_model,
                                   analyzer or analyze_video), daemon=True)
        self._thread.start()

    def _check(self):
        if self._cancel.is_set():
            raise AnalysisCancelled("Đã hủy phân tích; không lưu kết quả dở dang.")
        if time.monotonic() - self._started > 240:
            raise AnalysisCancelled("Đã dừng sau giới hạn xử lý 4 phút. Hãy thử video ngắn hơn.")

    def _progress(self, fraction, message):
        self._check()
        with self._lock:
            self._fraction, self._message = fraction, message

    def _run(self, video_bytes, video_name, loader, analyzer):
        try:
            self._check()
            video_hash, run_key, variant, seed, subject, window_seconds, _ = self.signature
            model, identity = loader(run_key, variant, seed, subject)
            self._check()
            preview, result = analyzer(video_bytes, model, identity, window_seconds,
                                       self._progress, self._check)
            result.update(video_sha256=video_hash, video_name=video_name[:200],
                          created_at=datetime.now(timezone.utc).isoformat())
            with self._lock:
                self._check()
                self._output = (preview, result)
                self._status, self._fraction = "completed", 1.0
                self._message = "Phân tích hoàn tất."
        except AnalysisCancelled as exc:
            with self._lock:
                self._status, self._message = "cancelled", str(exc)
        except Exception as exc:
            with self._lock:
                self._status = "failed"
                self._message = (str(exc) if isinstance(exc, (ValueError, FileNotFoundError))
                                 else "Không xử lý được video. Thử MP4/H.264 một người, tối đa 60 giây.")

    def cancel(self):
        self._cancel.set()
        with self._lock:
            self._output = None
            if self._status == "completed":
                self._status, self._message = "cancelled", "Đã bỏ kết quả cũ."
            elif self._status == "running":
                self._message = "Đang hủy và dọn file tạm..."

    def snapshot(self):
        with self._lock:
            return self._status, self._fraction, self._message

    def take_result(self):
        with self._lock:
            output, self._output = self._output, None
            return output
