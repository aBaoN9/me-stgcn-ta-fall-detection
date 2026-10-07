from pathlib import Path
import time

import cv2


def create_browser_preview(source, destination, report):
    started = time.monotonic()
    destination = Path(destination)
    capture = cv2.VideoCapture(str(source))
    writer = None
    try:
        if not capture.isOpened():
            raise ValueError("Không đọc được video để tạo bản xem lại.")
        scale = min(1.0, 1280 / report["width"], 720 / report["height"])
        width = max(2, int(report["width"] * scale) // 2 * 2)
        height = max(2, int(report["height"] * scale) // 2 * 2)
        writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"VP80"),
                                 report["fps"], (width, height))
        if not writer.isOpened():
            raise ValueError("Máy này không có bộ mã hóa WebM VP8.")
        frame_count = 0
        while True:
            valid, frame = capture.read()
            if not valid:
                break
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
            frame_count += 1
        if frame_count != report["frame_count"]:
            raise ValueError("Bản xem lại không có đủ frame của video đã phân tích.")
        writer.release()
        writer = None
        if not destination.exists() or destination.stat().st_size == 0:
            raise ValueError("Không tạo được bản WebM xem lại.")
        return {"codec": "WebM VP8", "frame_count": frame_count, "fps": report["fps"],
                "width": width, "height": height, "audio": False,
                "elapsed_seconds": round(time.monotonic() - started, 2),
                "inference_uses_original": True}
    except BaseException:
        if writer is not None:
            writer.release()
            writer = None
        destination.unlink(missing_ok=True)
        raise
    finally:
        capture.release()
        if writer is not None:
            writer.release()
