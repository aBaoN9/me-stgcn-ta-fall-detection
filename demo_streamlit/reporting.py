import copy
import csv
import html
import io
import json


def report_data(result):
    report = copy.deepcopy({key: value for key, value in result.items()
                            if key not in ("poses", "angles")})
    for event in report.get("events", []):
        event.pop("thumbnail", None)
    return report


def summary(result):
    duration = result["info"]["duration"]
    rows = result["rows"]
    warmup_seconds = unavailable_seconds = 0.0
    for index, row in enumerate(rows):
        stop = rows[index + 1]["time"] if index + 1 < len(rows) else duration
        interval = max(0, stop - row["time"])
        if row.get("warmup"):
            warmup_seconds += interval
        elif row["score"] is None:
            unavailable_seconds += interval
    return {"events": len(result["events"]),
            "high_seconds": sum(event["end"] - event["start"] for event in result["events"]),
            "valid_windows": sum(row["score"] is not None for row in rows),
            "total_windows": len(rows), "warmup_seconds": warmup_seconds,
            "unavailable_seconds": unavailable_seconds,
            "processing_seconds": result["processing_seconds"]}


def trace_csv(result):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=["time", "score", "quality", "model_ms", "warmup"])
    writer.writeheader()
    writer.writerows({key: row.get(key) for key in writer.fieldnames} for row in result["rows"])
    return output.getvalue().encode("utf-8-sig")


def readable_report(result, model_name):
    totals = summary(result)
    escape = lambda value: html.escape(str(value), quote=True)
    event_rows = "".join(
        f'<tr><td>{index + 1}</td><td>{event["start"]:.2f}s</td>'
        f'<td>{event["end"]:.2f}s</td><td>{event["peak"]:.3f}</td></tr>'
        for index, event in enumerate(result["events"]))
    metadata = escape(json.dumps(report_data(result)["identity"], ensure_ascii=False, indent=2))
    return f'''<!doctype html><html lang="vi"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FallScope - Báo cáo phân tích</title><style>
body{{font:16px/1.7 sans-serif;color:#183354;max-width:900px;margin:40px auto;padding:0 24px}}
h1,h2{{color:#215bd8}}table{{width:100%;border-collapse:collapse}}td,th{{text-align:left;padding:10px;border-bottom:1px solid #dce6f4}}
.metrics{{background:#edf3ff;padding:20px;border-radius:12px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}
@media print{{body{{margin:0}}}}</style><h1>FallScope</h1><h2>Báo cáo phân tích video</h2>
<p>Video: {escape(result.get("video_name", "Video tải lên"))}<br>Mô hình: {escape(model_name)}<br>
Thời điểm phân tích: {escape(result.get("created_at", "Không ghi nhận"))}</p>
<div class="metrics">Thời lượng: {result["info"]["duration"]:.2f}s · Vùng score cao: {totals["events"]}<br>
Tổng thời gian score ≥ 0.50: {totals["high_seconds"]:.2f}s<br>
Cửa sổ hợp lệ: {totals["valid_windows"]}/{totals["total_windows"]}<br>
Khởi động: {totals["warmup_seconds"]:.2f}s · Thiếu dữ liệu sau khởi động: {totals["unavailable_seconds"]:.2f}s<br>
Thời gian xử lý: {totals["processing_seconds"]:.2f}s</div>
<h2>Vùng nghi ngờ cần xem lại</h2><table><thead><tr><th>Vùng</th><th>Bắt đầu</th><th>Kết thúc</th><th>Score cao nhất</th></tr></thead>
<tbody>{event_rows or '<tr><td colspan="4">Không có vùng score cao; không đồng nghĩa chắc chắn không có ngã.</td></tr>'}</tbody></table>
<h2>Nguồn mô hình</h2><pre>{metadata}</pre>
<p>SHA-256 video: {escape(result.get("video_sha256", "Không ghi nhận"))}<br>
Pipeline: {escape(result["pipeline"])} · Cửa sổ: {result["window_seconds"]:.1f}s · Ngưỡng: 0.50</p>
<h2>Giới hạn</h2><p>Phân tích trước rồi phát đồng bộ, không phải livestream. Score không phải Accuracy hoặc xác suất đã hiệu chuẩn.
Vùng score cao chưa phải sự kiện ngã được xác nhận. Thiếu pose không được xem là ADL. Video chưa có nhãn nên không tính Accuracy/F1 mới.
Đây là prototype nghiên cứu, không phải hệ thống cấp cứu. Báo cáo không chứa video, ảnh hay landmarks.
Có thể dùng chức năng In của trình duyệt để lưu PDF.</p></html>'''.encode("utf-8")
