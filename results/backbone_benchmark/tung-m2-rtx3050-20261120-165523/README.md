# M2/M3 — TCN và A3 trên RTX 3050

Bộ bàn giao của Tùng. Dữ liệu nghiên cứu GMDCSA-24 gồm 160 clip (79 Fall, 81 ADL), bốn người. M2 chạy trên mã `12eaf80e3f2dcbdc9df33749fa61aad5b6f6ce0c`. Phiên M3 kiểm chứng bằng suy luận lại với trọng số đã lưu.

- [Khóa cấu hình](locked_protocol.json) lưu mã nguồn, bốn hash dữ liệu, môi trường, thiết bị và tham số.
- [Kết thúc M2](completion.json) ghi 96 lượt fit, 24 vòng ngoài, reload PASS và coverage PASS. Mỗi thư mục fold có lịch sử inner, ngưỡng, refit, dự đoán và `model.pt`.
- [Danh mục 24 checkpoint](models/manifest.json) ghi đường dẫn **tương đối với thư mục run này**, tên mô hình, người, seed, byte và SHA-256.
- [Dự đoán ngoài mẫu](predictions.csv), [lỗi](errors.csv), [chỉ số theo người](m3_fold_metrics.csv), [chỉ số theo seed](m3_seed_metrics.csv), [trung bình và độ lệch chuẩn mẫu](m3_seed_mean_std.csv), [chênh lệch TCN−A3](m3_delta_tcn_minus_a3.csv), [cặp dự đoán theo clip](m3_paired_predictions.csv) và [danh sách lỗi fixed](fixed_error_cases.csv). TP/TN/FP/FN tạo ma trận nhầm lẫn với thứ tự lớp ADL/Fall là [[TN,FP],[FN,TP]].
- [Kiểm chứng M3](m3_audit.json), [đo suy luận](m3_model_latency.csv), [danh mục SHA-256 bộ bàn giao](m3_files_sha256.json). ZIP gốc M2 đã được đối chiếu 60 tệp với máy huấn luyện. Git chứa từng tệp trong ZIP cùng phân tích M3, để tải trực tiếp bằng clone.

## Kiểm tra lại từ gốc repo

Dùng Python 3.11.9, PyTorch 2.10.0+cu126 (CUDA 12.6), NumPy 2.2.6, pandas 2.3.3 và scikit-learn 1.7.2. Cài theo hướng dẫn môi trường 07 và dùng đúng bản CUDA cho phép đối chiếu score trên GPU. Bốn tệp chuẩn nằm ở `data/gmdcsa24`.

```powershell
python scripts/training_inputs.py --input-root data/gmdcsa24
python -B -m training.backbone_benchmark.test_benchmark
python -B scripts/check_repository.py
```

Ví dụ kiểm toàn bộ checkpoint đã lưu trên cấu hình CUDA đã khóa. Đoạn lệnh chỉ kiểm tra và chạy suy luận, không gọi hàm huấn luyện.

```python
from pathlib import Path
import torch
from training.backbone_benchmark.runner import (
    collect, load_verified, protocol_lock, read_json, validate_summary,
)
run = Path("results/backbone_benchmark/tung-m2-rtx3050-20261120-165523")
device = torch.device("cuda")
data = load_verified(Path("data/gmdcsa24"))
lock = read_json(run / "locked_protocol.json")
assert protocol_lock(data, "full", device, False) == lock
print(validate_summary(data, lock, run, collect(data, lock, device, run)))
```

Có thể kiểm cấu trúc và nạp một checkpoint trên CPU bằng `load_model(run / record["file"], torch.device("cpu"))`, với `record` lấy từ danh mục. Kiểm score sát ngưỡng nên dùng đúng môi trường đã khóa. Lệnh chạy lại từ đầu và bằng chứng lần chạy gốc nằm trong [báo cáo Tùng](../../../docs/reports/dinhtung.md). [Phân tích khoa học](../../../docs/research/backbone-benchmark.md) được ghi riêng sau M0.
