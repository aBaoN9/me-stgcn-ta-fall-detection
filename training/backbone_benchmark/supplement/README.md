# GRU và CTR-GCN bổ sung

**Tích hợp 21/11/2026:** benchmark được nghiệm thu theo `GB-20261121-001`. Monitor hiện tự tạo thư mục cha `.runtime` bằng `backup.mkdir(parents=True)`; ba test riêng kiểm runtime mới/có sẵn và guard nhiệt, không khởi chạy train thật. Bản sửa tiện ích không đổi nguồn train đã khóa hoặc checkpoint/kết quả và không cần train lại. Lệnh dưới đây chạy đủ 16 tests local. Trên Windows nên dùng checkout ngắn hoặc bật Git long paths cho repo khi cần; giữ nguyên đường dẫn/byte artifact.

[Thiết kế và nguồn](../../../docs/research/gru-ctrgcn-addendum.md). Module này giữ riêng runner (chương trình huấn luyện) để không làm đổi chữ ký nguồn của run TCN–A3 đã nghiệm thu. Các bước kiểm dữ liệu, chia người, tính điểm, tạo đặc trưng và định nghĩa cấu hình vẫn dùng hạ tầng của benchmark gốc.

Chạy từ thư mục gốc với Python/PyTorch đã khóa trong kết quả. Trên Windows, có thể thay `python` bằng `.venv/Scripts/python.exe`.

```powershell
python -B scripts/training_inputs.py --input-root data/gmdcsa24
python -B -m training.component_ablation.validate_pipeline --input-root data/gmdcsa24
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
python -B -m unittest -v training.backbone_benchmark.test_benchmark training.backbone_benchmark.supplement.test_supplement training.backbone_benchmark.supplement.tools.test_monitor_windows
```

Giám sát Windows trên GPU NVIDIA số 0 với ngưỡng dừng 87°C, lấy mẫu nhiệt độ, VRAM (bộ nhớ GPU) và RAM (bộ nhớ hệ thống). Mỗi lượt chạy giám sát tạo bản sao riêng cho checkpoint và lịch sử từng vòng đã hoàn tất. Giữ `$smoke` và `$full` khi cần tiếp tục một lần chạy bị gián đoạn.

```powershell
$tag = Get-Date -Format 'yyyyMMdd-HHmmss'
$smoke = ".runtime/gru-ctr-$tag-smoke"
$full = ".runtime/gru-ctr-$tag-full"
python -B -m training.backbone_benchmark.supplement.tools.monitor_windows smoke $smoke
if ($LASTEXITCODE -ne 0) { throw "Smoke failed" }
python -B -m training.backbone_benchmark.supplement.tools.monitor_windows full $full $smoke
```

Lệnh trực tiếp khi cần kiểm chứng lại một run hoàn tất hoặc tiếp tục đúng cấu hình:

```powershell
python -B -m training.backbone_benchmark.supplement.runner --mode full --device cuda --input-root data/gmdcsa24 --output $full --smoke-evidence $smoke --resume
```

Tái tính bảng so sánh bốn mô hình và đo lại thời gian suy luận trên cùng một GPU. Tệp `comparison_fixed_errors.csv` liệt kê lỗi theo clip và seed. Tệp `comparison_fixed_paired.csv` ghép dự đoán bốn mô hình theo cùng clip và seed.

```powershell
$reference = "results/backbone_benchmark/tung-m2-rtx3050-20261120-165523"
python -B -m training.backbone_benchmark.supplement.tools.audit --input-root data/gmdcsa24 --run $full --reference $reference --output ".runtime/gru-ctr-$tag-audit"
```
