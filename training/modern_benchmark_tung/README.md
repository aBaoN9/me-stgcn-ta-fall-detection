# GB-20261124-001 — runner của Tùng

[Thiết kế và nguồn](../../docs/research/modern-benchmark-tung.md) · [Hợp đồng đánh giá](../../docs/research/modern-model-benchmark.md).

Chạy từ thư mục gốc. Dùng Python 3.11 cùng môi trường đã ghi trong khóa. Đây là ví dụ đã dùng để chạy A3 trên Windows CUDA. Các ID được hỗ trợ hiện trong `--help`. Mỗi mô hình dùng thư mục smoke và full riêng.

```powershell
.venv/Scripts/python.exe -B scripts/training_inputs.py --input-root data/gmdcsa24
.venv/Scripts/python.exe -B -m training.component_ablation.validate_pipeline --input-root data/gmdcsa24
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
.venv/Scripts/python.exe -B -m unittest -v training.modern_benchmark_tung.test_benchmark training.backbone_benchmark.test_benchmark
.venv/Scripts/python.exe -B -m training.modern_benchmark_tung.runner --help
$root = ".runtime/modern-reproduction"
$model = "A3_motion_attention"
$smoke = "$root/$model-smoke"
$full = "$root/$model-full"
.venv/Scripts/python.exe -B -m training.modern_benchmark_tung.monitor_windows smoke $model $smoke
if ($LASTEXITCODE) { throw "Smoke failed" }
.venv/Scripts/python.exe -B -m training.modern_benchmark_tung.monitor_windows full $model $full $smoke
```

Bộ giám sát chạy với bốn thread, sao lưu từng vòng đã hoàn tất, ghi nhiệt độ/bộ nhớ/mức sử dụng GPU và dừng nếu nhiệt độ đạt 87°C hoặc SSD còn dưới 1 GiB. Chạy lại lệnh với cùng thư mục để tiếp tục sau khi gián đoạn. Kiểm chứng full đã hoàn tất bằng lệnh sau, giữ thiết lập thread theo đúng khóa.

```powershell
$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
.venv/Scripts/python.exe -B -m training.modern_benchmark_tung.runner --model $model --input-root data/gmdcsa24 --output $full --mode full --device cuda --smoke-evidence $smoke --resume
```

Với CPU (bộ xử lý trung tâm), dùng runner trực tiếp với `--device cpu` cho cả smoke và full, cùng thread và môi trường. Dùng bằng chứng smoke cùng mô hình và chạy A3 đối chứng trên cùng môi trường. Các lượt cũ chỉ được bỏ qua khi khóa, checkpoint, dữ liệu, lịch sử, điểm số và phạm vi đều qua kiểm tra.

Sau khi có đủ sáu thư mục full và smoke, kiểm lại mô hình, bảng chỉ số và độ trễ bằng lệnh dưới. Các bảng giữ riêng kết quả cho ngưỡng fixed 0,5 và hai ngưỡng phụ.

```powershell
.venv/Scripts/python.exe -B -m training.modern_benchmark_tung.audit --input-root data/gmdcsa24 --runs $root --output "$root/comparison" --device cuda
```
