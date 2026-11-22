# SkateFormer_adapted_small

[Thiết kế M0 đã khóa](../../../docs/research/skateformer-addendum.md). Mã mới nằm riêng để giữ mã băm và kết quả TCN–A3, GRU và CTR-GCN đã nghiệm thu. Dùng cùng dữ liệu, cách tính đặc trưng, chia theo người, chế độ huấn luyện và ngưỡng.

Nguồn tham khảo Do và Kim (ECCV 2024), được đối chiếu tại `0249e19d96b33e9150760129bbede56464126ae7`. Kho tác giả có tệp MIT và README yêu cầu xin phép khi dùng thương mại. Thí nghiệm này triển khai lại ý tưởng, không nhập mã hoặc checkpoint của tác giả. [Mã gốc](https://github.com/KAIST-VICLab/SkateFormer/blob/0249e19d96b33e9150760129bbede56464126ae7/model/SkateFormer.py) · [Điều khoản công bố](https://github.com/KAIST-VICLab/SkateFormer/blob/0249e19d96b33e9150760129bbede56464126ae7/README.md#license)

Chạy từ thư mục gốc với môi trường Python/PyTorch trong khóa. Trên Windows có thể thay `python` bằng `.venv/Scripts/python.exe`.

```powershell
python -B scripts/training_inputs.py --input-root data/gmdcsa24
python -B -m training.component_ablation.validate_pipeline --input-root data/gmdcsa24
python -B -m unittest -v training.backbone_benchmark.test_benchmark training.backbone_benchmark.supplement.test_supplement training.backbone_benchmark.supplement.tools.test_monitor_windows training.backbone_benchmark.skate.test_skate
$tag = Get-Date -Format 'yyyyMMdd-HHmmss'
$smoke = ".runtime/skate-$tag-smoke"
$full = ".runtime/skate-$tag-full"
python -B -m training.backbone_benchmark.skate.tools.monitor_windows smoke $smoke
if ($LASTEXITCODE -ne 0) { throw "Smoke failed" }
python -B -m training.backbone_benchmark.skate.tools.monitor_windows full $full $smoke
```

Dùng lại đường dẫn full và smoke để tiếp tục. Monitor (bộ giám sát) ghi nhiệt độ, bộ nhớ, mức sử dụng GPU và thời gian. Khi đạt 87°C hoặc SSD còn dưới 1 GiB, tiến trình dừng và những vòng đã hoàn tất được giữ để tiếp tục.

```powershell
$reference = "results/backbone_benchmark/tung-m2-rtx3050-20261120-165523"
$supplement = "results/backbone_benchmark/tung-gru-ctr-rtx3050-20261120-233932"
python -B -m training.backbone_benchmark.skate.tools.audit --input-root data/gmdcsa24 --run $full --reference $reference --supplement $supplement --output ".runtime/skate-$tag-audit"
```

Kiểm chứng một lần chạy đã hoàn tất bằng cách chạy lại lệnh giám sát với cùng đường dẫn hoặc dùng trực tiếp runner với `--resume`. Lần kiểm chứng nạp checkpoint, kiểm dự đoán và số lượt theo khóa.
