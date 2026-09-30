# Full notebook 07

Đã chạy full ngày 30/09/2026: 160 video, 79 Fall/81 ADL, 4 subject, 3 seed, 5 cấu hình mạng + SVM. 64 outer run, 192 hàng fold/policy và 7.680 hàng dự đoán; không phải 7.680 video độc lập.

- `pooled_by_seed.csv`: số liệu tổng hợp 160 mẫu của mỗi seed/policy.
- `seed_mean_std.csv`: CSV đã làm phẳng header, mean/std qua seed; SVM chỉ có một lượt nên std trống.
- `fold_metrics.csv`: mỗi subject giữ lại, threshold, epoch và inner target met.
- `predictions.csv` / `errors.csv`: đủ mẫu và lỗi, không loại clip khó.
- `outer_reports/`: split, lựa chọn inner, score và kết quả outer.
- `histories/`: lịch sử inner/refit; không phải metric test dùng chọn epoch.
- `attention/`: các mảng từ đúng checkpoint outer; các bản average có trọng số đều theo thiết kế.
- `locked_protocol.json`, `input_provenance.json`, `completion.json`, `timing.json`: cấu hình/runtime/hash. Path input đã bỏ owner, hash dữ liệu giữ nguyên.

Không lưu 60 checkpoint fold hoặc ZIP private input. Full ZIP gốc giữ ngoài Git; SHA-256 nằm trong `docs/package_sources.json`. Model 07 chưa có final all-subject/ONNX và không dùng trên web. `scientific_result=true` chỉ phân biệt full với smoke, không chứng nhận phương pháp hoặc test độc lập.
