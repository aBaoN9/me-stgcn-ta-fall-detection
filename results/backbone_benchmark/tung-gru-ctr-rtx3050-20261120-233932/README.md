# Benchmark bổ sung GRU và CTR-GCN trên RTX 3050

Thực hiện ngày 20/11/2026 (UTC+7), trên nền `5939c9e`. [Thiết kế khóa](../../../docs/research/gru-ctrgcn-addendum.md) · [Lệnh tái lập và kiểm chứng](../../../training/backbone_benchmark/supplement/README.md) · [Mốc TCN–A3 đã nghiệm thu](../tung-m2-rtx3050-20261120-165523/README.md).

## Bằng chứng

- [Khóa full](locked_protocol.json) · [Completion](completion.json) · [24 checkpoint](models/manifest.json) · [Dự đoán ngoài mẫu](predictions.csv) · [Danh mục SHA-256](files_sha256.json).
- [Smoke thật 8/8](smoke/completion.json) · [Hai checkpoint smoke](smoke/models/manifest.json). Từng vòng full đã được đối chiếu checkpoint và lịch sử với bản sao cục bộ. Nạp lại 24 checkpoint full, kiểm 2.880 dự đoán và tái tính chỉ số trong sai số `1e-12`.
- [Chỉ số theo người](comparison/comparison_fixed_by_subject.csv) · [Theo seed](comparison/comparison_fixed_by_seed.csv) · [Trung bình và độ lệch chuẩn mẫu](comparison/comparison_fixed_mean_std.csv) · [Lỗi](comparison/comparison_fixed_errors.csv) · [Dự đoán ghép cặp](comparison/comparison_fixed_paired.csv) · [Đo suy luận](comparison/comparison_latency.csv).
- Full 428,11 giây, smoke 15,20 giây. Đỉnh full theo lấy mẫu là 66°C, 228 MiB VRAM toàn GPU, 14.064,3 MiB RAM hệ thống đang dùng, GPU utilization 44%. Đỉnh bộ nhớ CUDA do PyTorch cấp phát là 89.246.720 byte. [Đo full](monitoring/tung-gru-ctr-rtx3050-20261120-233932-full-monitor-20261120-233954-measurement.json) · [Dữ liệu lấy mẫu](monitoring/tung-gru-ctr-rtx3050-20261120-233932-full-monitor-20261120-233954-telemetry.csv).

## So sánh ngưỡng cố định 0,5

Chỉ số tỷ lệ là trung bình ba seed, FN/FP là tổng theo 480 lượt clip–seed mỗi mô hình. Thời gian đo lại trên cùng RTX 3050, 30 lượt làm nóng, 200 lượt đo, batch 1, input S1_ADL_01, checkpoint seed 42 / người 1. Chỉ đo mô hình.

| Mô hình | Accuracy | Precision | Sensitivity | Specificity | F1 | FN/FP | Tham số | Checkpoint mẫu (byte) | Trung vị/p95 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A3 | 89,38% | 86,90% | 92,41% | 86,42% | 89,57% | 18/33 | 422.871 | 1.742.362 | 2,549/3,490 |
| TCN | 91,04% | 88,55% | 94,09% | 88,07% | 91,20% | 14/29 | 56.434 | 237.298 | 2,178/3,361 |
| GRU | 89,58% | 88,18% | 91,14% | 88,07% | 89,62% | 21/29 | 64.018 | 262.560 | 0,366/0,666 |
| CTR-GCN adapted | 88,54% | 87,94% | 89,03% | 88,07% | 88,48% | 26/29 | 24.044 | 124.112 | 2,906/3,139 |

- TCN có Accuracy và Sensitivity trung bình cao nhất. GRU nhanh nhất theo phép đo, Accuracy cao hơn TCN ở seed 142 (91,25% so với 90,63%). CTR-GCN ít tham số nhất, nhưng có nhiều lần bỏ sót nhất.
- Trên S4, A3 đạt Accuracy trung bình 82,88%, TCN 80,18%, GRU và CTR-GCN đều 77,48%. Cả bốn mô hình đều báo nhầm `S1_ADL_02` và `S4_ADL_05` ở cả ba seed.
- GRU và TCN cùng bỏ sót `S1_FALL_13`, `S2_FALL_14`, `S4_FALL_03` ở cả ba seed. CTR-GCN bỏ sót `S4_FALL_05`, `S4_FALL_09`, `S4_FALL_17` ở cả ba seed. A3 cũng bỏ sót `S4_FALL_05` ở cả ba.
- Kiểm tra khóa cho thấy hai nghiên cứu cùng dữ liệu, GPU, phiên bản thư viện và cấu hình huấn luyện, chia tập, chọn epoch và ngưỡng. Mốc TCN–A3 lấy từ run đã nghiệm thu, còn thời gian suy luận được đo lại cho cả bốn. CTR-GCN là biến thể thu gọn sang MediaPipe 33 khớp. Bốn người và các clip lặp qua ba seed chỉ hỗ trợ kết luận khám phá trên bộ dữ liệu này. Cần tập độc lập để đánh giá khả năng khái quát.
