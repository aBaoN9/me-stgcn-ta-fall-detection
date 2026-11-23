# SkateFormer_adapted_small trên RTX 3050

22/11/2026 (UTC+7), trên nền `4edd8d4`. [Addendum M0](../../../docs/research/skateformer-addendum.md) · [Hướng dẫn chạy và tái lập](../../../training/backbone_benchmark/skate/README.md).

- Full **48/48** lượt (36 inner, 12 refit) và 12 checkpoint. Smoke thật **4/4**. SHA-256 bốn đầu vào, 20/20 kiểm thử, nạp lại mô hình, kiểm 1.440 dự đoán ngoài mẫu và tái tính chỉ số đều đạt.
- Full **304,50 giây**, smoke **11,56 giây**. Đỉnh đo full: 73°C, 197 MiB VRAM toàn GPU, 14.871,1 MiB RAM hệ thống đang dùng, 54% sử dụng GPU. Đỉnh CUDA do PyTorch cấp phát 78.285.312 byte.
- [Khóa full](locked_protocol.json) · [Trạng thái full](completion.json) · [Checkpoint](models/manifest.json) · [Dự đoán](predictions.csv) · [SHA-256](files_sha256.json) · [Smoke](smoke/completion.json) · [Giám sát](monitoring/tung-skate-rtx3050-20261122-205954-full-monitor-20261122-210007-measurement.json).

## So sánh ngưỡng cố định 0,5

Tỷ lệ là trung bình qua ba seed. TP/TN/FP/FN là tổng qua 480 lượt clip–seed trên mỗi mô hình. Mốc ME-STGCN-TA là `A3_motion_attention` chạy lại trong benchmark đã nghiệm thu, còn model triển khai là bản riêng. Chỉ số của bốn mốc lấy từ kết quả đã nghiệm thu sau khi đối chiếu SHA-256 dữ liệu, split, GPU, phiên bản phần mềm và cấu hình. Đo lại thời gian suy luận cả năm mô hình trên cùng RTX 3050, PyTorch 2.10.0+cu126, cùng đầu vào S1_ADL_01, batch 1, checkpoint seed 42/người 1, 30 warmup và 200 lần đo có đồng bộ GPU.

| Mô hình | Accuracy | Precision | Sensitivity | Specificity | F1 | TP/TN/FP/FN | Tham số | Trung vị/p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A3 | 89,38% | 86,90% | 92,41% | 86,42% | 89,57% | 219/210/33/18 | 422.871 | 4,006/5,861 |
| TCN | 91,04% | 88,55% | 94,09% | 88,07% | 91,20% | 223/214/29/14 | 56.434 | 3,494/5,126 |
| GRU | 89,58% | 88,18% | 91,14% | 88,07% | 89,62% | 216/214/29/21 | 64.018 | 0,700/0,850 |
| CTR-GCN adapted | 88,54% | 87,94% | 89,03% | 88,07% | 88,48% | 211/214/29/26 | 24.044 | 5,058/7,060 |
| SkateFormer adapted small | 84,17% | 83,47% | 84,81% | 83,54% | 84,05% | 201/203/40/36 | 22.820 | 6,772/10,012 |

SkateFormer kém A3 5,21 điểm phần trăm Accuracy và 7,59 điểm phần trăm Sensitivity, tính từ số chưa làm tròn. Ít tham số nhất và checkpoint mẫu 101.711 byte, nhưng đắt hơn về thời gian suy luận. Phép đo không tách riêng chi phí từng phép gom/phân vùng, nên chưa quy nguyên nhân độ trễ cho một thành phần.

Accuracy SkateFormer theo seed 42/142/242 là 83,75% / 83,13% / 85,63%. Theo người S1–S4 là 90,63% / 79,17% / 90,70% / 77,48%. Chênh lệch Accuracy so với A3 lớn nhất ở S2 (−12,50 điểm phần trăm) và hòa ở S3. Sensitivity trên S4 của SkateFormer là 82,35%, cao hơn A3 80,39%, dù báo nhầm nhiều hơn.

SkateFormer bỏ sót `S1_FALL_13`, `S2_FALL_08`, `S2_FALL_14`, `S2_FALL_18`, `S4_FALL_05` ở cả ba seed. Báo nhầm lặp ở `S2_ADL_22`, `S2_ADL_19`, `S2_ADL_12`, `S4_ADL_07`, `S4_ADL_06`, `S4_ADL_16`, `S4_ADL_05`, `S4_ADL_10`.

[Theo người](comparison/comparison_fixed_by_subject.csv) · [Theo seed](comparison/comparison_fixed_by_seed.csv) · [Ma trận nhầm lẫn](comparison/comparison_fixed_confusion.csv) · [Lỗi theo clip](comparison/comparison_fixed_errors.csv) · [Dự đoán ghép cặp](comparison/comparison_fixed_paired.csv) · [Đo suy luận](comparison/comparison_latency.csv).

Kết quả mang tính khám phá trên bốn người đã xem. Các lượt lặp theo seed cùng dùng 160 clip. Các cửa sổ dùng bảng nhóm cố định theo giải phẫu và kích thước lưới, nên một số cửa sổ trộn khớp khác vùng. Kết luận áp dụng cho bản thu gọn này và chế độ huấn luyện đã khóa.
