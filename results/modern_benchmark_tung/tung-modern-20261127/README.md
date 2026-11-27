# GB-20261124-001 — 5 mô hình và A3 trên RTX 3050

27/11/2026 UTC+7 · **PASS 288 full + 24 smoke**, 72 checkpoint full và 6 checkpoint smoke. A3 được huấn luyện lại. Cùng RTX 3050 Laptop 4 GiB, driver 566.07, Python 3.11.9, PyTorch 2.10.0+cu126, FP32, tắt TF32, bốn thread. [Thiết kế và nguồn](../../../docs/research/modern-benchmark-tung.md) · [Lệnh tái lập](../../../training/modern_benchmark_tung/README.md) · [Danh mục SHA-256](files_sha256.json).

## Bảng chính — ngưỡng cố định 0,5

Tỷ lệ là trung bình ba seed; TP/TN/FP/FN cộng trên 480 lượt clip–seed của mỗi mô hình. Các bảng chi tiết giữ đủ ba chính sách quyết định.

| Mô hình | Accuracy | Balanced accuracy | Precision | Sensitivity | Specificity | F1 Fall | F2 | Clip FPR | TP/TN/FP/FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A3 | 88,96% | 88,97% | 87,75% | 90,30% | 87,65% | 89,00% | 89,77% | 12,35% | 214/213/30/23 |
| CascadeFormer | 86,67% | 86,70% | 84,63% | 89,45% | 83,95% | 86,91% | 88,40% | 16,05% | 212/204/39/25 |
| DHA-eGCN | 83,33% | 83,31% | 84,66% | 81,01% | 85,60% | 82,62% | 81,61% | 14,40% | 192/208/35/45 |
| Hyper-GCN | 87,29% | 87,31% | 86,20% | 88,61% | 86,01% | 87,29% | 88,05% | 13,99% | 210/209/34/27 |
| ProtoGCN | 86,88% | 86,92% | 83,96% | 90,72% | 83,13% | 87,17% | 89,26% | 16,87% | 215/202/41/22 |
| InfoGCN++ | 90,63% | 90,64% | 89,68% | 91,56% | 89,71% | 90,61% | 91,18% | 10,29% | 217/218/25/20 |

- InfoGCN++ hơn A3 1,67 điểm phần trăm Accuracy và 1,27 điểm phần trăm Sensitivity theo số chưa làm tròn, giảm 3 FN và 5 FP. Accuracy ở S4 là 87,39% so với A3 81,08%; ở S2 là 88,19% so với 91,67%. InfoGCN++ đạt 90,625% ở cả ba seed, còn A3 cao hơn tại seed 242 (91,25%).
- CascadeFormer nhanh nhất trong phép đo nhưng Accuracy thấp hơn A3 2,29 điểm phần trăm. ProtoGCN ít tham số nhất, giảm một FN và tăng 11 FP. Hyper-GCN tăng Accuracy ở người S3/S4 nhưng giảm trung bình. DHA-eGCN có nhiều lần bỏ sót nhất.
- InfoGCN++ bỏ sót `S2_FALL_03`, `S2_FALL_18`, `S3_FALL_10` và báo nhầm `S2_ADL_12`, `S2_ADL_19`, `S4_ADL_05`, `S4_ADL_06`, `S4_ADL_07`, `S4_ADL_16` ở cả ba seed. A3 bỏ sót `S4_FALL_05` ở cả ba seed. Cả sáu mô hình đều báo nhầm `S4_ADL_05` ở cả ba seed.

[Theo seed](comparison/by_seed.csv) · [Theo người và seed](comparison/by_subject_seed.csv) · [Mean và sample SD](comparison/mean_sd.csv) · [Ma trận nhầm lẫn](comparison/confusion.csv) · [Chênh lệch ghép cặp theo seed](comparison/paired_delta_seed.csv) · [Chênh lệch ghép cặp theo người](comparison/paired_delta_subject_seed.csv) · [Lỗi lặp](comparison/recurring_fixed_errors.csv) · [Tất cả lỗi](comparison/errors_all_policies.csv).

## Tài nguyên

Suy luận chỉ đo forward mô hình, cùng S1_ADL_01 và checkpoint seed 42/người 1, batch 1, FP32, 30 lượt làm nóng và 200 lượt đồng bộ GPU. VRAM và RAM ở bảng là toàn hệ thống theo mẫu monitor trong full.

| Mô hình | Full (giây) | °C | VRAM MiB | RAM MiB | GPU % | Tham số | Checkpoint mẫu byte | Median/p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A3 | 224,66 | 80 | 297 | 14.733,0 | 83 | 422.871 | 1.742.362 | 2,852/3,433 |
| CascadeFormer | 113,98 | 65 | 123 | 14.084,2 | 42 | 61.874 | 257.054 | 1,412/1,584 |
| DHA-eGCN | 239,20 + khoảng 44,41 | 73 | 219 | 14.418,3 | 57 | 15.682 | 85.971 | 2,488/2,971 |
| Hyper-GCN | 387,47 | 60 | 155 | 15.577,6 | 61 | 18.866 | 100.785 | 3,273/3,716 |
| ProtoGCN | 281,19 | 65 | 173 | 13.888,9 | 50 | 10.328 | 56.772 | 2,158/2,325 |
| InfoGCN++ | 327,61 | 74 | 265 | 13.724,1 | 63 | 12.474 | 64.788 | 2,823/3,369 |

DHA-eGCN có một lần gián đoạn do Windows từ chối đổi tên thư mục sao lưu. 44,41 giây là mốc đo cuối trước gián đoạn, 239,20 giây là phiên tiếp tục. Vòng đã đạt được nạp lại và không chạy lại. Thời gian cộng không bao gồm khoảng chờ giữa hai phiên. [Đo suy luận chi tiết](comparison/latency.csv).

## Kiểm chứng

Bốn tệp đúng hash, validation 160 clip, 25/25 kiểm thử và trình kiểm tra kho mã đều đạt. Audit đối chiếu bản sao từng vòng, nạp lại 72 checkpoint full và 6 checkpoint smoke, kiểm đủ 8.640 dự đoán ngoài mẫu full và tính lại chỉ số. Mỗi thư mục mô hình có khóa, checkpoint, lịch sử từng vòng, dự đoán validation bên trong vòng và log giám sát.

[A3](A3_motion_attention/README.md) · [CascadeFormer](cascadeformer_v2_ce/README.md) · [DHA-eGCN](dha_egcn_v2_ce/README.md) · [Hyper-GCN](hyper_gcn_v2_ce/README.md) · [ProtoGCN](proto_gcn_v2_ce/README.md) · [InfoGCN++](infogcnpp_v2_ce/README.md).

Bốn người đã xuất hiện trong nghiên cứu trước và ba seed lặp lại cùng 160 clip. Phép so sánh áp dụng cho các bản thu gọn huấn luyện với cross-entropy, riêng InfoGCN++ được đo trên toàn clip. Cần dữ liệu độc lập để kiểm tra khả năng khái quát.
