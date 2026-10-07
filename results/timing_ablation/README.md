# Kết quả kiểm chứng thời gian

Kết quả full: hai nhánh T0/T1, ba seed, bốn chủ thể và ba chính sách ngưỡng. Có 24 lượt ngoài, 96 lần huấn luyện và 2.880 dự đoán; vẫn chỉ có 160 video độc lập.

| Tệp | Nội dung |
|---|---|
| `predictions.csv`, `errors.csv` | Dự đoán từng mẫu và các mẫu nhận sai |
| `fold_metrics.csv` | Kết quả theo chủ thể kiểm tra ngoài |
| `pooled_by_seed.csv`, `seed_mean_std.csv` | Kết quả gộp và trung bình/độ lệch chuẩn qua seed |
| `paired_predictions.csv`, `paired_metrics.csv`, `paired_delta_mean_std.csv` | Đối chiếu T1 với T0 trên cùng mẫu và seed |
| `inner_epoch_selection.csv` | Epoch được chọn ở vòng trong |
| `locked_protocol.json`, `input_provenance.json`, `completion.json`, `timing.json` | Cấu hình, hash input, trạng thái hoàn tất và thời gian chạy |

CSV tổng hợp dùng hai cấp tiêu đề và dòng tên chỉ mục. Với pandas, đọc `seed_mean_std.csv` bằng `header=[0, 1], index_col=[0, 1]`; đọc `paired_delta_mean_std.csv` bằng `header=[0, 1], index_col=0`. Bảng dự đoán và kết quả từng lượt dùng một hàng tiêu đề.

Không đóng gói ZIP, video, pose/features, checkpoint từng fold hoặc kết quả smoke. Notebook tự chứa mã và bằng chứng lấy mẫu cần thiết. T0/T1 chưa tạo mô hình triển khai thay thế.

[Báo cáo kết quả và giới hạn](../../docs/timing-ablation.md) · [Notebook](../../notebooks/08-timing-ablation.ipynb).
