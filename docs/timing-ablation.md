# Kiểm chứng khoảng thời gian trong đặc trưng vận tốc

## Thiết kế

Hai nhánh dùng chung ME-STGCN-TA tám kênh, chú ý thời gian và hệ số bổ sung lớp ngã 2,25:

- **T0:** dùng thời lượng video chia 31 cho mọi cặp mốc.
- **T1:** dùng chênh lệch chỉ số hai frame được chọn chia FPS cho từng cặp.

Chỉ mẫu số của hai kênh vận tốc thay đổi. Sáu kênh khác, cách chia chủ thể, seed và tăng cường được giữ tương ứng. Đây là thời gian danh định frame/FPS, không phải timestamp camera đo được; vận tốc chuẩn hóa không có đơn vị m/s.

Full đã hoàn tất: hai nhánh × ba seed × bốn chủ thể = 24 lượt đánh giá ngoài, 96 lần huấn luyện tính cả vòng trong và huấn luyện lại. Kết quả chính dùng ngưỡng cố định 0,5.

## Kết quả

| Nhánh | Accuracy ± SD (%) | Sensitivity (%) | Specificity (%) |
|---|---:|---:|---:|
| T0 · Khoảng chung | 89,58 ± 0,36 | 96,62 | 82,72 |
| T1 · Khoảng từng cặp | 87,71 ± 2,60 | 94,09 | 81,48 |

| Seed | FN T0 | FN T1 | FP T0 | FP T1 |
|---|---:|---:|---:|---:|
| 42 | 3 | 5 | 13 | 10 |
| 142 | 3 | 3 | 14 | 18 |
| 242 | 2 | 6 | 15 | 17 |

T1 chưa cải thiện ổn định. Ở seed 42, giảm báo nhầm đi kèm tăng bỏ sót; không lấy riêng seed này để kết luận cải tiến. Các seed vẫn đánh giá cùng 160 video, không phải 480 mẫu độc lập.

## Phạm vi kết luận

Đây là so sánh phát triển trên dữ liệu đã quan sát, chưa phải xác nhận độc lập trên người/cảnh mới. Thí nghiệm chưa tách riêng tác động làm tròn chỉ số và sự khác biệt F/FPS với (F−1)/FPS. Không thay mô hình hoặc tiền xử lý đang triển khai từ kết quả này.

[Notebook kiểm chứng thời gian](../notebooks/08-timing-ablation.ipynb) · [Dự đoán, cấu hình và bảng kết quả](../results/timing_ablation/README.md) · [Báo cáo Word](bao-cao.docx).
