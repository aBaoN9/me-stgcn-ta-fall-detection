# Kiểm tra kết quả notebook 07 — 30/09/2026

Nguồn: `output/07_controlled_ablation_full.zip` do người dùng tải từ Kaggle. Đây là kết quả chạy full trên 160 video đã dùng trong quá trình phát triển; không phải kiểm thử trên bộ dữ liệu ngoài hoàn toàn mới. Model web 06 chưa được thay.

## 1. Kiểm tra tính đầy đủ

- ZIP đọc được, kiểm tra CRC không lỗi; 313 mục, gồm 60 checkpoint `.pt`, 60 tệp attention `.npz`, 64 báo cáo outer (60 mạng + 4 SVM), các lịch sử train và CSV tổng hợp.
- `completion.json`: `complete=true`, `mode=full`, `outer_runs=64`. `locked_protocol.json`: GPU CUDA, seed 42/142/242, tối đa 50 epoch, patience 10, 4 outer subject và 3 inner subject. Thời gian của lần gọi train theo `timing.json`: khoảng 891 giây (14 phút 51 giây), không bao gồm mọi thao tác Kaggle.
- SHA-256 của cả bốn đầu vào trùng với gói `07-inputs-verified.zip` đã chuẩn bị. Bảng dự đoán có 7.680 hàng = 160 video × (5 cấu hình × 3 seed + SVM) × 3 chính sách; không có khóa mẫu/cấu hình/seed/chính sách trùng. Cùng một mẫu giữ nguyên nhãn và subject. 965 hàng trong `errors.csv` khớp chính xác các dự đoán sai. `fold_metrics.csv` đủ 192 hàng, không thiếu các chỉ số chính.
- `seed_mean_std.csv` là thống kê qua seed, không phải cộng 3 seed thành 480 mẫu độc lập. SVM chỉ chạy một cấu hình, không có độ lệch chuẩn qua seed.

## 2. Kết quả kiểm chứng trên 160 video

Trung bình ± độ lệch chuẩn mẫu của **ba seed**, dùng chính sách **fixed** (ngưỡng 0,5 cho mạng); mỗi seed tổng hợp bốn outer fold. FN là Fall bị bỏ sót; FP là ADL bị báo nhầm. Số FN/FP bên dưới lần lượt theo seed 42/142/242.

| Mô hình | Accuracy | Sensitivity | Specificity | FN | FP |
|---|---:|---:|---:|---|---|
| A0 pose + average | 84,58 ± 4,73% | 85,65 ± 9,33% | 83,54 ± 0,71% | 3 / 17 / 14 | 13 / 13 / 14 |
| A1 motion + average | 89,38 ± 1,08% | 91,14 ± 2,53% | 87,65 ± 4,45% | 9 / 5 / 7 | 6 / 13 / 11 |
| A2 pose + attention | 87,71 ± 0,72% | 93,25 ± 0,73% | 82,30 ± 0,71% | 5 / 5 / 6 | 14 / 14 / 15 |
| A3 motion + attention | 89,58 ± 3,44% | 91,98 ± 3,65% | 87,24 ± 3,97% | 3 / 8 / 8 | 8 / 9 / 14 |
| A3_cost_225 | 89,58 ± 0,36% | 96,62 ± 0,73% | 82,72 ± 1,23% | 3 / 3 / 2 | 13 / 14 / 15 |
| SVM v2 (một lượt) | 85,00% | 88,61% | 81,48% | 9 | 15 |

Không cấu hình nào có accuracy trung bình ≥95%. Trong phép so sánh có kiểm soát cùng recipe, motion cải thiện rõ so với A0. Thêm attention lên A1 chỉ tăng accuracy trung bình khoảng 0,21 điểm phần trăm ở fixed và không nhất quán qua seed: A3 đạt 93,13 / 89,38 / 86,25%. Không thể tuyên bố attention chắc chắn cải thiện mô hình.

So với A3 fixed, tăng trọng số Fall lên 2,25 làm sensitivity trung bình tăng từ 91,98% lên 96,62%, nhưng specificity giảm từ 87,24% xuống 82,72%. FN giảm từ 3/8/8 xuống 3/3/2; FP tăng từ 8/9/14 lên 13/14/15. Đây là đánh đổi, không phải cải thiện đồng thời cả hai phía. A3_cost_225 **vẫn bỏ sót Fall ở cả ba seed**.

Chính sách `sensitivity` được chọn trên inner validation, nhưng không đảm bảo đạt mục tiêu trên outer-test. Ví dụ A3_cost_225 trung bình sensitivity 96,20%, specificity 76,13%; ở inner, 10/12 outer run của cấu hình này đạt cả hai mục tiêu, còn 2/12 không đạt. A3 thường có inner mục tiêu đạt 12/12 nhưng sensitivity outer trung bình theo chính sách ấy chỉ 94,51%. Không lấy kết quả validation làm bảo đảm an toàn.

## 3. Những lỗi đáng xem đầu tiên

- `S4_FALL_05` là **FN của A3_cost_225 trong cả ba seed**; điểm Fall 0,221 / 0,059 / 0,246. Model 06 trước đây nhận đúng video này theo checkpoint/threshold cũ. Đây là trường hợp cần xem lại video, pose và nhãn, không được tự sửa nhãn hoặc đặt ngưỡng theo chính video outer-test.
- `S3_FALL_06` là FN của A3 thường trong cả ba seed; cấu hình tăng trọng số đã nhận đúng nó. `S2_FALL_03` là FN của A3 thường ở hai seed, còn cấu hình cost bỏ sót ở seed 42.
- Với A3_cost_225, các ADL `S1_ADL_02`, `S1_ADL_03`, `S3_ADL_05`, `S4_ADL_05`, `S4_ADL_06` đều bị báo nhầm trong cả ba seed. Bản 06 cũng báo nhầm ít nhất `S1_ADL_03` và `S4_ADL_05`; nên đối chiếu hành động thực tế thay vì quy hết cho lỗi pose.
- Cost không chỉ "cứu" các FN: ở seed 42 nó nhận đúng thêm `S2_FALL_13`, `S3_FALL_06`, nhưng **mới bỏ sót** `S2_FALL_15` và `S4_FALL_05` so với A3. Thay hàm mất mát thay cả ranh giới quyết định, không phải chỉ dịch ngưỡng.

## 4. Kết luận và bước tiếp theo

1. **Chưa thay model web 06.** Bản 06 có 87,50% accuracy và 0/79 FN trên đánh giá cũ; notebook 07 dùng pipeline và quy trình khác, nên các con số không phải phép đối chứng trực tiếp để tuyên bố v2 hơn/kém v1. Model 07 chưa có checkpoint train trên toàn bộ 160 video và chưa có ONNX triển khai.
2. Giữ nguyên ZIP gốc và phân tích video/pose của các FN lặp lại, đặc biệt `S4_FALL_05`; xem cả FP và thời điểm attention, không diễn giải attention như chứng cứ nhân quả.
3. Nếu ưu tiên giảm bỏ sót, A3_cost_225 fixed là **ứng viên để nghiên cứu tiếp**, không phải model an toàn đã xác nhận. Cần kiểm tra trên dữ liệu người/cảnh quay **độc lập** và đặt tiêu chí báo nhầm có thể chấp nhận trước khi chọn để triển khai.
4. Không chọn seed 42 riêng vì nó đạt 93,13%; lựa chọn bằng outer-test sau khi đã xem số liệu sẽ gây thiên lệch. Trước khi có tập ngoài và quy trình chốt model, không quảng bá mốc 95% hoặc khả năng hỗ trợ quyết định y tế.
