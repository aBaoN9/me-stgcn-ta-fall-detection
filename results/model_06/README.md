# Kết quả 06 và checkpoint đang triển khai

OOF nested LOSO trên 160 video: accuracy 87,50%, sensitivity 100% (79/79), specificity 75,31% (61/81), FP20/FN0. `outer_test_predictions_160.csv` và `outer_fold_results.csv` đến từ bundle gốc; `oof_diagnostics_160.csv` đã được đối chiếu lại bằng checkpoint đúng fold ở lượt audit.

`models/deployment_06/` chứa checkpoint final học cả 160 video và ONNX tương ứng, không phải một checkpoint outer-fold. Không dùng metric OOF để khẳng định final model chưa từng học video đánh giá. Ngưỡng deployment 0,425 lấy từ quy trình inner-selection đã ghi trong config, không đặt lại theo video test ngoài.

`false_positives_20_reviewed.csv` có quan sát trên ảnh mẫu và mô tả nguồn. Đây không phải đánh giá nhãn độc lập; không sửa ADL thành Fall để xóa lỗi. 20 FP có pose detection trung bình khoảng 99,17%, không thể quy mọi lỗi cho mất pose.

Lượt đóng gói trích nguyên ONNX/PT/config, không retrain. Web nay đọc file ONNX trực tiếp; `model_sha256` của kết quả mới là hash ONNX. Các báo cáo cũ ghi hash **toàn bundle**; mapping được ghi trong `artifact_manifest.json`, không coi hai loại hash khác nhau là hai model khác nhau.
