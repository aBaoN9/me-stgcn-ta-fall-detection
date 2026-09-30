# Sáu clip URFD — development, không phải held-out test

Ba clip ADL và ba clip Fall đã được xem khi phát triển. Source label giữ nguyên; mốc bắt đầu ngã trong `annotations.json` do AI ước lượng từ RGB, chưa được người gán nhãn độc lập duyệt và không blind với kết quả model.

Theo các mốc tạm của lượt 30/09/2026: TP3, FP1, FN0, TN2; delay ba TP khoảng 1,0 / 2,0 / 1,9 giây. Đây là kiểm tra chức năng cửa sổ trên sáu clip, không phải chứng nhận sensitivity 100% ngoài thực tế. Delay từ mốc đến cửa sổ cảnh báo không phải latency xử lý backend.

ADL-01 đúng ở phân loại toàn clip nhưng báo nhầm ở cửa sổ 3 giây. Fall-02 xảy ra trong warmup; điều này minh họa vì sao model toàn clip không tự bảo đảm cảnh báo sớm. Không chỉnh ngưỡng để sửa riêng những clip đã xem.

Không chứa video hoặc ảnh RGB của người tham gia. Chỉ giữ bảng báo cáo, nhãn tạm và ảnh giao diện trong `docs/images/batch-demo.png`. Hash model của báo cáo cũ là hash bundle 06, xem `models/deployment_06/artifact_manifest.json`.
