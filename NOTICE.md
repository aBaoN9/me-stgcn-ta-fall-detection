# Điều kiện sử dụng

Dự án phục vụ nghiên cứu phát hiện té ngã từ video. Chưa chọn giấy phép phát hành chung; việc truy cập repo không tự cấp quyền tái phân phối mã nguồn, trọng số, tài liệu hoặc dữ liệu của bên thứ ba.

MediaPipe, PyTorch, ONNX Runtime và các thư viện dùng trong nghiên cứu/xuất mô hình có giấy phép riêng. Mã app PoseLab và requirements của app đã gỡ ngày 21/11/2026; không có bước chuẩn bị/chạy demo hiện hành. Pose Landmarker không đóng gói trong Git. Video gốc GMDCSA-24, URFD và dữ liệu riêng khác không được tái phân phối trong repo.

Theo yêu cầu của chủ repo ngày 19/11/2026, bốn tensor/CSV đầu vào dẫn xuất từ notebook 01 được chia sẻ tại [data/gmdcsa24/](data/gmdcsa24/) để tái lập nghiên cứu, không phải video gốc. Nguồn: Ekram Alam, **GMDCSA-24**, `v2.1`, [DOI 10.5281/zenodo.13354453](https://doi.org/10.5281/zenodo.13354453); metadata nguồn công bố [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Giữ ghi công, thông báo về xử lý dẫn xuất và tuân thủ giấy phép nguồn; xem [nguồn/hash/phạm vi chia sẻ](data/README.md). Ngoại lệ này không cấp giấy phép chung cho mã nguồn, model hoặc dữ liệu bên thứ ba khác của repo.

Sơ đồ là minh họa phương pháp; bảng và biểu đồ kết quả dựa trên các dự đoán đã lưu. Ảnh ứng dụng trong tài liệu/lịch sử chỉ minh họa giao diện cũ, không phải bằng chứng hiệu quả lâm sàng hoặc app đang hoạt động. Hệ thống chưa được kiểm chứng như thiết bị y tế hoặc công cụ thay thế việc chăm sóc và giám sát con người.
