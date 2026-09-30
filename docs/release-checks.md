# Kiểm tra bản đóng gói GitHub — 30/09/2026

## Đã kiểm tra trên Windows

- Hash tất cả artifact deployment 06 khớp manifest; ONNX giống từng byte với bundle gốc. Suy luận trên sample input giữa ONNX gốc và bản đóng gói có sai khác tối đa bằng 0.
- Đủ tám notebook 01–07 và 04b; nbformat validation hợp lệ. Code cell giống nguồn; đã xóa output và execution count, thêm ghi chú trạng thái. Không chạy lại các notebook lịch sử để tuyên bố tái lập kết quả.
- Ba source feature/model/training của research07 giống byte nguồn. Validation bằng 160 input private đạt: parity legacy sai khác tối đa 1,19 × 10⁻⁷; 100 trường hợp kiểm tra augmentation nhất quán; shape cả năm biến thể hợp lệ.
- Kết quả full 07: 64 outer report, 192 dòng fold metric, 48 dòng pooled, 18 dòng mean/std, 7.680 prediction không trùng khóa; errors khớp các prediction sai. Đây là kiểm tra output đã tải, không phải lượt train mới.
- Các liên kết Markdown local hợp lệ. Đề cương PDF có 26 trang; là bản lưu theo trạng thái trong document-status.md, không tự cập nhật nội dung Word gốc.
- Web đóng gói được khởi động ở cổng 8766, runtime cách ly; web gốc cổng 8765 không bị thay đổi. GET trang chính, batch, research, API config, hai biểu đồ và CSV đều thành công. POST thiếu header local bị từ chối với HTTP 403.
- Một clip ADL private 170 frame được xử lý thành công cả whole-clip và timeline. Whole-clip khoảng 8,03 giây; timeline khoảng 8,50 giây với 6 cửa sổ. Đây là kiểm tra chức năng trên máy hiện tại, không phải đánh giá accuracy mới hoặc benchmark đại diện.
- Trang research đã được xem trên trình duyệt: bảng số liệu, cảnh báo và biểu đồ hiển thị. Ngưỡng web vẫn 0,425; model vẫn 06; không triển khai weights 07.
- Danh sách file Git được rà soát để loại video, ZIP, input pose/features, virtual environment, runtime, MediaPipe task, token/hóa đơn và checkpoint theo fold. Chỉ giữ checkpoint final 06; attention NPZ của 07 là output nghiên cứu, không phải input pose private.

## Chưa xác nhận

- Chưa cài môi trường từ đầu trên một máy Windows sạch; thử nghiệm sử dụng môi trường Python web/research có sẵn. Chưa chạy trên Linux/macOS.
- Chưa train lại tám notebook sau đóng gói, chưa có dữ liệu held-out mới và chưa kiểm định webcam/cảnh báo theo thời gian.
- Chưa kiểm chứng độc lập checkpoint final 06; không suy ra sensitivity 100% cho video mới từ kết quả OOF cũ.
- Nhãn mốc ngã của sáu clip URFD còn provisional; cần người gán nhãn độc lập trước khi dùng làm bằng chứng định lượng chính thức.

Các giới hạn này không được xóa chỉ vì bản web chạy được. Bản phát hành là lưu trữ nghiên cứu và demo, không phải chứng nhận an toàn.
