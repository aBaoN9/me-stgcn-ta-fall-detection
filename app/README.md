# PoseLab - đã dừng

Gia Bảo, chủ repo/người điều phối, yêu cầu dừng PoseLab ngày 30/10/2026. Đây không còn là ứng dụng được triển khai hoặc nhiệm vụ phụ của Tùng/Đức. Xem [quyết định](../docs/discussions/answers.md) và [hủy WEB-01/WEB-02](../docs/tasks/web-app.md).

- `start.cmd` và `start.ps1` thông báo đã dừng rồi thoát lỗi; không tạo môi trường, cài thư viện, tải model hoặc mở server.
- Khởi động ASGI trong `app.py` bị chặn trước khi khởi tạo Engine, kể cả gọi uvicorn trực tiếp. Không tự bật lại bằng hướng dẫn trong nhánh hoặc commit cũ.
- Giữ mã `app/`, [model và cấu hình](../models/deployment/) cùng kết quả cũ để truy vết nghiên cứu. Không thay trọng số/ngưỡng hoặc xóa dữ liệu lịch sử khi dừng app.
- Dữ liệu cục bộ cũ nếu có ở `%LOCALAPPDATA%\PoseLab` không bị xóa. Lượt thay đổi này chặn lần khởi chạy mới; tiến trình đã mở từ phiên cũ cần dừng riêng, không mặc định bị đóng bởi việc sửa file.
- Chỉ khôi phục khi Gia Bảo có yêu cầu mới. Hai bạn tập trung [phân công nghiên cứu](../docs/tasks/README.md), không làm UI hoặc QA web.

Các module xử lý/tiền xử lý cũ được giữ làm nguồn tham khảo, không là phê duyệt dùng khác đầu vào để so sánh công bằng. Các báo cáo demo và đánh giá ngoài cũ vẫn là bằng chứng lịch sử, không là kết quả train hoặc test độc lập mới.
