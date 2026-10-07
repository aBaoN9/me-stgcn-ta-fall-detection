# PoseLab — phân tích video và cảnh báo té ngã

## Khởi chạy

Cần Windows và Python 3.11. Mở `start.cmd` trong thư mục này hoặc chạy `.\app\start.cmd` từ thư mục gốc. Lần đầu chương trình tạo môi trường Python, cài thư viện và tải MediaPipe Pose Landmarker có kiểm tra SHA-256.

Giữ terminal mở và truy cập **http://127.0.0.1:8765**. Đổi cổng bằng `powershell -ExecutionPolicy Bypass -File .\app\start.ps1 -Port 8767`. Dừng server bằng Ctrl+C; mặc định server chỉ lắng nghe trên máy hiện tại.

## Chức năng

- **Phân tích video:** video, khung xương, Tổng hợp, Điểm Fall theo thời gian và Dữ liệu chuyển động hiển thị cùng lúc.
- **Lịch sử cảnh báo:** mở lại phiên đã lưu, đánh dấu đã xem/báo nhầm và ghi chú.
- **Nhiều video:** nhập nhãn, đối chiếu mốc ngã và tải bảng kết quả JSON/CSV.
- **Thông tin mô hình:** phân biệt mô hình triển khai với các thí nghiệm kiểm chứng.

Chọn video một người, chọn phân tích cửa sổ trượt hoặc phân loại toàn clip, rồi bấm Phân tích. Phát video hoặc kéo thanh thời gian để đối chiếu khung xương và các số đo. Khung xương hiển thị từ phát hiện trực tiếp, không vẽ pose nội suy như thể đó là quan sát thật.

## Mô hình và đầu vào

Ứng dụng dùng ME-STGCN-TA đã hiệu chỉnh góc thân: tám kênh, 32 mốc, 33 khớp và ngưỡng 0,425. Trọng số, cấu hình và manifest nằm trong [models/deployment](../models/deployment/). Không tự thay mô hình bằng checkpoint ngoài của các thí nghiệm A/T.

Toàn clip phù hợp đơn vị huấn luyện ban đầu. Cửa sổ mặc định dài 3 giây, bước 0,5 giây và cần hai lần xác nhận; đây là xử lý video file, chưa phải kiểm chứng camera trực tiếp. Không đủ pose dẫn tới trạng thái thiếu dữ liệu, không tự xem là ADL hoặc an toàn.

Chuyển động của thanh điểm là hiệu ứng giao diện; không làm mượt điểm thật, thay ngưỡng hoặc trì hoãn quyết định. Các đồ thị vận tốc, góc thân và hông mô tả đầu vào, không tự chứng minh nguyên nhân khiến mạng quyết định ngã.

## Video và dữ liệu lưu trữ

MediaPipe và mô hình nhận video gốc. Sau phân tích, ứng dụng có thể tạo bản WebM VP8 để trình duyệt phát được codec, giữ số frame/FPS danh định, giảm kích thước tối đa 1280×720 và không giữ âm thanh. Nếu chuyển mã không khả dụng, kết quả phân tích vẫn tồn tại nhưng video gốc có thể không phát được.

Dữ liệu được xử lý trên máy, không tải lên cloud. Runtime và lịch sử nằm trong `%LOCALAPPDATA%\PoseLab`; lịch sử dùng SQLite và các tệp video/báo cáo. Người dùng chủ động xóa phiên trong Lịch sử; thao tác này không xóa video gốc. Xóa phiên tạm không đồng nghĩa xóa lịch sử đã lưu.

`POSELAB_RUNTIME_DIR` chọn vị trí runtime; `POSELAB_EXAMPLE_VIDEO` chọn video mẫu có quyền sử dụng. Mặc định không tải video dataset hoặc mở webcam. Thông báo hiện lưu cục bộ; bộ điều hợp đã tách riêng nhưng Telegram chưa được tích hợp.

## Đánh giá nhiều video

Tối đa 30 video, 120 MB/tệp và 600 MB/lượt. Nhãn do người dùng nhập, không suy ra từ tên tệp. Mốc ngã chưa biết cần để trống; ứng dụng không tự tạo nhãn chuẩn. Báo cáo giữ cấu hình, hash mô hình, cửa sổ và mức bao phủ dữ liệu.

Các chỉ số đánh giá theo thời gian chỉ dùng những video đủ điều kiện, với mốc ngã hợp lệ cho Fall. Độ trễ trên các ca nhận đúng không bao gồm ca bỏ sót và chưa phải độ trễ toàn hệ thống; thống kê lịch sử không phải sensitivity hoặc báo nhầm/giờ. Sáu clip ngoài đã xem thuộc tập phát triển, không phải xác nhận độc lập.

## Thành phần chính

| Thành phần | Vai trò |
|---|---|
| `engine.py` | Trích xuất MediaPipe và suy luận ONNX |
| `features.py`, `motion.py` | Tiền xử lý, chuyển động và cửa sổ cảnh báo |
| `app.py` | API và điều phối xử lý |
| `event_store.py` | Lịch sử phiên/sự kiện và bộ điều hợp thông báo |
| `batch_evaluation.py` | Đánh giá nhiều video |
| `video_preview.py` | Tạo video tương thích trình duyệt |
| `static/` | Giao diện và tài nguyên cục bộ |

Xem [phương pháp và giới hạn](../docs/methodology.md). Mô hình đã học toàn bộ 160 video không có tập test độc lập riêng; ứng dụng chưa được kiểm chứng như thiết bị y tế/cấp cứu.
