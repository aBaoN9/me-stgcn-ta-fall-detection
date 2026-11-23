# Nhật ký dự án

[README ở gốc repo](../../README.md) là bản hiện trạng: model đã nhận, kết quả so sánh mới nhất, cách chạy và giới hạn. Thư mục này giữ diễn biến các lần cập nhật, không thay thế trạng thái hiện hành hoặc giao thức đã khóa.

## Tra cứu

| Tài liệu | Nội dung |
|---|---|
| [Nhật ký tháng 11/2026](2026-11.md) | Phân công/quyền tự chủ, input, các lần nghiệm thu benchmark, gỡ PoseLab, FallScope và tổ chức lại README |
| [Bản lưu README main tại ba1b24b](README-main-ba1b24b.md) | Toàn bộ README trước lần tái cấu trúc: các khối cập nhật cũ, bảng lịch sử, số đo latency từng lượt và hướng dẫn trước đó |
| [Quyết định nhóm](../discussions/answers.md) | Nội dung quyết định có ID và phạm vi cấp quyền; nhật ký chỉ dẫn lại, không tự sửa quyền |
| [Báo cáo cá nhân](../reports/) | Chi tiết công việc, bằng chứng và trạng thái theo từng thành viên |

Bản lưu có ghi commit nguồn. Liên kết tương đối được điều chỉnh để vẫn mở từ thư mục nhật ký; không sửa nội dung nghiên cứu hoặc làm cho trạng thái cũ trông như trạng thái hiện tại. Những mốc trước khi có thư mục này được tổng hợp từ lịch sử Git và quyết định đã công bố, không phải tạo lại log chạy còn thiếu.

## Quy tắc cập nhật

1. **Sửa trực tiếp hiện trạng trong README gốc.** Thay bảng/kết luận/khả năng chạy đã lỗi thời bằng thông tin mới được kiểm chứng; không xếp thêm các khối ngày tháng ở đầu README.
2. Ghi thay đổi vào file tháng theo mẫu `YYYY-MM.md`, theo thời gian UTC+7. Cùng ngày có nhiều thay đổi thì tách theo chủ đề, không ghi đè mất mốc trước.
3. Mỗi mục nêu ngày, thay đổi và trạng thái cũ → mới, nguồn commit/PR/run, kiểm tra, giới hạn và điều không thay đổi. Chỉ ghi commit/PR đã tồn tại; không tự bịa SHA cho commit đang chuẩn bị.
4. Khi dọn/rewrite lớn, lưu bản README trước đó với tên chứa commit nguồn; không chép toàn bộ checkpoint hoặc dataset vào nhật ký. Các mốc và số liệu cũ chỉ dùng truy vết.
5. Không coi notebook có outputs, quyền chạy hoặc log PARTIAL là đã đủ model/kết quả nghiệm thu. Không xóa/sửa artifact đã khóa khi viết lại tài liệu; không lưu token, video riêng hoặc log chứa bí mật.

## Ranh giới tài liệu

- README gốc: **đang có gì và dùng thế nào**.
- Nhật ký: **đã thay đổi gì, khi nào và từ đâu**.
- `docs/reports/`: công việc/bằng chứng cá nhân; `docs/discussions/answers.md`: quyết định điều phối.
- `results/`: bằng chứng thực nghiệm gốc; `docs/research/`: thiết kế/đăng ký phương án. Không thay các tài liệu này bằng bản tóm tắt trong nhật ký.
