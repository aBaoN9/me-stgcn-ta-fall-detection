# Trạng thái tài liệu ngày 30/09/2026

`de_cuong.docx` và PDF là bản sao đề cương kỹ thuật đã được duyệt bố cục, không sửa trang đầu hay tài liệu gốc trong lượt đóng gói. Đây là ảnh chụp tài liệu **trước khi kết quả full notebook 07 được trả về**, không phải báo cáo thực nghiệm cuối đã viết lại toàn bộ. Một số câu trong đó còn mô tả thí nghiệm 07 là kế hoạch.

Trạng thái hiện hành được bổ sung bằng **README**, `report07.md`, `methodology.md` và các CSV trong `results/notebook_07/`: full 07 đã hoàn tất 64 outer run; không thay model web. Khi có khác biệt về trạng thái thực thi, dùng phần bổ sung này, không trích câu “chưa chạy full” như trạng thái hiện tại.

Đề cương có thông tin nhóm học tập trên trang đầu. Repo đang Private; cần rà soát quyền riêng tư và quyền dùng hình/tài liệu trước khi đổi sang Public. Không có tuyên bố toàn bộ tài liệu tham khảo trong đề cương đã được kiểm chứng lại ở lượt đóng gói này.

## Bổ sung đã duyệt ngày 01/10/2026

[Báo cáo bộ dữ liệu và kiểm tra chất lượng đầu vào, bản 2](bao_cao_dataset_2.docx) là phần bổ sung cho mục 3.2, không thay thế hoặc sửa các trang đã duyệt của đề cương. Báo cáo gồm nguồn và phân bố 160 video, ý nghĩa nhãn, độ bao phủ pose, biểu diễn đầu vào, giao thức nested LOSO và kiểm tra định tính chín ca. Tài liệu đã được kiểm tra đủ 11 trang, với 5 bảng, 2 hình và 2 công thức Word chỉnh sửa được.

Nhận xét người kiểm tra và độ bao phủ pose không phải nhãn tọa độ khớp chuẩn hoặc bằng chứng mô hình hiệu quả trong thực tế. Bản này chưa phải báo cáo đầy đủ các thí nghiệm huấn luyện. Không kèm video, pose/features riêng tư hoặc bản nháp kiểm tra bố cục.

## Bổ sung huấn luyện và kết quả ngày 01/10/2026

[Báo cáo Word bản 4](bao_cao_dataset_4.docx) giữ nguyên nội dung mục 3.2 của bản 2 và thêm mục 3.3–3.4 về kiến trúc, thiết lập tối ưu, lựa chọn theo nested LOSO, chỉ tiêu, kết quả lịch sử và so sánh có kiểm soát. Số đếm nhầm lẫn của 48 dòng kết quả gộp được đối chiếu lại từ dự đoán từng video; trung bình và độ lệch chuẩn được tính lại. Bản nháp 3 không được đưa vào repo.

Bản 4 có 22 trang, 12 bảng, 3 hình và 5 công thức Word chỉnh sửa được; Times New Roman 13 pt, màu đen. Nội dung cũ được đối chiếu OOXML không thay đổi; mười trang đầu có ảnh render trùng từng byte với bản 2 đã kiểm tra. Những trang mới đã được kiểm tra bố cục.

Phần kiểm tra kỹ thuật phân biệt pose phát hiện trực tiếp với pose điền biên, đối chiếu phép tính đặc trưng và xem đủ 32 mốc của ca chuyển tiếp. Đây là kiểm tra sau khi đã biết lỗi, không phải bằng chứng một mô hình mới đã cải thiện. Chưa train lại, chưa đổi ngưỡng, chưa thay model web 06. Hồ sơ thực thi tải về của thử nghiệm ưu tiên độ nhạy lịch sử không đủ để báo cáo riêng, nên không tự tạo số liệu cho giai đoạn đó.
