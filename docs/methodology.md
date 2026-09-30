# Phương pháp, phạm vi và bằng chứng

## Chuỗi xử lý

Video RGB → MediaPipe Pose Full (VIDEO, một người) → đọc liên tục mọi frame → nội suy tọa độ thiếu → lấy đều 32 mốc → 33 khớp → đặc trưng 8 kênh → bốn block ST-GCN → temporal attention → logits hai lớp và điểm Fall. Đây là một luồng skeleton, không phải CNN–LSTM hoặc Transformer.

Các kênh gồm x/y tương đối tâm hông, visibility, vận tốc x/y của từng khớp tương đối, góc thân và dịch chuyển hông x/y. Vận tốc chia thời gian tương ứng, không phải m/s. Không có kênh gia tốc trong model đang triển khai. Khớp dùng graph MediaPipe 33 node; adjacency và edge importance mô tả quan hệ khớp.

## Phân biệt 06 và 07

- **06:** sửa đọc góc thân ở global feature index 9, nested leave-one-subject-out. Mỗi outer giữ một người; inner chọn epoch/ngưỡng trên những người còn lại. Checkpoint final train trên toàn bộ 160 video dùng cho web, ngưỡng 0,425. Điểm OOF là từ model theo fold, không phải test độc lập của checkpoint final.
- **07:** giữ bốn block và graph 33 khớp; sửa visibility ở mốc mất pose và augmentation trước khi tính motion. So sánh A0/A1/A2/A3 cùng recipe; A3_cost_225 là thay đổi hàm mất mát riêng. Ba seed, bốn outer và ba inner. Mỗi model dùng lại cùng score cho fixed/balanced/sensitivity; ngưỡng lựa chọn chỉ từ inner.
- Bộ pose 32 mốc đã nội suy từ 01; v2 không khôi phục được đầy đủ các khoảng mất pose gốc. 156 mốc trong 36 video bị ảnh hưởng. Không bỏ video khó để tăng metric.

## Web và cửa sổ thời gian

Toàn clip là bài toán model đã được xây dựng. Cửa sổ gần nhất 3 giây, bước 0,5 giây và hai lần xác nhận là **thử nghiệm** thay đổi đầu vào. Backend không dùng frame tương lai cho một cửa sổ đã kết thúc, nhưng đây là xử lý file, không phải webcam đã kiểm định. Khởi đầu cần tích lũy đủ cửa sổ; mốc ngã sớm có thể nằm trong warmup.

Không đủ pose → không kết luận, không tự gán ADL. Mốc ngã do người dùng nhập không phải mốc tự phát hiện. Không coi thời gian từ mốc ngã đến mốc cửa sổ báo động là toàn bộ latency xử lý trên máy. Đồ thị vận tốc/góc/hông mô tả đầu vào; attention mô tả trọng số thời gian của mạng, **không chứng minh nhân quả**.

## Hạn chế quan trọng

Chỉ 160 video của bốn người; clip có thể tương quan trong cùng người/bối cảnh. Ba seed không tăng số video độc lập. Các outer-test cũ đã được xem để phát triển, nên không gọi toàn bộ quá trình là test chưa từng sử dụng. Sáu clip URFD đã xem là development, mốc do AI ước lượng chưa có duyệt độc lập. Pose rate cao không đảm bảo từng khớp đúng.

Không hứa 95% accuracy, không chọn seed tốt nhất bằng outer-test, không coi điểm Fall là xác suất đã hiệu chuẩn. Chưa kiểm định đa người, người ngoài khung, đối tượng cao tuổi, môi trường thực tế, webcam hoặc khả năng dùng trong cấp cứu. Tài liệu hiện tại lưu cả lỗi; repo không tự nâng cấp model theo kết quả đẹp nhất.
