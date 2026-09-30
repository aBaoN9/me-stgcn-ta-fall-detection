# Hướng train và cải tiến tiếp — chưa thực hiện

## Quyết định hiện tại

Giữ nguyên web/model 06 và lưu notebook 07 như bằng chứng kiểm chứng. Không train thêm trong lượt đóng gói GitHub, không chọn model/seed thắng outer-test để tự thay deployment. A3_cost_225 fixed là **ứng viên để nghiên cứu**, chưa phải model an toàn đã xác nhận: 96,62% sensitivity vẫn tương ứng 3/3/2 ca bỏ sót theo seed.

## Ưu tiên theo thứ tự

| Thứ tự | Việc nên làm | Vì sao | Điều kiện để đi tiếp |
|---|---|---|---|
| 1 | Xem lại video/pose và nhãn các lỗi lặp lại | `S4_FALL_05` bị cost bỏ sót cả ba seed; `S3_FALL_06` bị A3 thường bỏ sót cả ba seed | Ghi quan sát, không đổi nhãn chỉ vì model sai |
| 2 | Tạo dữ liệu ngoài theo người/bối cảnh, gán nhãn trước khi xem dự đoán | Bốn subject quá ít để kết luận mọi đối tượng; URFD đã xem không phải test mới | Tách development/validation/held-out theo người, không chia ngẫu nhiên các clip gần trùng |
| 3 | Lưu pose liên tục, timestamp, mask và các khoảng mất pose trước lấy 32 mốc | File 32 mốc hiện tại không khôi phục được đoạn thiếu; detection rate không chứng minh khớp đúng | Kiểm tra pipeline mới, giữ 06/07 làm baseline và version hóa |
| 4 | Nếu cần cảnh báo theo thời gian, train/validation cho cửa sổ có nhãn thời gian | Model toàn clip đang được áp vào rolling window khác phân phối | Gán mốc sự kiện, tránh gán toàn bộ cửa sổ của video Fall thành Fall |
| 5 | Thử cost/regularization hoặc pretrain có kiểm soát | Motion có ích, attention trên motion chưa cải thiện chắc chắn; epoch cao hơn không tự sửa thiếu dữ liệu | Chốt grid trước, chọn bằng inner validation, giữ nguyên held-out |

Không cố tình yêu cầu người cao tuổi hoặc người không được huấn luyện thực hiện động tác té ngã để thu dữ liệu. Ưu tiên dữ liệu được phép sử dụng và quy trình thu thập an toàn/được duyệt.

## Một lượt train tiếp nên được chốt ra sao?

1. Giữ **ME-STGCN-TA** một luồng và pipeline được chọn cố định. Giữ cả A1 để biết attention có thật sự cần hay không; không đổi sang CNN–LSTM/Transformer chỉ để chạy lại.
2. Ghi trước tiêu chí: ưu tiên giảm FN nhưng báo cáo cả FP. Dùng sensitivity/specificity và FN/FP theo từng subject; khi đánh giá stream, thêm số báo nhầm/giờ, độ trễ sự kiện, warmup và tỷ lệ không đủ pose. Chưa có căn cứ đặt 95% accuracy làm lời hứa.
3. Trên tập phát triển có thể thử multiplier Fall **1,0 / 1,5 / 2,25** như một grid đề xuất, không phải cấu hình chắc chắn tốt hơn. Chọn multiplier/ngưỡng bằng validation theo subject. Không đặt ngưỡng bằng score của `S4_FALL_05` hoặc một clip outer-test đã xem.
4. Khởi đầu giữ tối đa **50 epoch**, early stopping và ba seed như đối chứng. Kiểm tra train/validation loss trước khi tăng epoch. Nếu train tốt nhưng người mới kém, tăng epoch có thể tăng overfit. Nếu gradient hoặc loss chưa ổn định, mới xét learning rate/regularization có kiểm soát.
5. Chạy nhiều seed để đo ổn định, không lựa seed có metric test cao nhất. Sau khi chọn bằng development/validation, train final trên development đủ người, export ONNX và đối chiếu PyTorch–ONNX trên input mẫu trước khi thay web.
6. Freeze model, preprocessing, threshold và quy tắc cảnh báo rồi mới mở held-out mới. Nếu sửa sau khi xem held-out, phải coi đó là development và cần tập xác nhận khác.

## Pretrain khi nào có ích?

Pretrain là học một biểu diễn chuyển động từ dữ liệu rộng hơn trước khi fine-tune bài toán Fall/ADL. Không phải chỉ đưa file model nhỏ vào dataset lớn là xong. Phải có nhãn/mục tiêu học và loop huấn luyện phù hợp.

Ưu tiên một nguồn dữ liệu được phép sử dụng có RGB và chạy lại **cùng MediaPipe 33 khớp/đặc trưng** nếu muốn giảm khác biệt input. Nếu checkpoint pretrained dùng graph, số khớp hoặc số kênh khác, không thể load toàn bộ weights một cách mù quáng; cần mapping/adaptation và kiểm tra riêng. Không coi fine-tune là bảo đảm khái quát tốt hơn.

Chưa chốt dataset mới trong repo này. Khi chọn, kiểm tra nguồn chính thức, quyền truy cập/tái phân phối, số người, kiểu ngã, ADL dễ nhầm, camera, occlusion và khả năng tách subject độc lập. Chỉ sau khi thống nhất nguồn và cách chia mới lập notebook 08; không có training 08 tự chạy ngầm.
