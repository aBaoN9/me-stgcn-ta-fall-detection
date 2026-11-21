# Hướng nghiên cứu tiếp và kết quả đã nghiệm thu

**Tiến độ 21/11/2026:** GRU/CTR-GCN adapted bổ sung đã train/đánh giá và nghiệm thu qua PR #3 (`GB-20261121-001`), không còn là đề xuất. [Bộ bàn giao](../results/backbone_benchmark/tung-gru-ctr-rtx3050-20261120-233932/README.md) đã có bảng lỗi/ghép cặp bốn model, cùng lần đo lại latency. TCN vẫn cao nhất về Accuracy/Sensitivity trung bình; GRU nhanh nhất theo phép đo model-only; A3 mạnh hơn trên S4. Robustness, equal-FP và A3 gọn hơn vẫn chưa chạy, không là điều kiện chặn nghiệm thu.

**Cập nhật 20/11/2026:** TCN–A3 của Tùng đã nghiệm thu và merge PR #2 theo `GB-20261120-001`; [bộ bàn giao](../results/backbone_benchmark/tung-m2-rtx3050-20261120-165523/README.md) đã có trên main, không còn là đề xuất chưa train. TCN đạt Accuracy 91,04% so với A3 chạy lại 89,38%, nhưng chưa có tập xác nhận độc lập. Các thí nghiệm equal-FP/robustness/A3 gọn bên dưới **chưa được thực hiện**; bảng lỗi/ghép cặp đã được bổ sung ngày 21/11, không đặt thêm điều kiện nghiệm thu Tùng. Bài Đức vẫn xét riêng.

## Sau benchmark TCN–A3: đề xuất, không phải kết quả

- So Sensitivity ở cùng mức báo động giả, chọn ngưỡng bằng inner validation thay vì chọn điểm có lợi trên outer-test. Báo cáo cả hai mô hình và toàn bộ subject/seed, không giấu vế bất lợi.
- Đặt trước mức mất pose/nhiễu và phép đo để kiểm robustness công bằng. Chưa có bằng chứng A3 thắng ở đây; không ghi đặc tính graph/attention thành lợi ích thực nghiệm chưa kiểm.
- Bảng lỗi/ghép cặp bốn model đã có trong PR #3; phát triển A3 gọn hơn với ngân sách tương đương vẫn là đề xuất chưa chạy. Phân tích nảy sinh sau khi xem test phải ghi là thăm dò; sửa thiết kế cần khóa trước đánh giá mới, không tối ưu theo các outer clip đã biết.

Không bắt Tùng phải tìm số liệu để A3 thắng. Nếu TCN tiếp tục tốt hơn dưới điều kiện kiểm tra công bằng, báo cáo kết quả đó. Các quyền tự chủ hiện hành vẫn giữ, không tự cấp tiền/tài nguyên hoặc đổi giao thức chung/deployment.

## Phân công hiện hành ngày 30/10/2026

Yêu cầu học thuật: so phương pháp khác trên chính dữ liệu nhóm và kiểm chứng thay đổi mô hình hiện tại. Xem [protocol](research-protocol.md), [Tùng](tasks/dinhtung.md), [Đức](tasks/vinhduc.md), [bàn giao](handoff.md).

Theo quyết định của Gia Bảo ngày 30/10/2026: Tùng giữ dataset, tự đề xuất mô hình/phương pháp khác để so với ME-STGCN-TA; Đức thay đổi có kiểm soát mô hình hiện tại và so với bản gốc. Cùng giao thức đánh giá, không chỉ định sẵn CTR-GCN, quality-aware hay công thức. Hai bạn giải thích thay đổi và ghi M0 trước full; theo `GB-20261118-001` (Tùng) và `GB-20261116-002` (Đức), cả hai tự khóa M0, tự kiểm test/smoke thật rồi full, không chờ duyệt; tiến độ thực tế xem báo cáo/commit. PoseLab và nhiệm vụ web đã dừng. Khuyến nghị bên dưới chỉ là bối cảnh, không thay phân công mới hoặc chứng minh cải thiện. Không cần chờ dataset ngoài mới làm so sánh phát triển, nhưng không gọi dữ liệu cũ là test độc lập mới.

## Quyết định hiện tại

Giữ nguyên model 06 và notebook 07 như bằng chứng. [PoseLab/WEB-01/WEB-02 đã dừng](tasks/web-app.md); không làm lại frontend, inference hoặc model/threshold ứng dụng. Lượt cập nhật phân công này không chạy thêm train hoặc chọn model/seed thắng outer-test để thay checkpoint. A3_cost_225 fixed là ứng viên nghiên cứu, chưa phải model an toàn đã xác nhận: 96,62% sensitivity vẫn tương ứng 3/3/2 ca bỏ sót theo seed. Các mục mở rộng phía sau không phải việc giao bắt buộc; cả hai tự quyết phương án trong nhiệm vụ theo hai quyết định tự chủ, Gia Bảo review nghiệm thu sau; chỉ yêu cầu ngoài phạm vi mới cần xin quyền.

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

1. Giữ **ME-STGCN-TA** một luồng làm đối chứng, giữ A1 để biết attention có cần không. Theo phân công mới được thêm backbone tham chiếu trong thí nghiệm riêng với input/recipe khóa; không đổi chỉ để có tên model mới, không thay deployment.
2. Ghi trước tiêu chí: ưu tiên giảm FN nhưng báo cáo cả FP. Dùng sensitivity/specificity và FN/FP theo từng subject; khi đánh giá stream, thêm số báo nhầm/giờ, độ trễ sự kiện, warmup và tỷ lệ không đủ pose. Chưa có căn cứ đặt 95% accuracy làm lời hứa.
3. Trên tập phát triển có thể thử multiplier Fall **1,0 / 1,5 / 2,25** như một grid đề xuất, không phải cấu hình chắc chắn tốt hơn. Chọn multiplier/ngưỡng bằng validation theo subject. Không đặt ngưỡng bằng score của `S4_FALL_05` hoặc một clip outer-test đã xem.
4. Khởi đầu giữ tối đa **50 epoch**, early stopping và ba seed như đối chứng. Kiểm tra train/validation loss trước khi tăng epoch. Nếu train tốt nhưng người mới kém, tăng epoch có thể tăng overfit. Nếu gradient hoặc loss chưa ổn định, mới xét learning rate/regularization có kiểm soát.
5. Chạy nhiều seed để đo ổn định, không lựa seed có metric test cao nhất. Sau khi chọn bằng development/validation, train final trên development đủ người, export ONNX và đối chiếu PyTorch–ONNX trên input mẫu trước khi thay web.
6. Freeze model, preprocessing, threshold và quy tắc cảnh báo rồi mới mở held-out mới. Nếu sửa sau khi xem held-out, phải coi đó là development và cần tập xác nhận khác.

## Pretrain khi nào có ích?

Pretrain là học một biểu diễn chuyển động từ dữ liệu rộng hơn trước khi fine-tune bài toán Fall/ADL. Không phải chỉ đưa file model nhỏ vào dataset lớn là xong. Phải có nhãn/mục tiêu học và loop huấn luyện phù hợp.

Ưu tiên một nguồn dữ liệu được phép sử dụng có RGB và chạy lại **cùng MediaPipe 33 khớp/đặc trưng** nếu muốn giảm khác biệt input. Nếu checkpoint pretrained dùng graph, số khớp hoặc số kênh khác, không thể load toàn bộ weights một cách mù quáng; cần mapping/adaptation và kiểm tra riêng. Không coi fine-tune là bảo đảm khái quát tốt hơn.

Kiểm chứng thời gian đã hoàn tất trong [thí nghiệm khoảng vận tốc](timing-ablation.md), nhưng chưa cho cải thiện ổn định và chưa thay mô hình ứng dụng. Nghiên cứu tiếp cần chốt dữ liệu và giao thức mới trước khi huấn luyện. Khi chọn nguồn dữ liệu, kiểm tra quyền truy cập/tái phân phối, số người, kiểu ngã, ADL dễ nhầm, camera, che khuất và khả năng tách chủ thể độc lập.
