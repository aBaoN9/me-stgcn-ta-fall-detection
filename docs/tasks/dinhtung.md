# Tùng - Mô hình hoặc phương pháp khác trên cùng dataset

Cập nhật theo phân công của Gia Bảo ngày 30/10/2026. Nhánh `dinhtung`. [Báo cáo](../reports/dinhtung.md). Đọc [AGENTS](../../AGENTS.md), [bàn giao](../handoff.md), [protocol](../research-protocol.md) và quyết định mới trong [answers](../discussions/answers.md) trước khi sửa.

## Câu hỏi được giao

**Giữ bộ dữ liệu GMDCSA-24, dùng mô hình hoặc phương pháp khác để xử lý bài toán Fall/ADL, rồi so sánh với ME-STGCN-TA theo cùng giao thức đánh giá.**

Tùng tự tìm hiểu và đề xuất phương án phù hợp, giải thích khác gì so với đề xuất hiện tại và vì sao đáng so sánh. Không chỉ định sẵn mô hình, công thức hoặc tham số. CTR-GCN và ma trận B0-B3 trong giao việc cũ không còn bắt buộc; không cần chạy cho bằng được một mô hình đã chọn hộ hoặc đạt số cao hơn.

Nếu các phương pháp khác cả biểu diễn đầu vào và cách học, kết luận là so sánh phương pháp/hệ thống hoàn chỉnh. Chỉ kết luận hiệu quả riêng của kiến trúc khi cặp so sánh giữ được các yếu tố còn lại. Không lấy số công bố trên dataset khác để thay kết quả trên dữ liệu nhóm.

## Phạm vi được ghi

- `training/backbone_benchmark/**`: mã thực nghiệm, kiểm tra, hướng dẫn và notebook riêng nếu cần; giữ tên thư mục để không đụng bài làm đã có, không giới hạn nhiệm vụ vào backbone.
- `results/backbone_benchmark/**`: mỗi run một thư mục, lưu bằng chứng và cấu hình riêng.
- `docs/research/backbone-benchmark.md`: phương án, lý do, nguồn tham khảo, bảng so sánh và kết luận.
- `docs/reports/dinhtung.md`, `docs/discussions/dinhtung.md`: tiến độ và trao đổi cá nhân theo [quy trình](../discussions/README.md).

Được đọc/import nguồn chung khi phù hợp, không sửa helper/kết quả cũ, phần Đức, app, model, checker hoặc protocol chung. PoseLab và WEB-02 đã hủy; không nhận QA web như việc phụ.

## Mốc thực hiện

1. **TUNG-01 / M0 - Đề xuất:** nêu mô hình/phương pháp định dùng, phần giữ nguyên/thay đổi, lý do, nguồn và quyền sử dụng, đầu vào cần thiết, mốc ME-STGCN-TA đem so, ngân sách và cách bảo đảm đánh giá như nhau. Gửi qua Markdown để Gia Bảo duyệt trước full train.
2. **TUNG-02 / M1 - Có thể tái lập:** triển khai trong vùng riêng; kiểm dữ liệu, shape, chia subject, không rò rỉ, metric và smoke theo phương án. Không coi synthetic/smoke là kết quả nghiên cứu.
3. **TUNG-03 / M2 - So sánh:** chạy phương án đã duyệt và mốc đối chiếu phù hợp theo protocol chung. Lưu dự đoán ngoài mẫu, cấu hình/seed/hash/log và bảng chỉ số, FN/FP; báo khác biệt đầu vào, ngân sách hoặc môi trường còn tồn tại.
4. **TUNG-04 / M3 - Bàn giao:** giải thích thay đổi, kết quả hơn/kém và giới hạn. Nộp mã, lệnh tái lập, bằng chứng và báo cáo; không ép phương pháp khác phải thắng hay thua đề xuất.

## Tiêu chí hoàn tất

- [ ] Phương án M0 được duyệt; phạm vi và mốc so sánh rõ.
- [ ] Validation/smoke có bằng chứng; không thay dữ liệu hoặc xem test để chọn cấu hình.
- [ ] Full theo protocol đã khóa, đủ OOF và chỉ số tái tính từ dự đoán; không cherry-pick seed/subject.
- [ ] Giải thích phần khác nhau và đánh đổi, kể cả kết quả âm; chi phí nếu đo phải cùng điều kiện.
- [ ] Checker repo PASS; báo cáo và PR nêu rõ phần đã làm/chưa chạy.

Đầu phiên fetch và đọc `origin/main:docs/discussions/answers.md`; nhận hướng dẫn mới bằng merge an toàn khi đã publish, không reset bài làm. Thiếu input/GPU thì báo lý do và làm tiếp phần độc lập. Giữ mã/kết quả đã làm, không tự xóa để đổi hướng. Không tự sửa hoặc push nhánh Đức/main.
