# Tùng - Mô hình hoặc phương pháp khác trên cùng dataset

## Nhiệm vụ mới 24/11/2026: năm đối thủ đầu tiên

Theo **GB-20261124-001**, Tùng triển khai và **train CascadeFormer, DHA-eGCN, Hyper-GCN, ProtoGCN, InfoGCN++**, cùng A3 tham chiếu. Đọc [thiết lập, nguồn pin, hợp đồng input/recipe/metric và prompt AI](../research/modern-model-benchmark.md). Đây là nhiệm vụ cụ thể mới; phần "không chỉ định sẵn mô hình" bên dưới là phạm vi cũ, không thay danh sách này.

- Nhánh `dinhtung`; vùng mới `training/modern_benchmark_tung/`, `results/modern_benchmark_tung/<run-id>/`, `docs/research/modern-benchmark-tung.md`, báo cáo/trao đổi cá nhân. Không sửa helper/runner/kết quả cũ để triển khai.
- Giữ tensor v2 `(N,8,32,33)`, recipe CE chung, nested LOSO/seed/threshold/F1-Fall giống A3. Tự khóa M0, test/smoke thật rồi full, không đợi duyệt. Ghi adaptation/loss khác paper.
- Trạng thái giao việc: `ASSIGNED / NOT_RUN`; không đổi nghiệm thu TCN–A3/GRU/CTR-GCN/SkateFormer cũ. Nộp model thật, raw OOF và audit qua PR, không chỉ nộp danh sách.
- Xác nhận **GB-20261124-001** trong file trao đổi Tùng và push. Hạn bài cũ 25/11 giữ nguyên; báo tiến độ/ETA/blocker của đợt mới, chưa có hạn riêng đủ cả mười model.

## Bối cảnh nhiệm vụ trước

**Nghiệm thu bổ sung 23/11/2026 — `ACCEPTED_BENCHMARK`:** PR #4 SkateFormer adapted small đã merge tại `3c6a39e`, nguồn `20c67f5`, theo `GB-20261123-001`. [Bộ kết quả](../../results/backbone_benchmark/tung-skate-rtx3050-20261122-205954/README.md) có 48 fits/12 checkpoint full; nhận kết quả âm hợp lệ (Accuracy 84,17%), không bắt train lại hoặc thay deployment. Gia Bảo tích hợp vào [FallScope local](../../demo_streamlit/README.md); đây không phải giao lại việc UI/QA cho Tùng. Giữ nguyên những nghiệm thu trước và phạm vi Đức.

**Nghiệm thu bổ sung 21/11/2026 — `ACCEPTED_BENCHMARK`:** GRU/CTR-GCN adapted trong PR #3 được nhận theo `GB-20261121-001`. [Bộ bổ sung](../../results/backbone_benchmark/tung-gru-ctr-rtx3050-20261120-233932/README.md) có 96 fits và 24 checkpoint full mới; [README](../../README.md) đã có so sánh bốn model, ưu/nhược và giới hạn. Gia Bảo sửa tiện ích monitor và thêm test mà không đổi code train đã khóa hoặc artifact; không cần train lại, không thêm điều kiện nghiệm thu. Giữ nguyên mốc TCN–A3, deployment và phạm vi bài Đức.

**Nghiệm thu 20/11/2026 — `ACCEPTED_BENCHMARK`:** M0–M3 TCN–A3 đã được nhận theo `GB-20261120-001`, PR #2 merge vào main (`bbe3c64`). Đủ 96 fits, 24 checkpoint và kết quả thật; [bộ bàn giao](../../results/backbone_benchmark/tung-m2-rtx3050-20261120-165523/README.md) đã có trên main. Không cần làm thêm để A3 thắng mới chấp nhận bài. Các yêu cầu/tiêu chí bên dưới giữ làm phạm vi và lịch sử; nghiên cứu bổ sung không tự là nhiệm vụ bắt buộc mới.

**Cập nhật hỗ trợ 19/11/2026:** đọc [bắt đầu train](../training-start.md); đủ bốn input chuẩn, có tool kiểm/tải, cách Kaggle/local và checklist nộp. Theo `GB-20261119-002`, được tự full CPU nếu không có GPU; khóa addendum/config/code trước outer-test, không chờ duyệt lại.

Cập nhật ngày 18/11/2026 theo `GB-20261118-001`: **AUTONOMOUS — tự triển khai và train, không chờ Gia Bảo duyệt M0 hoặc từng bước**. Nhánh `dinhtung`. [Báo cáo](../reports/dinhtung.md). Đọc [AGENTS](../../AGENTS.md), [bàn giao](../handoff.md), [protocol](../research-protocol.md) và quyết định mới trong [answers](../discussions/answers.md) trước khi sửa.

M0 TCN so với A3 tại `d65e59c` đã được cho phép triển khai theo `TUNG-20261117-001`; bộ thực nghiệm sau đó đã nghiệm thu ngày 20/11 theo `GB-20261120-001`. Quyền tự chủ tiếp tục trong phạm vi đã giao, không cho phép đổi kết quả/lock đã công bố. Tùng tự khóa cấu hình/ngân sách trước outer-test, tự điều chỉnh hoặc chuyển GRU/phương án khác trong nhiệm vụ trước outer-test có ghi lý do, không phải xin duyệt lại. Tự chọn Kaggle/GPU local có quyền và không phát sinh chi phí. Được commit/push sản phẩm, upload artifact và mở PR đúng phạm vi, không xin lại từng lần. Hạn **25/11/2026 (UTC+7)** vẫn yêu cầu model đã train + kết quả thật trên GitHub.

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

1. **TUNG-01 / M0 - Tự chốt phương án:** nêu mô hình/phương pháp định dùng, phần giữ nguyên/thay đổi, lý do, nguồn và quyền sử dụng, đầu vào cần thiết, mốc ME-STGCN-TA đem so, ngân sách và cách bảo đảm đánh giá như nhau. Ghi và khóa trước outer-test, cập nhật `M0_DOCUMENTED`; đây là đăng ký phương án, không phải đơn xin duyệt.
2. **TUNG-02 / M1 - Có thể tái lập:** triển khai trong vùng riêng; kiểm dữ liệu, shape, chia subject, không rò rỉ, metric và smoke theo phương án. Không coi synthetic/smoke là kết quả nghiên cứu.
3. **TUNG-03 / M2 - So sánh:** tự chạy phương án đã khóa và mốc đối chiếu phù hợp theo protocol chung khi dữ liệu/test/smoke thật PASS, không chờ duyệt full hoặc review lại bản sửa. Lưu dự đoán ngoài mẫu, cấu hình/seed/hash/log và bảng chỉ số, FN/FP; báo khác biệt đầu vào, ngân sách hoặc môi trường còn tồn tại.
4. **TUNG-04 / M3 - Bàn giao:** giải thích thay đổi, kết quả hơn/kém và giới hạn. Nộp mã, lệnh tái lập, bằng chứng và báo cáo; không ép phương pháp khác phải thắng hay thua đề xuất.

## Tiêu chí hoàn tất

- [ ] Phương án M0 đã ghi/khóa trước outer-test; phạm vi và mốc so sánh rõ, không cần phê duyệt thủ công.
- [ ] Validation/smoke có bằng chứng; không thay dữ liệu hoặc xem test để chọn cấu hình.
- [ ] Full theo protocol đã khóa, đủ OOF và chỉ số tái tính từ dự đoán; không cherry-pick seed/subject.
- [ ] Giải thích phần khác nhau và đánh đổi, kể cả kết quả âm; chi phí nếu đo phải cùng điều kiện.
- [ ] Checker repo PASS; báo cáo và PR nêu rõ phần đã làm/chưa chạy.

Đầu phiên fetch và đọc `origin/main:docs/discussions/answers.md`; nhận hướng dẫn mới bằng merge an toàn khi đã publish, không reset bài làm. Thiếu input/GPU thì báo lý do và làm tiếp phần độc lập. Giữ mã/kết quả đã làm, không tự xóa để đổi hướng. Không tự sửa hoặc push nhánh Đức/main.
