# Trả lời và quyết định của người điều phối

File do người điều phối cập nhật trên `main`. Thành viên fetch và đọc `origin/main:docs/discussions/answers.md`, không sửa file này trên nhánh cá nhân để tự cấp quyền cho mình.

## Snapshot

- Cập nhật: 30/10/2026, Gia Bảo xác nhận phân công hai hướng và dừng PoseLab, mã `GB-20261030-001`.
- Tùng tự đề xuất mô hình/phương pháp khác trên cùng dataset; Đức thay đổi có kiểm soát ME-STGCN-TA hiện tại. Cùng giao thức đánh giá, không áp sẵn CTR-GCN hoặc quality-aware.
- `DUC-20261028-001`: giữ `CHANGES_REQUESTED` về lỗi kiểm dữ liệu/nghiệm thu nếu dùng tiếp mã đã review; yêu cầu cố định hướng/công thức Q0/Q1 đã được thay bởi quyết định mới.
- Chưa cấp phê duyệt full train, nguồn dữ liệu mới, dịch vụ GPU trả phí hoặc thay model/backend triển khai.
- Quy trình chung: [kênh trao đổi](README.md).

## GB-20261030-001 - Phân công hai hướng và dừng PoseLab

- Người quyết định: Gia Bảo, chủ repo/người điều phối; xác nhận trực tiếp trong chat ngày 30/10/2026, múi giờ UTC+7.
- Quyết định: `APPROVED` cho phạm vi phân công mới và dừng app; **không phải phê duyệt cấu hình hoặc full train**.
- **Tùng / dinhtung:** giữ dataset, dùng mô hình hay phương pháp khác để xử lý và so sánh với ME-STGCN-TA. Tùng tự lựa chọn và giải thích thay đổi, không bắt buộc CTR-GCN hoặc ma trận B0-B3.
- **Đức / vinhduc:** thay đổi có kiểm soát mô hình hiện tại, giữ ME-STGCN-TA làm mốc; so bản gốc với thay đổi được giải thích. Không giao sẵn quality-aware/công thức Q0-Q1 và không đổi hẳn mô hình như nhiệm vụ Tùng. Giữ kiến trúc trong hướng chính, sửa kiến trúc cần xin mở rộng phạm vi.
- **Chốt nhận diện mốc cho Đức:** `A3_motion_attention` / `fixed` của notebook 07 và mã `training/component_ablation/` tại `c8f654bf0f1943ffcfe375911c8b3e0015beadc5`; `channels=8`, `attention=True`, `fall_multiplier=1.0`, feature v2. Nguồn/cấu hình/kết quả lịch sử và cách chạy lại bản gốc cùng bản thay đổi ghi trong [nhiệm vụ Đức](../tasks/vinhduc.md). Không dùng checkpoint final 06, A3_cost_225, T0/T1 hoặc Q1 thay mốc. Đây là chốt đối chứng, không áp sẵn cách cải tiến hoặc phê duyệt full train.
- Cùng dữ liệu/nhãn/split theo subject, quy tắc chọn cấu hình/ngưỡng, seed, ngân sách tương đương và cách tính kết quả theo [protocol](../research-protocol.md). Nộp M0 ngắn về phần giữ nguyên/thay đổi, lý do, cặp so và ngân sách trước full; không chọn bằng outer-test.
- Hủy `WEB-01` và `WEB-02`, không còn việc PoseLab khi rảnh hoặc chờ GPU. Chặn launcher và khởi động ASGI; giữ mã/model/bằng chứng để truy vết, không xóa lịch sử cục bộ hoặc kết quả.
- Thay phạm vi giao ngày 28/10/2026 và phần ép hướng quality-aware/công thức trong `DUC-20261028-001` ngày 29/10. Giữ phát hiện lỗi phần mềm/kiểm dữ liệu và nghĩa vụ sửa nếu mã còn dùng; không tự nâng trạng thái bài làm đã review.
- Không yêu cầu bỏ bài làm cũ: thành viên chỉ rõ phần tái sử dụng, phần cần sửa và phương án tiếp theo. Không bắt hai bạn tạo số tốt hơn; báo cả kết quả âm và đánh đổi.
- Gia Bảo đã yêu cầu bổ sung mốc rõ ràng cho Đức rồi publish lên main trong chat ngày 30/10/2026. Tích hợp sau kiểm tra và xác nhận remote/PR trước khi nói đã gửi; hai bạn fetch/đọc main và nhận cập nhật an toàn, không reset bài làm hoặc ghi đè nhánh cá nhân. Quyết định này không cho phép AI đẩy thay nhánh Tùng/Đức.

## DUC-20261028-001 - Review cũ, phạm vi đã cập nhật

**Đọc GB-20261030-001 trước.** Nội dung sau giữ làm lịch sử review; phần chỉ định quality-aware/Q0-Q1 không còn bắt buộc. Các lỗi kỹ thuật và yêu cầu trung thực bằng chứng vẫn áp dụng nếu tiếp tục dùng mã đó.

- ID câu hỏi / người hỏi: `DUC-20261028-001` / Đức, nhánh `vinhduc`.
- Thời gian / vai trò: 2026-10-29 19:18 UTC+7 / AI điều phối theo yêu cầu chủ repo.
- Snapshot đã review: `origin/vinhduc` tại `1cbb778` (28/10/2026); không suy ra commit sau đã được duyệt.
- Quyết định: `CHANGES_REQUESTED`, không bác bỏ hướng nghiên cứu, không yêu cầu làm lại từ đầu. Chưa có quyết định cũ cho ID này.

### Đánh giá: giữ phần đúng, chưa nghiệm thu nghiên cứu

- Đúng phân công, không sửa model triển khai/nguồn cũ, không lấy web thay nghiên cứu. DUC-01 có script và số liệu tái lập được, không yêu cầu train lại 07/08 hay viết lại bảng đúng. Failure mode chưa xác minh bằng video vẫn phải ghi giới hạn.
- Người điều phối chạy lại audit PASS 22/22, checker repo PASS; bộ kiểm pooling PASS 13/13, smoke tổng hợp đủ 8 outer / 32 fits. Môi trường review: Python 3.11, torch 2.11.0+cpu, numpy 1.26.4, pandas 2.2.3, scikit-learn 1.8.0; khác môi trường Đức đã báo. Chưa chạy smoke thật/full.
- Giữ đề xuất `w = softmax(a + log(0.1 + q'))`, q từ visibility đã mask/augment, epsilon 0.1 cố định, avg-pool kernel 3 / stride 2 / padding 1 bỏ padding; Q0 multiplier 1.0, recipe 07. Không yêu cầu đổi kiến trúc hoặc search epsilon. Đây chưa là APPROVED cho full.
- Related work vẫn là bản thảo; review này không xác nhận đã thẩm định toàn văn từng nguồn hoặc tính mới. Smoke giả không chứng minh cải thiện khoa học.

### DUC-REV-01 - Khóa đúng dữ liệu trước train (bắt buộc)

- Sửa guard trong `training/contribution_study/runner.py`. Signature hiện chỉ ghi hash input và kiểm lần chạy trước trong cùng output, chưa đối chiếu bộ được duyệt.
- Kiểm đủ bốn SHA-256 với `results/component_ablation/locked_protocol.json` trước fit; sai hash/thiếu input phải dừng. Không tự nhận hash mới là bộ chuẩn. Smoke thật cũng qua guard.
- Test: bộ chuẩn qua; đổi một hash hoặc thiếu một input bị chặn trước train. Test tổng hợp dùng đường riêng có nhãn synthetic, không bypass guard dữ liệu thật và không được xuất scientific_result=true.

### DUC-REV-02 - Không báo hoàn tất khi thiếu artifact (bắt buộc)

- Sửa `run_neural`/`summarize_run` trong runner. Đã tái hiện: 8 báo cáo complete=false, fits=0, chỉ có fixed, metrics rỗng vẫn xuất complete=true, fits_expected=32, fits=0.
- Kiểm đúng tập variant/seed/outer, đủ fixed/balanced/sensitivity, complete từng fold, số inner/refit/fits kỳ vọng. Không dùng số file hoặc coverage nhóm đang có làm bằng chứng duy nhất.
- Kiểm ID/index/subject/label với dữ liệu nguồn, score/threshold hữu hạn, prediction khớp threshold; tái tính và đối chiếu fold/pooled metric. Giữ kiểm uniqueness/coverage hiện có.
- Resume chỉ bỏ qua fold có metadata/checkpoint/attention/history hợp lệ. Artifact thiếu/hỏng phải báo rõ hoặc chạy lại an toàn, không ghi complete=true, không trộn signature.
- Test âm: complete=false, sai fits, thiếu policy/artifact, sai nhãn/căn chỉnh, metric sai đều bị chặn. Test dương: run hợp lệ vẫn PASS. Fixture không được coi là bằng chứng train thật.

### DUC-REV-03 - Đồng bộ đúng GPU khi đo latency

- Sửa `benchmark_latency.py`: chuẩn hóa `torch.device`, kiểm device.type và đồng bộ đúng thiết bị CUDA. Hiện so sánh chuỗi đúng bằng cuda khiến cuda:0 không synchronize.
- Test CPU không gọi CUDA sync; cuda và cuda:0 đều đồng bộ đúng. Có thể mock trên máy CPU; chưa có GPU thì ghi benchmark GPU NOT_RUN, không giả kết quả.
- Số CPU đã nộp không chịu lỗi này; không bắt đo lại CPU chỉ vì sửa nhánh CUDA. Giữ warmup/eval/no-grad/batch/median/p95.

### DUC-REV-04 - Audit không được bỏ qua metric NaN

- Sửa `audit_existing.py`, hàm compare và đối chiếu seed summary: kiểm đủ khóa, uniqueness, giá trị hữu hạn trước tính sai lệch. Trường hợp metric không xác định hợp lệ cần quy ước rõ, không âm thầm bỏ qua.
- Kiểm bổ sung đã tái hiện: accuracy tái tính 0.9/0.8 so với bảng lưu NaN/0.8 vẫn PASS, max_abs_diff=0 do np.nanmax bỏ ô NaN. Không có bằng chứng bảng hiện tại chứa lỗi này; đây là yêu cầu làm checker đáng tin cậy.
- Test ô thiếu/NaN và khóa trùng phải FAIL; audit 07/08 nguyên trạng vẫn PASS, không thay nguồn kết quả cũ.

### DUC-REV-05 - Cập nhật báo cáo và cách tái lập

- Snapshot còn ghi code chưa commit dù đã có `1cbb778`: cập nhật commit, từng DUC-REV đã xử lý, lệnh/check/PASS/FAIL/NOT_RUN, hash và artifact kiểm chứng. Giữ lịch sử, không tự ghi DONE.
- Làm rõ lệnh kiểm/audit Q0-Q1 sau full. Audit hiện hard-code cặp A/T, chưa có lệnh dùng trực tiếp Q0/Q1; bổ sung entrypoint hoặc sửa mô tả đang hứa tái sử dụng để nêu đúng phần còn thiếu.
- Giữ parity Q0, shape/mask/augmentation, all-missing, gradient/leakage; chạy lại toàn bộ cùng checker sau sửa. Không yêu cầu Q1 phải thắng để nghiệm thu.

### Phạm vi, blocker và thứ tự tiếp tục

1. Sửa trên vinhduc trong [phạm vi Đức](../tasks/vinhduc.md), không sửa helper 07/checker/protocol chung/model triển khai hoặc answers.md. Các việc sửa/test và chuẩn bị notebook Kaggle làm được ngay, không chờ Tùng; web vẫn phụ.
2. Nguồn bốn input đúng hash còn chờ chủ repo/nhóm cung cấp nơi có quyền truy cập. Đây là phụ thuộc phía điều phối, không phải lý do quy lỗi Đức; không tạo tensor giả thay dữ liệu hoặc tự tải nguồn không rõ quyền.
3. Có input thì chạy validation, guard hash và smoke thật; báo phân bố q, mức clip/frame thiếu pose, min/mean/phân bố. Không đổi công thức theo outer score. Máy không GPU không phải lỗi thực hiện; không tự mua dịch vụ hoặc public dữ liệu.
4. Nộp lại code/báo cáo/log/check trên vinhduc, cập nhật cùng câu hỏi `DUC-20261028-001` dẫn commit và DUC-REV, mở PR theo quy tắc commit/push hiện hành. Chưa có input vẫn nộp review phần mềm, ghi smoke thật/full NOT_RUN.
5. Chỉ sau review bản sửa và smoke thật, main mới trả lời APPROVED riêng cho full Q0/Q1 24 outer / 96 fits. Hiện CHƯA được chạy full, thêm cost/loss/backbone hoặc thay deployment.

Điều kiện nhận lại: test hồi quy dương/âm cho DUC-REV-01/02/03/04, các lỗi không còn tái hiện, checker và bộ kiểm hiện có PASS, báo cáo cập nhật đúng. Không bắt làm lại phần đã đạt.

Đức và AI của Đức đọc phản hồi từ main, không cần merge để nhận chỉ dẫn:

```bash
git fetch origin
git show origin/main:docs/discussions/answers.md
```

Sau khi đọc, làm tiếp trên vinhduc, cập nhật báo cáo và cùng câu hỏi theo ID này. Không tự đánh dấu APPROVED hoặc push lên main.

## Mẫu trả lời - chưa phải quyết định

- ID câu hỏi / người hỏi: điền đúng ID từ file trao đổi cá nhân.
- Thời gian trả lời UTC+7 / người quyết định: điền rõ.
- Quyết định: `APPROVED`, `CHANGES_REQUESTED`, `REJECTED` hoặc `NEED_INFO`.
- Phạm vi được phép / điều kiện / lý do: điền cụ thể.
- Việc cần làm tiếp / bằng chứng cần nộp: điền rõ.
- Thay thế quyết định cũ (nếu có): dẫn ID/commit, không để hai quyết định mâu thuẫn không giải thích.

Giữ các câu trả lời còn liên quan trong snapshot để thành viên đối chiếu; lịch sử đầy đủ ở Git. Không hứa đã kiểm câu mới nếu chưa fetch/đọc nhánh của thành viên.
