# Trả lời và quyết định của người điều phối

File do người điều phối cập nhật trên `main`. Thành viên fetch và đọc `origin/main:docs/discussions/answers.md`, không sửa file này trên nhánh cá nhân để tự cấp quyền cho mình.

## Snapshot

- Cập nhật: 29/10/2026, review nhánh Đức tại `1cbb778`.
- `DUC-20261028-001`: `CHANGES_REQUESTED`. Giữ hướng quality-aware pooling; sửa kiểm dữ liệu/nghiệm thu và nộp lại trước full.
- Chưa cấp phê duyệt full train, nguồn dữ liệu mới, dịch vụ GPU trả phí hoặc thay model/backend triển khai.
- Quy trình chung: [kênh trao đổi](README.md).

## DUC-20261028-001 - Review và yêu cầu sửa cho Đức

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
