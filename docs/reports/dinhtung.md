# Báo cáo công việc - Tùng

Người thực hiện: Tùng. Nhánh: `dinhtung`. [Nhiệm vụ](../tasks/dinhtung.md).

## Snapshot

- Trạng thái: `IN_PROGRESS`; M0: **`M0_PROPOSED — PENDING_APPROVAL`**.
- Cập nhật: 17/11/2026 (UTC+7), **Tùng đã kiểm tra và đồng ý gửi M0 chính thức**; câu `TUNG-20261117-001` OPEN, chờ Gia Bảo phê duyệt.
- Đã đọc `GB-20261030-001`, `GB-20261116-001`, `GB-20261116-002` trên `origin/main` tại `8ab7b29`; ghi nhận hạn **25/11/2026**, sản phẩm model đã train + kết quả GitHub. Quyền tự chủ Đức không áp dụng Tùng.
- Commit code / run ID / PR: chưa có code M1 / `NOT_RUN` / chưa mở. Tài liệu M0 được Tùng cho phép commit/push lên origin/dinhtung; commit phát hành chứa báo cáo này, kết quả gửi phải đối chiếu remote sau push.
- M0 duyệt bởi / ngày / config: **chưa duyệt**; đề xuất TCN nhỏ + chạy lại A3 chuẩn/07, xem [M0](../research/backbone-benchmark.md).
- Mốc: TUNG-01 bản đề xuất đã được Tùng kiểm tra, cho phép gửi, chờ Gia Bảo duyệt; TUNG-02, TUNG-03, TUNG-04 `NOT_RUN`.

Trạng thái: `TODO`, `IN_PROGRESS`, `BLOCKED`, `READY_FOR_REVIEW`, `DONE`. Chỉ người điều phối xác nhận DONE sau review. Cập nhật snapshot mới nhất; nhật ký nối thêm, không xóa lỗi cũ.

## Nhật ký mỗi phiên - mẫu để điền

### YYYY-MM-DD - nội dung phiên

- Đã làm / lý do: chưa có.
- File / commit thay đổi: chưa có.
- Lệnh kiểm tra và PASS/FAIL/NOT_RUN: `NOT_RUN`.
- Input/hash, protocol/hash, code/hash, environment, seed, thiết bị: chưa có.
- Mode / số outer-inner-refit thực tế và kỳ vọng / coverage: `NOT_RUN`.
- Bằng chứng CSV/config/log hoặc artifact có quyền truy cập: chưa có.
- Kết quả tốt / kết quả âm / FN-FP / chi phí: chưa đo.
- Còn thiếu / blocker / việc main cần quyết định: nhận nhiệm vụ, cấp dữ liệu, duyệt M0.
- Bước tiếp: đọc quy tắc, làm TUNG-01.

### 2026-11-17 — M0: nghiên cứu và đề xuất TCN, chưa triển khai

- Phạm vi: chỉ M0/TUNG-01; bảo toàn biểu mẫu và lịch sử cũ. Tùng yêu cầu chưa commit/push nên không thực hiện bước gửi Markdown mặc định của nhóm.
- Git: vùng làm việc đầu phiên sạch trên main. Fetch mặc định báo Permission denied ở `.git/FETCH_HEAD`; chạy lại qua quyền công cụ được phép và thành công. Đã tạo/chuyển local `dinhtung` theo dõi `origin/dinhtung` ở `1ab8164`. `origin/main=8ab7b29`; đọc quyết định mới bằng `git show`, không merge/reset hoặc sửa tài liệu chung. Nhánh local còn quy định cũ 28/10; M0 dẫn bản main cố định mới hơn, trước M1 phải nhận main an toàn trong phiên sau.
- Đã đọc: AGENTS, README, handoff, bảng việc/nhiệm vụ Tùng, báo cáo/trao đổi Tùng, quy trình/answers, protocol, methodology, notebook-guide, data/README; các bản chung mới đọc từ origin/main; ba module component_ablation, notebook07, locked_protocol, phân tích07 và config06. Mã mốc liên quan không khác origin/main.
- Nghiên cứu nguồn gốc/mã chính thức: TCN của Bai/Kolter/Koltun; GRU của Cho và nn.GRU PyTorch2.10; CTR-GCN của Chen. Đã đọc cấu trúc và adaptation (điều chỉnh) cần thiết, giấy phép; không clone/cài/chạy model bên ngoài. Commit TCN `2f8c2b8`, CTR-GCN `e13d7582` đã xác minh bằng `git ls-remote`.
- Phương án: **TCN_v2_small**, bốn khối hidden32/dilation1,2,4,8, cùng tám kênh v2; GRU một lớp hidden64 dự phòng, cần duyệt lại khi kích hoạt. Đề xuất chạy lại A3 chuẩn/07, không tự nhận đó là mốc bắt buộc của Tùng. Cặp so 96 fits (lần huấn luyện), 24 outer (vòng đánh giá ngoài), ba seed; Accuracy ngưỡng0.5 chính. Không giả định TCN tốt hơn.
- Ba tài liệu: mới `docs/research/backbone-benchmark.md`; cập nhật báo cáo này và `docs/discussions/dinhtung.md`, ID `TUNG-20261117-001`, OPEN/bản nháp chưa gửi. Không sửa code hoặc tạo checkpoint/kết quả giả.
- Đã kiểm thực tế: Git status/remote/nhánh; đọc/tổng hợp hàng fixed A3/A3_cost trong CSV lịch sử, không suy thành kết quả mới; `data/` chỉ có README, không có bốn input. `nvidia-smi` ghi RTX3050 Laptop, 4096MiB, driver566.07. Có `kaggle.exe` trong PATH (đường dẫn lệnh); chưa kiểm tài khoản/quota (hạn mức)/GPU Kaggle, PyTorch CUDA hoặc quyền push/upload. Không đọc secrets (bí mật).
- Hash chuẩn: đã chép đủ bốn SHA-256 từ locked_protocol07 vào M0; **hash input thực tế NOT_RUN** vì chưa có file. Code/config hash run mới chưa có vì chưa triển khai. Môi trường train chưa thiết lập/xác minh; không thay dependency (thư viện phụ thuộc).
- Kiểm bàn giao tài liệu đã chạy: `git diff --check` PASS; `python -B scripts/check_repository.py` PASS (hash model cũ, chín notebook sạch, một Word, kết quả/liên kết cũ); kiểm UTF-8/liên kết nội bộ ba tài liệu PASS; đối chiếu bốn hash ghi trong M0 với locked_protocol07 PASS; số học kế hoạch 96fits/24outer/2.880 dự đoán/72 hàng fold-policy PASS. Kiểm phạm vi đúng ba file và staging (vùng chờ commit) rỗng PASS. Đây là kiểm tài liệu/toàn vẹn, không phải kiểm model hoặc hash dữ liệu thật.
- Smoke giả/thật, full, OOF mới (dự đoán ngoài mẫu), reload model, benchmark (đo hiệu suất): **NOT_RUN**. Số fits thực tế 0; kế hoạch M2 96 fits không phải tiến độ đã chạy.
- Còn thiếu: Gia Bảo duyệt phương pháp/mốc/ngân sách; nguồn có quyền của bốn input đúng hash; xác nhận tài nguyên GPU/môi trường và quyền artifact GitHub. Có GPU vật lý không chứng minh train chạy được.
- Bước tiếp: **dừng ở M0**, Tùng kiểm ba tài liệu; chỉ khi Tùng cho phép mới commit/push đề xuất. Chờ Gia Bảo duyệt M0 trước full; không tự chuyển M1 trong phiên này.

### 2026-11-17 — Phát hành M0 theo yêu cầu Tùng

- Tùng đồng ý bản M0 và cho phép đồng bộ/commit/push ba file; đây là quyền gửi, không phải phê duyệt nghiên cứu của Gia Bảo. Các câu "chưa gửi/không commit" ở nhật ký soạn phía trên là lịch sử trước yêu cầu phát hành này.
- Fetch origin thành công; origin/main vẫn 8ab7b29, đã đọc lại answers.md, chưa có trả lời TUNG-20261117-001. Không thay thiết kế: TCN chính, A3/07 là mốc đề xuất, GRU dự phòng cần xác nhận; M0_PROPOSED — PENDING_APPROVAL; train/smoke/M1/M2 NOT_RUN.
- Đồng bộ bằng git merge --ff-only origin/main từ 1ab8164 đến 8ab7b29, không xung đột. So hash SHA-256 ba tài liệu trước/sau khớp hoàn toàn; không stash/reset/force-push hoặc ghi đè công việc. Các cập nhật chung là commit main đã có, không phải thay đổi tự soạn của Tùng.
- Cập nhật trạng thái hiện tại ba tài liệu từ bản nháp sang đề xuất chính thức được phép gửi; bảo toàn lịch sử và ID. Chỉ stage ba đường dẫn M0, kiểm outgoing (commit sắp gửi) không có code ngoài phạm vi; không push main.
- Kiểm tra phát hành: git diff --check PASS; python -B scripts/check_repository.py PASS; kiểm UTF-8/liên kết ba file, bốn hash trong M0, trạng thái đề xuất/ID và phạm vi đúng ba file PASS; origin/main là tổ tiên của HEAD sau đồng bộ. Không coi kiểm tài liệu là test model. Commit/push và SHA remote kiểm riêng trước khi báo gửi thành công; không tự ghi Gia Bảo đã đọc/duyệt.
- Còn cần Gia Bảo xác nhận: TCN/A3 và ngân sách96fits, nguồn bốn input đúng hash, tài nguyên GPU, điều kiện GRU dự phòng, kênh/quyền upload artifact model. Push tài liệu nếu thành công không chứng minh quyền upload Release.
- Điểm dừng: sau xác minh push, chờ quyết định M0; không triển khai TCN hoặc chạy smoke/full.

## Bàn giao review

- Checklist nhiệm vụ: chưa hoàn tất.
- Lệnh tái lập / thời gian / tài nguyên đã dùng: chưa có.
- Khác paper/recipe (quy trình cấu hình), giới hạn kết luận: có thiết kế/adaptation trong M0; chưa có bằng chứng thực nghiệm mới. TCN thay cả backbone (phần mạng học biểu diễn) và tổng hợp, không cô lập tác dụng riêng một thành phần.
- PR về main: chưa mở; sau push main có thể đọc `git show origin/dinhtung:docs/reports/dinhtung.md` hoặc PR.
