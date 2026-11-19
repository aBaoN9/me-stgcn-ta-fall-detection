# Báo cáo công việc - Gia Bảo

Người thực hiện: Gia Bảo. Nhánh phát triển: `giabao`. Vai trò: chủ repo/người điều phối. [Nhiệm vụ](../tasks/giabao.md).

## Snapshot

- Cập nhật: 19/11/2026, theo yêu cầu Gia Bảo đưa bốn input chuẩn vào `data/gmdcsa24/` trên main (`GB-20261119-001`), hash đúng protocol 07 và validation PASS. Cả hai giữ quyền tự chủ; không chạy train mới trong phiên cấp dữ liệu.
- Tùng dùng mô hình/phương pháp khác trên cùng dataset; Đức thay đổi có kiểm soát ME-STGCN-TA. Không áp sẵn mô hình/công thức, cùng giao thức đánh giá. Cả hai tự ghi/khóa M0 và kiểm test/smoke thật rồi full, không chờ duyệt từng bước.
- Quyết định chính thức và các phản hồi cho Tùng/Đức vẫn ở `docs/discussions/answers.md` trên main; báo cáo này không thay kênh phê duyệt.
- Trạng thái tạo/publish nhánh phải đối chiếu Git/remote, không dùng nội dung báo cáo này làm bằng chứng push thành công.

## Nhật ký tiếp theo

Mỗi phiên nối thêm ngày, yêu cầu, phạm vi, file/commit/PR, lệnh kiểm tra và PASS/FAIL/NOT_RUN, giới hạn/blocker và bước tiếp. Khi review nhánh khác ghi đúng commit; khi tích hợp ghi phần được duyệt và bằng chứng, không ghi code chưa review thành DONE.

## 19/11/2026 - Cấp đủ bốn input để thành viên triển khai/train

- Gia Bảo chuyển tin Tùng thiếu bốn input và yêu cầu kiểm main/đưa lên. Main trước cập nhật chỉ có hướng dẫn, không có input hoặc Release. Không quy thiếu dữ liệu thành lỗi train của Tùng.
- Tìm đủ export notebook 01 trong workspace, đối chiếu cả bốn SHA-256 với locked protocol 07: khớp hoàn toàn. Chép nguyên byte vào `data/gmdcsa24/` dưới tên NPZ/CSV chuẩn; không đổi nội dung/nhãn/split hoặc tạo bộ dữ liệu thay thế.
- Tổng 15.100.639 byte; thêm manifest nguồn/hash/dung lượng, ngoại lệ ignore giới hạn đúng năm tệp và cập nhật hướng dẫn/NOTICE/AGENTS/bàn giao/answers. Kiểm metadata nguồn GMDCSA-24 v2.1 trên Zenodo: CC BY 4.0; giữ ghi công và thông báo dẫn xuất.
- `python -m training.component_ablation.validate_pipeline --input-root data/gmdcsa24`: PASS, 160 mẫu, parity sai lệch tối đa 1.1920928955078125e-07, 100 ca augmentation và shape năm biến thể gốc. Không phải smoke/train TCN; không xác nhận môi trường/GPU Tùng/Đức.
- Kiểm tra repo, hash staged và bản tải lại từ GitHub phải đạt trước bàn giao; xác nhận publish bằng remote sau push. Không sửa nhánh/báo cáo cá nhân hai bạn hoặc nói đã nhận dữ liệu khi chưa có phản hồi.

## 18/11/2026 - Tùng tự chủ triển khai và train

- Đã fetch/review `origin/dinhtung` tại `d65e59c` (17/11/2026): chỉ ba tài liệu M0/trao đổi/báo cáo, chưa code M1/smoke/train/kết quả mới. Không lấy bằng chứng 06/07/08 thành thành quả mới của Tùng.
- Theo yêu cầu trực tiếp của Gia Bảo, ban hành `GB-20261118-001`, trả lời `TUNG-20261117-001`: cho phép bắt đầu TCN so với A3, tự chọn/điều chỉnh cấu hình, kiến trúc, dự phòng và thiết bị trong nhiệm vụ; không chờ duyệt M0/từng bước/full. Giữ quyền tự chủ của Đức.
- Đồng bộ AGENTS, README, bàn giao, protocol, bảng việc/nhiệm vụ Tùng, quy trình trao đổi và roadmap để bỏ chỉ dẫn chờ cũ. Không sửa/push nhánh hoặc báo cáo Tùng, không tự nhận Tùng đã đọc hay đã train.
- Kiểm `git diff --check` và `python scripts/check_repository.py`: PASS; model/hash, chín notebook, kết quả cũ và liên kết tài liệu giữ nguyên. Không chạy train hoặc bật PoseLab trong phiên điều phối này.
- Hạn 25/11/2026 và sản phẩm model + kết quả thật trên GitHub giữ nguyên. Input/GPU/quyền truy cập còn thiếu là blocker thực tế cần báo, không phải lý do chờ phê duyệt kỹ thuật. Commit phát hành chứa nhật ký này; xác nhận publish bằng remote sau push, không suy thành người nhận đã đọc.

## 16/11/2026 - Bỏ cổng chờ duyệt cho Đức

- Theo yêu cầu trực tiếp của Gia Bảo: ban hành `GB-20261116-002`, cho Đức tự chọn thay đổi có kiểm soát, kể cả thành phần pooling/attention, tự triển khai/test/smoke/full và nộp sản phẩm trên `vinhduc`, không chờ duyệt M0/phạm vi thành phần kiến trúc hoặc review lại bản sửa trước chạy. Tùng giữ quy trình hiện tại.
- Đồng bộ AGENTS, README, bàn giao, protocol, bảng việc/nhiệm vụ, kênh trao đổi, roadmap và answers để AI không bị chỉ dẫn cũ chặn. Quyết định cũ giữ làm lịch sử, điều kiện chờ dành cho Đức đã thay thế; lỗi DUC-REV vẫn phải tự xử lý/test nếu dùng mã.
- Giữ mốc A3/07, dữ liệu/split/seed/giao thức, M0 ghi/khóa trước outer-test và test/smoke thật PASS. Review cuối quyết định nghiệm thu/merge main. Hạn 25/11/2026 và sản phẩm model đã train + kết quả trên GitHub không đổi.
- Không sửa bài làm/nhánh thành viên, model/kết quả/Word hoặc bật PoseLab; không giải quyết thay blocker input/GPU. Chưa độc lập review lại code Đức và không xác nhận hai bạn đã đọc. `git diff --check` và `python scripts/check_repository.py`: PASS (model, 9 notebook, Word, dự đoán/tổng hợp và liên kết không đổi/hợp lệ). Full train/review lại code Đức: `NOT_RUN`. Đối chiếu remote trước khi báo phát hành thành công.

## 16/11/2026 - Nhắc Tùng và Đức thực hiện nhiệm vụ

Lịch sử trước `GB-20261116-002`: các câu chờ duyệt trước chạy dành cho Đức dưới đây đã được quyết định mới thay thế.

- Gia Bảo bổ sung sản phẩm bàn giao: MODEL ĐÃ TRAIN + KẾT QUẢ ĐÁNH GIÁ phải có trên GitHub trước hạn 25/11/2026, không chỉ M0/tiến độ. Đồng bộ AGENTS, bảng việc và thông báo; thêm checklist model/load/OOF/metric/hash, vị trí riêng và quyền nộp trên nhánh cá nhân, Release asset cho artifact lớn. Không upload artifact thay thành viên, không duyệt full hay thay model triển khai; thiếu sản phẩm vẫn chưa hoàn tất và blocker phải báo sớm. Kiểm tra diff/checker trước publish; đối chiếu remote để xác nhận phát hành.
- Cập nhật tiếp theo theo yêu cầu Gia Bảo: chốt hạn đề xuất và báo cáo tiến độ cho cả hai là thứ Tư 25/11/2026 (UTC+7), không phải 18/11. Đồng bộ answers, AGENTS và bảng việc; bỏ câu chưa đặt deadline trong hướng dẫn hiện hành. Không duyệt M0/full hoặc thay yêu cầu input/GPU/review; phần chưa chạy vẫn phải báo đúng hạn với `NOT_RUN`/blocker. Kiểm tra diff và checker trước phát hành; xác nhận commit/push bằng Git/remote.
- Theo yêu cầu Gia Bảo, chuẩn bị thông báo `GB-20261116-001` trên main và thêm điểm đọc trong AGENTS/bảng việc; không sửa code, model, kết quả hoặc nhánh thành viên.
- Đã fetch: Tùng `1ab8164` (28/10), Đức `da5b552` (03/11). Ghi nhận Đức đã nộp bản sửa/đề xuất nhưng kiểm tra mới chỉ do Đức báo cáo; không nhận là đã độc lập nghiệm thu hoặc đã full train.
- Nhắc Tùng nộp M0, Đức tiếp tục bài đã nộp/cập nhật bằng chứng và blocker; cả hai xác nhận thông báo trên file trao đổi cá nhân ở phiên tiếp theo. Không tự duyệt M0/kiến trúc/full hay bỏ yêu cầu input/GPU/review. PoseLab tiếp tục dừng.
- `python scripts/check_repository.py` và `git diff --check`: PASS cho thông báo; full train và review lại code Đức: `NOT_RUN`. Phát hành qua commit thông báo trên main; đối chiếu Git/remote để xác nhận push, không suy ra hai bạn đã đọc từ việc publish.

## 30/10/2026 - Phân công hai hướng và dừng PoseLab

- Yêu cầu: Gia Bảo xác nhận mình là người điều phối; Tùng dùng mô hình/phương pháp khác, Đức thay đổi có kiểm soát mô hình hiện tại; không giao sẵn cách triển khai; tắt hoặc bỏ PoseLab.
- Thực hiện trên `giabao`, nền `c8f654b`: cập nhật AGENTS, README, bàn giao, protocol, bảng việc/nhiệm vụ, roadmap và quyết định `GB-20261030-001` trong answers. Giữ review cũ làm lịch sử, không xóa lỗi kỹ thuật hoặc bài làm Tùng/Đức; hủy WEB-01/WEB-02 và quyền frontend/QA.
- Tắt `app/start.cmd`, `app/start.ps1`; chặn ASGI lifespan trước Engine và middleware trả 503 kể cả không chạy lifespan. Giữ mã nguồn, dữ liệu lịch sử, model/threshold, notebook, kết quả và Word; bỏ phần quảng bá/hướng dẫn chạy PoseLab.
- `python scripts/check_repository.py`: **PASS**, hash model, 9 notebook, Word, dự đoán/tổng hợp và liên kết tài liệu không đổi/hợp lệ. `git diff --check`: **PASS**.
- Kiểm hành vi bằng Python tạm trong bộ nhớ: compile cú pháp toàn `app.py`; trích nguyên AST cờ/lifespan/middleware để chạy cùng FastAPI/TestClient, không import Engine hoặc tạo runtime. **PASS**: lifespan từ chối trước Engine; GET/POST/DELETE trả 503 khi không chạy lifespan. Đây là kiểm tra chặn app, không phải QA/inference toàn pipeline.
- Chạy thật hai launcher bằng subprocess: **PASS**, cùng thoát mã 1 và thông báo đã tắt; PowerShell thử cả cổng tùy chọn. Không cài thư viện/tải model/mở server. Không thấy tiến trình Python/uvicorn mang dấu hiệu PoseLab đang chạy lúc kiểm tra; không dừng tiến trình khác.
- Trạng thái giao: thay đổi local, **chưa commit/push/merge main**, không có PR; chưa tới máy hai thành viên. Publish main cần yêu cầu của Gia Bảo; sau đó hai bạn fetch/nhận cập nhật an toàn, không reset nhánh có bài làm. Full train: `NOT_RUN`.

## 30/10/2026 - Chốt mốc Đức và phát hành theo yêu cầu

- Yêu cầu tiếp: Gia Bảo cho phép commit/publish main sau khi làm rõ "hiện tại làm mốc" cho Đức. Dòng trạng thái local của mục trước là lịch sử phiên chuẩn bị, không mô tả trạng thái remote sau phát hành.
- Chốt `A3_motion_attention` / `fixed` của 07 tại nguồn `c8f654bf0f1943ffcfe375911c8b3e0015beadc5`, không phải checkpoint 06, A3_cost_225, T0/T1 hoặc Q1. Ghi nguồn model/features/recipe/input, kết quả lịch sử để nhận diện và yêu cầu tái lập bản gốc cùng bản thay đổi; không giao sẵn cách cải tiến.
- Đồng bộ vào AGENTS, README, bàn giao, protocol, bảng việc, nhiệm vụ Đức và quyết định GB-20261030-001. Không sửa helper, notebook, model, kết quả hoặc bài làm nhánh cá nhân.
- Phát hành bộ phân công/dừng PoseLab từ giabao qua PR về main sau kiểm tra; không force-push hoặc xóa nhánh. Kết quả thực tế phải xác nhận bằng trạng thái PR và SHA remote, không coi yêu cầu publish là bằng chứng push thành công. Hai bạn nhận main bằng fetch/merge an toàn; full train vẫn `NOT_RUN`.
- Kiểm tra lại trước phát hành: checker repo, diff check, nhận diện A3/feature/seed/metric gốc và kiểm chặn PoseLab đều **PASS**. Commit `777295926502207551d645f5a4bbbb2670ad14ba` đã push `origin/giabao`; [PR #1](https://github.com/aBaoN9/me-stgcn-ta-fall-detection/pull/1) về main đã mở, kiểm thấy `MERGEABLE` / `CLEAN`, không có status check CI được cấu hình. Các kiểm tra PASS là local, không gọi là CI PASS.
- Gia Bảo đã cho phép merge PR này lên main. Trạng thái merge/SHA cuối tra trực tiếp PR và `origin/main`; không tự cập nhật hoặc ghi đè nhánh `dinhtung`/`vinhduc` thay thành viên.
