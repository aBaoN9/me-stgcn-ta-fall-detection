# Báo cáo công việc - Gia Bảo

Người thực hiện: Gia Bảo. Nhánh phát triển: `giabao`. Vai trò: chủ repo/người điều phối. [Nhiệm vụ](../tasks/giabao.md).

## Snapshot

- Cập nhật: 20/11/2026, đã nghiệm thu M0–M3 TCN–A3 của Tùng và merge PR #2 tại `bbe3c64638ce2cfae31348cb64a455a97c3548bf`; đồng bộ README/quyết định/bàn giao/bảng việc theo `GB-20261120-001`. Kiểm tra trên main sau merge: 10/10 tests, 68 hash artifact, input/code lock, bảng README và repo checker PASS. Không train full mới, đổi model triển khai, sửa Word hoặc nghiệm thu thay bài Đức.
- Tùng dùng mô hình/phương pháp khác trên cùng dataset; Đức thay đổi có kiểm soát ME-STGCN-TA. Không áp sẵn mô hình/công thức, cùng giao thức đánh giá. Cả hai tự ghi/khóa M0 và kiểm test/smoke thật rồi full, không chờ duyệt từng bước.
- Quyết định chính thức và các phản hồi cho Tùng/Đức vẫn ở `docs/discussions/answers.md` trên main; báo cáo này không thay kênh phê duyệt.
- Trạng thái tạo/publish nhánh phải đối chiếu Git/remote, không dùng nội dung báo cáo này làm bằng chứng push thành công.

## Nhật ký tiếp theo

Mỗi phiên nối thêm ngày, yêu cầu, phạm vi, file/commit/PR, lệnh kiểm tra và PASS/FAIL/NOT_RUN, giới hạn/blocker và bước tiếp. Khi review nhánh khác ghi đúng commit; khi tích hợp ghi phần được duyệt và bằng chứng, không ghi code chưa review thành DONE.

## 20/11/2026 - Nghiệm thu Tùng, merge PR #2 và cập nhật README

- Gia Bảo yêu cầu bổ sung bài Tùng lên main và cập nhật README, sau khi được báo đủ điều kiện nghiệm thu. PR #2 giữ head `b978ff0fdd2e4a9f38d9c2cd5bcc72950d98bcf4`, base `e16ee1a`, MERGEABLE và không có CI check được cấu hình. Các PASS dưới đây là local, không gọi là CI PASS.
- Trước tích hợp đã kiểm 68 hash, raw OOF/metric, nhãn/video/subject, bốn input chuẩn, code lock, lịch sử 96 fits và chia subject; nạp và suy luận lại 24 checkpoint trên CPU cho cùng 2.880 quyết định. Score khác tối đa khoảng `0.000508` so với run CUDA; không suy ra tái lập bit-exact, latency GPU hoặc train lại. 10/10 tests PASS, gồm đối chiếu A3 gốc, hash/coverage/reload/resume và guard full/smoke.
- Thực hiện `gh pr merge 2 --merge --match-head-commit b978ff0fdd2e4a9f38d9c2cd5bcc72950d98bcf4`, không squash/xóa nhánh; GitHub xác nhận MERGED ngày 20/11, merge commit `bbe3c64638ce2cfae31348cb64a455a97c3548bf`. Fetch và fast-forward local main. Không sửa/push nhánh Tùng hoặc Đức.
- `GB-20261120-001` chốt `ACCEPTED_BENCHMARK` M0–M3 Tùng, không yêu cầu tìm điểm A3 thắng để được nhận bài. README tách A3 lịch sử 07 (89,58%) và A3 chạy lại trong run mới (89,38%); thêm định nghĩa/kiến trúc TCN, bảng mean/SD và từng seed, tham số/checkpoint/latency cùng điều kiện đo, lỗi lặp, giới hạn dữ liệu, artifact và hướng dẫn kiểm/chạy mới. Không đưa số chưa đo hoặc coi 2.880 dòng dự đoán là video độc lập.
- Đồng bộ snapshot nghiệm thu trong AGENTS, nhiệm vụ/bảng việc, bàn giao và hướng dẫn train; không sửa báo cáo/trao đổi cá nhân Tùng sau merge. Ghi đề xuất nghiên cứu tiếp trong roadmap là chưa thực hiện, có nguyên tắc chọn ngưỡng/robustness/phân tích lỗi công bằng và công bố post-hoc. Không giao thêm thí nghiệm bắt buộc hoặc thay nhiệm vụ/mốc Đức.
- Kiểm trên main đã tích hợp: `python -B -m training.backbone_benchmark.test_benchmark` (đặt 4 luồng Torch trong harness, Torch 2.11.0+cpu) **10/10 PASS**; 68 hash artifact và toàn bộ code lock/bốn input không đổi **PASS**; tính lại mean/sample-SD từ CSV fixed rồi đối chiếu bảng README **PASS**; `python scripts/check_repository.py` và `git diff --check` **PASS**. CSV/JSON artifact giữ CRLF theo `.gitattributes`, không normalize để đổi hash. Kiểm lại liên kết sau cập nhật report trước publish.
- Sau merge chỉ thay tài liệu điều phối trên main, publish theo yêu cầu và kiểm SHA remote. Không sửa scientific code/lock/manifest, notebooks 06/07/08, model/threshold triển khai hoặc Word; PoseLab vẫn dừng. Full train mới và tái đo GPU **NOT_RUN**; công việc Đức chưa nghiệm thu trong phiên này.

## 19/11/2026 - Rà lần cuối và gỡ các vướng mắc vận hành

- Yêu cầu Gia Bảo: kiểm lại phần cần thiết cho Tùng/Đức và đưa lên main, không bắt duyệt từng bước. Làm code hỗ trợ trên `giabao` sau fast-forward từ main `afc7370`; không sửa/push nhánh hai bạn.
- Fetch xác nhận `dinhtung` ở `d65e59c5396503f48c05a0d102fd2b4e829d0bd9` (M0 TCN), `vinhduc` ở `eac041170f4d85a059386c87df2f1ad332a91a38`. Đức hỏi full CPU trong `DUC-20261117-001`; notebook vẫn pin `c72f33a547d3f130d3768611c08949cf45c37b56`, chỉ tìm input trong `/kaggle/input`, config cấm CPU full và cell full assert CUDA. Chỉ có input trên main chưa tự giải quyết các chỗ này.
- Quyết định `GB-20261119-002`: trả lời đúng ID, cả hai được tự full CPU khi không có GPU, hai vế cùng điều kiện, tự ghi addendum/khóa code-config trước outer-test; tự chọn commit checkpoint nhỏ đích danh hoặc Release, không xin lại. Không bỏ guard, thay giao thức, cấp tiền/tài khoản hoặc nhận code đã nghiệm thu. Sửa chỉ dẫn cũ trong nhiệm vụ Gia Bảo còn bắt Tùng duyệt M0; đồng bộ điểm vào và nhiệm vụ hai bạn.
- Thêm `scripts/training_inputs.py` (Python chuẩn): mặc định verify-only, `--download` chỉ tải file thiếu từ data commit bất biến `afc7370`, xác thực size/SHA-256 trước khi lưu, từ chối ghi đè. Thêm `docs/training-start.md`: merge an toàn, dependency/runtime, Kaggle tách code/data và thay cell tìm input, CPU cần cập nhật lock trong code riêng, kiểm tra/nộp artifact. Không sửa notebook/runner của Đức, helper/model/06-07-08, protocol hoặc kết quả cũ.
- Kiểm tra local **PASS**: 9 ca stdlib unittest trong bộ nhớ (verify thiếu không tạo thư mục, file đúng không gọi mạng, sai size/hash giữ nguyên, chỉ tải phần thiếu, download sai hash không cài file, lỗi mạng dọn staging, file phát sinh đồng thời không bị ghi đè, lỗi copy dọn file dở). Đối chiếu bảng size/hash của tool với `input_manifest.json` và map hash `inputs` của protocol 07 PASS; CLI thiếu file trả lỗi và không ghi dữ liệu PASS.
- Chạy thật `python scripts/training_inputs.py --input-root data/gmdcsa24` và `python scripts/training_inputs.py --input-root .runtime/readiness/downloaded_inputs --download`: **PASS**, bốn file từ public GitHub đúng hash. `python -m training.component_ablation.validate_pipeline --input-root .runtime/readiness/downloaded_inputs`: **PASS**, 160 mẫu, parity `1.1920928955078125e-07`, 100 ca augmentation, shape 5 biến thể gốc. `python scripts/check_repository.py` và `git diff --check`: **PASS**. Máy kiểm: Python 3.11.9, torch 2.11.0+cpu, numpy 1.26.4, pandas 2.2.3, sklearn 1.8.0, không CUDA; không tuyên bố khớp môi trường pinned hoặc máy thành viên.
- Phát hành phần hỗ trợ lên main theo yêu cầu; đối chiếu SHA remote sau push. Tài khoản/quota, dependency, smoke thật và full trên máy hai bạn chưa được xác minh; không suy ra hai bạn đã đọc/nhận ở máy riêng. Tùng vẫn tự triển khai runner/model; Đức vẫn tự áp dụng thay đổi vận hành ở vùng riêng. Không bật PoseLab, không train hộ hoặc merge bài chưa review.

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
