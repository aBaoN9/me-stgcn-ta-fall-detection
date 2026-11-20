# Báo cáo công việc - Tùng

Người thực hiện: Tùng. Nhánh: `dinhtung`. [Nhiệm vụ](../tasks/dinhtung.md).

## Snapshot

- 19/11/2026. Tùng đã nhận `GB-20261118-001` tại `origin/main` `af9e7fb`. `APPROVED_TO_PROCEED / AUTONOMOUS`; M0 `M0_DOCUMENTED`.
- TUNG-02: dữ liệu, validation, kiểm thử phần mềm và smoke thật CPU PASS. Smoke Kaggle T4, TUNG-03/M2 và TUNG-04 `NOT_RUN`.
- M1 được lưu trong commit `6b863fa` trên `dinhtung`. Nhận main `e16ee1a` qua commit nhập nhánh `4a96d4d`.

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

### 2026-11-19 — M1 TCN và A3

- Thêm `training/backbone_benchmark/models.py` (TCN bốn khối, 32 kênh ẩn, kernel 3, dilation 1/2/4/8), `runner.py` (hash, Nested LOSO, đánh giá, checkpoint), `test_benchmark.py` và `README.md` (Kaggle T4). Cập nhật báo cáo, trao đổi và trạng thái M0. Dùng trực tiếp A3/07, không sửa nguồn hay kết quả gốc.
- `python -B -m training.backbone_benchmark.test_benchmark`: 7/7 PASS trên CPU. Kiểm thứ tự và kích thước tensor, mask, tích chập nhân quả, A3 tương đương, chặn thiếu/sai hash, chia theo người, chỉ số, vòng chạy tổng hợp 8 lần fit và nạp lại checkpoint. Kết quả tổng hợp chỉ dùng kiểm phần mềm, tệp đầu ra nằm trong thư mục tạm.
- Tìm cả tên chuẩn và tên tương thích trong dự án, không thấy bốn input. SHA-256 dữ liệu thật, kiểm căn chỉnh và smoke thật NOT_RUN. CUDA Kaggle/Colab do Tùng xác nhận, phiên này chưa chạy trên nền tảng đó và chưa xác nhận quyền dữ liệu.
- Trước M2: nhận nguồn đúng hash, kiểm input và chạy smoke thật trên thiết bị đã chọn, lưu bằng chứng và khóa cấu hình trước đánh giá ngoài. Phiên này giới hạn M1.

### 2026-11-19 — Gia cố M1 trước M2

- `runner.py`: ghi từng vòng vào thư mục tạm, kiểm trước khi đổi tên, thêm `--resume`. Khóa dữ liệu/mã/cấu hình/môi trường phải khớp. Vòng đã lưu chỉ được sử dụng sau khi đối chiếu split, lịch sử fit, SHA-256 và metadata checkpoint, ngưỡng và dự đoán từ model nạp lại. Vòng đã lưu bị lỗi làm chương trình dừng để bảo toàn bằng chứng.
- Full xác minh lại cả hai vòng smoke, 6 inner + 2 refit, checkpoint, dự đoán đủ mẫu, manifest và completion. `README.md` hướng dẫn sao lưu, khôi phục và giới hạn số đo bộ nhớ sau resume.
- `test_benchmark.py`: thêm ngắt khi đang lưu vòng thứ hai, tiếp tục chỉ chạy vòng thiếu, checkpoint hỏng, thiếu/sai seed hoặc vòng, khóa khác mã/cấu hình/thiết bị, smoke thiếu fit/dự đoán/sai metadata, thiếu evidence và evidence tổng hợp; đối chiếu A3 về epoch, lịch sử, điểm và ngưỡng. Lượt đầu có 3 lỗi do tuple chuyển thành danh sách trong JSON; đã sửa. Lượt cuối `python -B -m training.backbone_benchmark.test_benchmark`: PASS 10/10 trên CPU.
- `python -B scripts/check_repository.py` và `git diff --check`: PASS. Mã A3 và bằng chứng cũ giữ nguyên. Đã fetch và xác nhận quyết định vẫn ở `af9e7fb`. Không commit/push.
- Không thấy bốn tệp chuẩn trong dự án. Smoke thật `BLOCKED`; M2 `NOT_RUN`. Cần bộ dữ liệu có quyền đúng hash và chạy smoke thật trên môi trường GPU dự kiến dùng cho M2.

### 2026-11-19 — Nhận dữ liệu chuẩn và kiểm M1 bằng dữ liệu thật

- Đã nhận GB-20261119-001 và GB-20261119-002. Bốn tệp ở `data/gmdcsa24/` đến từ main. Công cụ `python scripts/training_inputs.py --input-root data/gmdcsa24` PASS 4/4 hash và dung lượng.
- `python -B -m training.component_ablation.validate_pipeline --input-root data/gmdcsa24` PASS với 160 mẫu, 79 Fall, 81 ADL và bốn người. Căn chỉnh input và tách người đạt. Sai số đối chiếu đặc trưng 1.1920928955078125e-07, kiểm 100 trường hợp tăng cường và kích thước năm biến thể gốc.
- `python -B -m training.backbone_benchmark.test_benchmark` PASS 10/10. Checker repo và git diff --check PASS. Đối chiếu Git với main xác nhận mã/kết quả gốc và phần Đức giữ nguyên.
- Smoke thật CPU PASS với Python 3.10.11, Torch 2.12.1+cpu, NumPy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2; OMP/MKL đặt 4 luồng. Lệnh: `python -B -m training.backbone_benchmark.runner --input-root data/gmdcsa24 --output .runtime/tung-smoke-cpu-20261119 --mode smoke --device cpu`.
- Bằng chứng local tại `.runtime/tung-smoke-cpu-20261119/`. Chữ ký `86876850d5898c7dd25e3bc975d0847eb71e51ba946702f3db49be22c55064d1`. Hai mô hình, seed 42, test S1, 6 inner + 2 refit, 192 dòng dự đoán. Hai checkpoint khớp SHA-256 trong manifest, reload/coverage PASS. Chạy lại với `--resume` PASS. `synthetic=false`, `scientific_result=false`. Không đưa checkpoint thử vào Git.
- Kaggle T4 NOT_RUN. Phiên này không có kết nối tới phiên Kaggle của Tùng. Dùng hướng dẫn Kaggle trong README vùng M1 với `--input-root data/gmdcsa24` khi checkout nhánh đã đồng bộ. Lưu thư mục run ở nơi có thể khôi phục và kiểm smoke GPU trước M2. Full NOT_RUN.

### 2026-11-20 — Kiểm khả năng huấn luyện trên RTX 3050

- Mã `dinhtung` tại `12eaf80`, quyết định main `e16ee1a`. Laptop Ryzen 5 7535HS (6 nhân/12 luồng), RAM báo 15,32 GiB, ổ C là SSD NVMe WD SN740, RTX 3050 Laptop 4 GiB và driver 566.07.
- Môi trường riêng `.venv`: Python 3.11.9, Torch 2.10.0+cu126, CUDA 12.6, NumPy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2. Kiểm phụ thuộc và phép nhân ma trận CUDA/đổi sang NumPy PASS. Môi trường Python hệ thống vẫn giữ nguyên.
- Hash dữ liệu PASS 4/4, validation PASS 160 mẫu, unit test PASS 10/10. Checker ban đầu đi vào tài liệu thư viện dưới tên `.venv-rtx3050`; chuyển sang `.venv` theo thư mục checker đã loại trừ và chạy lại PASS, giữ nguyên checker.
- Smoke thật CPU/GPU là các lượt riêng. RTX 3050 chạy `--mode smoke --device cuda` cho TCN và A3 với OMP/MKL 4 luồng, đủ 8 lần fit, 2 outer, 192 dòng dự đoán, reload/coverage PASS, synthetic=false, scientific_result=false. Thời gian đo gồm khởi động và ghi kết quả khoảng 14,45 giây. Bằng chứng local: `.runtime/tung-smoke-rtx3050-20261120-160820/`, chữ ký `0a135996fe5686f40c989945ba3a1d190c3ebd45b4417699d6f29867b4935c95`. Nhật ký đo và tóm tắt nằm cạnh thư mục này.
- Đỉnh PyTorch cấp phát 154441728 byte (147,3 MiB); 19 mẫu từ NVIDIA cho thấy VRAM dùng đỉnh 306 MiB, GPU 57%, nhiệt độ 60°C, RAM hệ thống dùng đỉnh 13169,4 MiB. Ổ C còn khoảng 42 GiB sau cài/chạy. `--resume` PASS với bản sao chỉ có vòng A3 đã lưu từ RTX 3050, giữ nguyên hash A3 và bổ sung vòng TCN. Nạp lại cả lượt hoàn tất bằng `.venv` cũng PASS.
- 24 checkpoint của M2 ước tính 22,7 MiB theo kích thước hai checkpoint smoke, chưa tính lịch sử/dự đoán. Nên chừa ít nhất 1 GiB cho run và bản sao, sao lưu lock cùng các thư mục fold đã hoàn tất sang nơi bền vững khác. M2 NOT_RUN trong phiên này; khi chạy, dùng thư mục full mới và smoke evidence RTX 3050 cùng môi trường đã xác minh. Theo dõi tài nguyên thực tế suốt lượt dài.

### 2026-11-20 — M2 đầy đủ trên RTX 3050

- M2 PASS tại `.runtime/tung-m2-rtx3050-20261120-165523/`, mã `12eaf80`, main `e16ee1a`, chữ ký `24bb52cca33da8e171254850d6266bd1ece6a21f53e629f2c75c87b0744a3222`. Khóa cấu hình trước đánh giá, SHA-256 input 4/4, runner xác minh lại smoke RTX 3050. Python 3.11.9, Torch 2.10.0+cu126, NumPy 2.2.6, pandas 2.3.3, sklearn 1.7.2, OMP/MKL 4 luồng. Lệnh: `.venv/Scripts/python.exe -B -m training.backbone_benchmark.runner --input-root data/gmdcsa24 --output .runtime/tung-m2-rtx3050-20261120-165523 --mode full --device cuda --smoke-evidence .runtime/tung-smoke-rtx3050-20261120-160820`.
- 96/96 lượt, 48 mỗi mô hình, 72 inner + 24 refit; 24 outer, reload và coverage PASS, scientific_result=true. Tái tính 2.880 dự đoán theo fold/seed khớp bảng đã lưu; kiểm hash 24 checkpoint và lịch sử giữa thư mục chính và bản sao. Thời gian tường 314,50 giây. 407 mẫu đo: nhiệt độ GPU đỉnh 78°C, VRAM 308 MiB, GPU 73%, RAM hệ thống 14752,7 MiB. PyTorch cấp phát đỉnh 154441728 byte. Không gián đoạn.
- Chính sách chính fixed=0,5: trung bình qua ba seed, Accuracy/Sensitivity/Specificity/F1 (%) lần lượt TCN 91,04/94,09/88,07/91,20 và A3 89,38/92,41/86,42/89,57. TCN có Accuracy cao hơn trung bình 1,67 điểm phần trăm nhưng thấp hơn A3 ở seed 242. TCN bỏ sót S1_FALL_13, S2_FALL_14, S4_FALL_03 ở cả ba seed; A3 bỏ sót S4_FALL_05 ở cả ba seed. Xem `fixed_error_cases.csv` và `errors.csv`. Đánh giá có kiểm soát trên bốn người đã dùng trong phát triển, không suy ra ưu thế trên tập độc lập.
- Bản sao theo vòng tại `.runtime/tung-m2-rtx3050-20261120-165523-backup/`. ZIP cùng tên run, 21554783 byte, SHA-256 `814415188654587a1e0936c0b34500d9d9cdac8ea48932f95c711b848415f15c`. Đã giải nén thử và đối chiếu hash đủ 60 tệp. Checker repo và git diff --check PASS. Chưa commit/push; khi bàn giao đưa kết quả, checkpoint, manifest và đường dẫn ZIP đã kiểm lên nhánh được giao theo quy định.

## Bàn giao review

- Checklist nhiệm vụ: chưa hoàn tất.
- Lệnh tái lập / thời gian / tài nguyên đã dùng: chưa có.
- Khác paper/recipe (quy trình cấu hình), giới hạn kết luận: có thiết kế/adaptation trong M0; chưa có bằng chứng thực nghiệm mới. TCN thay cả backbone (phần mạng học biểu diễn) và tổng hợp, không cô lập tác dụng riêng một thành phần.
- PR về main: chưa mở; sau push main có thể đọc `git show origin/dinhtung:docs/reports/dinhtung.md` hoặc PR.

### 2026-11-20 — Kiểm chứng và đóng gói M3

- PASS 24 checkpoint nạp lại trên CUDA và đối chiếu đủ 2.880 dự đoán. PASS SHA-256 ZIP gốc, 60 tệp khớp byte; checkpoint/lịch sử khớp bản sao. Chỉ số độc lập khớp 1e-12, kiểm thử 10/10 PASS. Không chạy lại M2.
- Đo suy luận riêng trên RTX 3050: TCN 56.434 tham số, 2,850/3,034 ms trung vị/p95; A3 422.871, 3,331/4,238 ms.
- [Bộ bàn giao M3](../../results/backbone_benchmark/tung-m2-rtx3050-20261120-165523/README.md) có đủ checkpoint, dự đoán, chỉ số, lịch sử, hash và hướng dẫn. [Phân tích khoa học](../research/backbone-benchmark.md) giữ nguyên lịch sử M0.
