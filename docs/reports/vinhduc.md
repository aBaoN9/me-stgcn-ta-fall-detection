# Báo cáo công việc - Đức

Người thực hiện: Đức. Nhánh: `vinhduc`. [Nhiệm vụ](../tasks/vinhduc.md).

## Snapshot

- Trạng thái: `IN_PROGRESS`.
- Cập nhật: 2026-11-18 UTC+7. Đã nhận `GB-20261116-001` (hạn 25/11/2026: model + kết quả trên GitHub) và `GB-20261116-002` (tự chủ M0/smoke/full). M0 tự khóa `M0_DOCUMENTED`, thêm addendum 18/11 khóa phân tích DUC-04 trước kết quả thật. Code/test/notebook Kaggle/phân tích sẵn sàng. Smoke thật/full `NOT_RUN` vì vẫn thiếu bốn input (`DUC-20261117-001`); GPU không còn là blocker cứng (ước tính CPU ≤ ~3,9 giờ, cần cho phép).
- Commit code / run ID / PR:
  - `1cbb778`: code DUC-01..03.
  - `2368345`: bản sửa DUC-REV-01..05.
  - Phiên 17/11: merge `origin/main` (`8ab7b29`), M0 lock, checkpoint `models/`, reload, quality report, notebook Kaggle. Commit trong lần push này trên `vinhduc`.
  - Chưa có run thật. PR về main mở sau khi có model + kết quả, hoặc tại hạn với phần đã làm.
- M0: `M0_DOCUMENTED` ngày 17/11/2026 theo `GB-20261116-002`, file `results/contribution_study/m0_locked.json`, code hash `6ba0420b4b2a…`, ε = 0,1, 24 outer / 96 fits.
- Mốc:
  - DUC-01 `READY_FOR_REVIEW` (failure mode chưa xác minh bằng video).
  - DUC-02 `M0_DOCUMENTED`.
  - DUC-03 `BLOCKED`: thiếu bốn input đúng hash + GPU.
  - DUC-04 `IN_PROGRESS`: kế hoạch + code phân tích khóa và kiểm PASS 12/12 trên smoke giả; chạy thật `NOT_RUN`, chờ DUC-03.
  - WEB-01 `CANCELLED`.

Trạng thái: `TODO`, `IN_PROGRESS`, `BLOCKED`, `READY_FOR_REVIEW`, `DONE`. DUC-04 có thể `WAITING_FOR_TUNG` riêng, không dùng để bỏ DUC-01 đến DUC-03. Chỉ người điều phối xác nhận DONE sau review. Nhật ký nối thêm, không xóa lỗi cũ.

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
- Bước tiếp: audit DUC-01; chuẩn bị công thức/related work DUC-02.

### 2026-10-28 - DUC-01 audit OOF 07/08

- Đã làm / lý do: chuyển sang `vinhduc` (worktree sạch, trùng `origin/main` 1ab8164); đọc `origin/main:docs/discussions/answers.md` (chưa có quyết định). Viết script tái tính metric độc lập từ `predictions.csv` 07/08, kiểm toàn vẹn, tách 6 cặp một yếu tố theo seed/subject/policy, thống kê clip lỗi lặp. Ưu tiên DUC-01 vì làm được ngay không cần GPU/dữ liệu; chưa làm WEB-01.
- File / commit thay đổi: `training/contribution_study/audit_existing.py`, `training/contribution_study/README.md`, `results/contribution_study/audit/*`, `docs/research/contribution-study.md`, báo cáo này. Chưa commit.
- Lệnh kiểm tra và PASS/FAIL/NOT_RUN: `python -m training.contribution_study.audit_existing` PASS 22/22; `python scripts/check_repository.py` PASS.
- Input/hash, protocol/hash, code/hash, environment, seed, thiết bị: 07 `predictions.csv` sha256 `2d229063…b8bb8`, 08 `predictions.csv` `43cbc6cc…d286` (đủ trong `audit_checks.json`); script `b46012da…1192`; Python 3.12.10, pandas 3.0.5, numpy 2.5.3, Windows 11, CPU; không có seed mới (chỉ đọc seed 42/142/242 cũ).
- Mode / số outer-inner-refit thực tế và kỳ vọng / coverage: không train. Coverage OOF: 48 nhóm 07 + 18 nhóm 08, mỗi nhóm đủ 160 ID / 79 Fall / 81 ADL.
- Bằng chứng: bảng cũ tái tạo khớp (sai lệch ≤ 3,6e-15). Fixed policy: motion A0→A1 +4,79 ± 3,66 pp, tăng 3/3 seed; attention A1→A3 +0,21 ± 2,53 pp, không nhất quán; cost: ΔFN −3,7 / ΔFP +3,7, accuracy không đổi; T1 −1,88 ± 2,25 pp.
- Kết quả âm / FN-FP: motion trong nhánh attention giảm sensitivity −1,27 pp; attention A1→A3 âm ở policy balanced/sensitivity; `S4_FALL_05` đúng ở biến thể chỉ pose nhưng sai ở mọi biến thể có motion; `S4_ADL_05/06` FP 21/21 lần.
- Phát hiện phụ: score T0 (08) giống hệt A3_cost_225 (07) trên 480 cặp → không tính là hai bằng chứng độc lập.
- Còn thiếu / blocker: chưa có video/pose/NPZ trên máy nên failure mode chưa xác minh; chưa có `data/local/` để khóa q cho DUC-02.
- Việc main cần quyết định: review DUC-01; cấp nguồn bốn input NPZ/CSV (đúng hash `locked_protocol.json`) để chuẩn bị M0 DUC-02.
- Bước tiếp: related work confidence/occlusion-aware skeleton; đọc `models.py` để ánh xạ q qua stride; soạn M0 gửi qua `docs/discussions/vinhduc.md`.

### 2026-10-28 - DUC-02 M0 và code DUC-03 (chưa train thật)

- Đã làm / lý do:
  - Đọc `models.py`/`features.py`: `masks` là cờ theo frame (32,), kênh 2 là visibility đã mask; attention có 16 bước sau stride 2.
  - Khóa đề xuất `w = softmax(a + log(0,1 + q'))`, q từ kênh 2 của input thô (khớp augmentation). Đọc related work chính chủ (ST-GCN, Attention-MIL, RA-GCN, PeGCN, PoseConv3D, STEP). Kết luận: chỉ là adaptation + kiểm chứng, không tuyên bố mới.
  - Viết model/runner/check/latency. Gửi câu `DUC-20261028-001` xin duyệt M0 và cấp input/GPU.
- File: `training/contribution_study/{quality_pooling,runner,check_quality_pooling,benchmark_latency}.py`, README training/results, `results/contribution_study/latency/model_only_pytorch_cpu.json`, `docs/research/contribution-study.md`, `docs/discussions/vinhduc.md` (đã commit/push riêng).
- Lệnh kiểm tra và PASS/FAIL/NOT_RUN:
  - `.venv/Scripts/python -m training.contribution_study.check_quality_pooling` PASS 13/13, có thêm kiểm đột biến thủ công.
  - `benchmark_latency` đã đo.
  - Full Q0/Q1 `NOT_RUN`. Smoke trên dữ liệu thật `NOT_RUN` (không có input).
  - `python scripts/check_repository.py` PASS.
- Environment:
  - `.venv` riêng (Git bỏ qua), không đụng Python hệ thống: Python 3.12.10, torch 2.10.0+cpu, numpy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2.
  - Máy Windows 11, CPU AMD, không GPU NVIDIA.
  - Smoke dữ liệu giả seed 42, ở `.runtime/` ngoài Git, không phải kết quả.
- Mode / outer-inner-refit: smoke giả 8/8 outer, 32/32 fits. Full kỳ vọng 24 outer / 96 fits, chưa chạy.
- Chi phí: Q0 và Q1 cùng 422.871 tham số. CPU 1 thread batch 1: Q0 median 7,45 / p95 9,32 ms; Q1 7,90 / 10,94 ms, model-only PyTorch. Không phải ONNX/pipeline.
- Kết quả tốt/âm/FN-FP: chưa có, không có run thật.
- Blocker:
  - Bốn input không có trên máy (đã tìm C:/D:, không thấy). Không tự tải dataset khi chưa rõ quyền/hash.
  - Không GPU.
  - M0 chưa duyệt.
- Việc main cần quyết định: xem `DUC-20261028-001`.
- Bước tiếp: khi có input thì chạy `validate_pipeline`, smoke thật và báo phân bố q trước full. Sau duyệt, chạy full Kaggle rồi audit cặp Q0→Q1.

### 2026-11-03 - Nhận phân công mới, sửa DUC-REV-01..05

- Đã làm / lý do:
  - Fetch và đọc `origin/main:docs/discussions/answers.md` tại `f6a23a5`: có GB-20261030-001 (phân công mới, hủy WEB-01) và review `DUC-20261028-001` `CHANGES_REQUESTED`.
  - Merge `origin/main` vào `vinhduc` (không xung đột, không reset/rebase).
  - WEB-01 đã hủy nên không làm UI, dù yêu cầu phiên ban đầu có nhắc `web-app.md`.
  - Đức chọn giữ hướng quality-aware. Đã sửa đủ các lỗi review bắt buộc vì code vẫn dùng tiếp.
- File / commit thay đổi:
  - `training/contribution_study/{runner,audit_existing,benchmark_latency}.py`, mới `check_review_fixes.py`, README training.
  - `results/contribution_study/audit/audit_checks.json`: sinh lại, CSV không đổi.
  - `docs/research/contribution-study.md`, báo cáo này, `docs/discussions/vinhduc.md`.
  - Code chưa commit.
- Từng DUC-REV:
  - REV-01: `verify_inputs` dừng trước fit nếu thiếu/thừa input hoặc sai hash so với `results/component_ablation/locked_protocol.json`. Synthetic chỉ chạy smoke, `scientific_result=false`. Test: bộ chuẩn qua; đổi hash hoặc thiếu từng input trong bốn input đều bị chặn, output không được tạo.
  - REV-02: `validate_fold` kiểm complete, chữ ký run, inner subject, fits, đủ ba policy, ID/index/subject/label khớp nguồn, score/threshold hữu hạn, prediction khớp threshold, metric tái tính, artifact `.pt`/attention/history. Resume chỉ bỏ qua fold hợp lệ, fold hỏng chạy lại. `summarize_run` kiểm đúng tập variant×seed×outer và nhóm OOF, xóa `completion.json` cũ trước khi kiểm. Ca review tái hiện nay bị chặn; thêm 15 mutation âm, 1 dương và 1 resume.
  - REV-03: `make_sync` chuẩn hóa `torch.device`, đồng bộ `cuda` và `cuda:0` đúng thiết bị (mock). Số CPU cũ không đo lại; GPU latency `NOT_RUN`.
  - REV-04: `compare` không còn `nanmax`; NaN chỉ khớp NaN, còn NaN với số hữu hạn, inf, khóa trùng, thiếu hàng/cột đều FAIL. Seed summary dùng cùng hàm. Ca 0,9/0,8 so NaN/0,8 nay FAIL. Audit 07/08 nguyên trạng vẫn PASS 22/22 + 1 INFO, 0 ô NaN.
  - REV-05: thêm `audit_existing --run <run> --baseline --treatment` cho cặp Q0→Q1, chạy thử trên smoke giả PASS. Báo cáo, README và tài liệu đã cập nhật.
- Lệnh kiểm tra và PASS/FAIL/NOT_RUN:
  - `.venv/Scripts/python -m training.contribution_study.check_review_fixes` PASS 44/44.
  - `.venv/Scripts/python -m training.contribution_study.check_quality_pooling` PASS 13/13.
  - `python -m training.contribution_study.audit_existing` PASS.
  - `python scripts/check_repository.py` PASS.
  - Smoke thật / full / latency GPU: `NOT_RUN`.
- Code hash (SHA-256, 12 ký tự): `runner.py` 5c089a8b0b62, `audit_existing.py` 6147ac68a163, `benchmark_latency.py` 185fc55bb11e, `check_review_fixes.py` 98bac7b0bbda, `quality_pooling.py` 3f3e1cecf8c8, `check_quality_pooling.py` c27b43cbc115.
- Environment: `.venv` Python 3.12.10, torch 2.10.0+cpu, numpy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2. Audit dùng Python hệ thống 3.12.10. Windows 11, CPU, không GPU. Fixture/smoke giả seed 3/7/11, nằm ngoài Git (`.runtime/`), không phải kết quả.
- Mốc: `training/component_ablation/` giống hệt `c8f654b` (`git diff` rỗng). Q0 = A3_motion_attention fixed.
- Kết quả tốt/âm/FN-FP: chưa có run thật.
- Blocker / main cần quyết định: duyệt M0 mở rộng phạm vi `DUC-20261103-001`; review lại bản sửa `DUC-20261028-001`; cấp bốn input đúng hash và GPU (vẫn chờ từ 28/10).
- Bước tiếp: khi Đức yêu cầu thì commit code và mở PR; có input thì chạy smoke thật và báo phân bố q trước full.

### 2026-11-17 - Nhận quyền tự chủ, khóa M0, chuẩn bị sản phẩm model/kết quả

- Đã làm / lý do:
  - Fetch, đọc `origin/main:docs/discussions/answers.md` tại `8ab7b29`: `GB-20261116-001`, `GB-20261116-002`. Merge `origin/main` vào `vinhduc`, không xung đột, không reset/rebase.
  - Kiểm lại parity Q0 với 07: `fit_model`/`run_neural` chỉ khác factory model và guard loss không hữu hạn. Seed inner/refit, epoch median, chọn ngưỡng inner, loader outer giống hệt.
  - Khóa M0 (`results/contribution_study/m0_locked.json`): mốc, thay đổi, phần giữ nguyên, ngân sách, metric chính, quy tắc báo cáo, giới hạn, config full, code hash, hash từng file, bốn hash input. Khóa trước mọi kết quả outer của Q0/Q1.
  - Theo checklist `GB-20261116-001`:
    - Runner lưu checkpoint vào `<run>/models/`, kèm seed/refit seed/outer/variant/chữ ký run.
    - Thêm `evaluate_models.py`: nạp lại từng checkpoint, chấm lại outer, đối chiếu score/metric, ghi `models/manifest.json` (SHA-256) và `model_reload_check.json`.
  - Sửa lỗi `runner.code_hash()`: hash byte thô nên checkout Windows (CRLF) và Kaggle (LF) cho hash khác nhau. Lỗi này làm hỏng kiểm khóa M0 và kiểm nạp lại model chạy từ Kaggle. Nay chuẩn hóa CRLF→LF trước khi hash; hash khóa tính lại là `6ba0420b4b2a…`.
  - Thêm `quality_report.py`: phân bố q/frame mất pose trên input thật trước full, review 29/10 đã yêu cầu.
  - Thêm `kaggle_q0_q1.ipynb`: chạy từ đầu đến cuối trên Kaggle GPU, dừng nếu code/config khác M0 hoặc sai hash input.
  - Tìm lại bốn input trên C:/D: (không có), kiểm GPU (không có), Kaggle API (chưa cấu hình), `gh` CLI (không có). Gửi `DUC-20261117-001`.
- File thay đổi:
  - `training/contribution_study/{runner,check_review_fixes}.py`.
  - Mới: `evaluate_models.py`, `quality_report.py`, `kaggle_q0_q1.ipynb`.
  - README training, `results/contribution_study/m0_locked.json`, `docs/research/contribution-study.md`, báo cáo này, `docs/discussions/vinhduc.md`.
- Lệnh kiểm tra và PASS/FAIL/NOT_RUN:
  - `.venv/Scripts/python -m training.contribution_study.check_review_fixes` PASS 50/50. Gồm 44 test cũ và 6 test reload: run hợp lệ PASS; trọng số bị nhiễu, metadata sai fold, thiếu checkpoint, score lệch, pooled bị sửa đều FAIL. Lần đầu test trọng số nhiễu báo sai do lỗi trong chính test (cộng đều hai logit nên softmax không đổi); đã sửa test để chỉ nhiễu logit Fall.
  - `.venv/Scripts/python -m training.contribution_study.check_quality_pooling` PASS 13/13.
  - `python -m training.contribution_study.audit_existing --output .runtime/...` PASS (07/08 nguyên trạng).
  - `python scripts/check_repository.py` PASS.
  - `quality_report` chạy thử trên dữ liệu giả để kiểm phần mềm (160 clip, không NaN), không phải kết quả.
  - Smoke thật / full / reload trên run thật / latency GPU: `NOT_RUN`.
- Environment: `.venv` Python 3.12.10, torch 2.10.0+cpu, numpy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2. Windows 11, CPU, không GPU. Fixture seed 3/7/11, ở `.runtime/` ngoài Git.
- Kết quả tốt/âm/FN-FP: chưa có run thật.
- Blocker / main cần quyết định (`DUC-20261117-001`):
  1. Cấp quyền truy cập bốn input đúng hash (ví dụ share Kaggle Dataset).
  2. Có cho full trên CPU không, nếu tới 22/11 vẫn không có GPU.
  3. Commit trực tiếp khoảng 41 MB checkpoint hay dùng Release asset.
- Bước tiếp:
  - Có input thì chạy notebook Kaggle (đặt `COMMIT` = commit đã push).
  - Tải zip về `results/contribution_study/runs/<run>/`, kiểm lại `evaluate_models` trên máy.
  - Commit model + kết quả, viết phân tích DUC-04, mở PR trước 25/11.

### 2026-11-18 - Khóa phân tích DUC-04, ước tính CPU, sửa resume notebook

- Đã làm / lý do:
  - Fetch `origin/main` tại `af9e7fb`: chưa có trả lời cho `DUC-20261117-001`. Tìm lại bốn input trên C:/D: (không có); không GPU, không Kaggle API/`gh`. Repo public (HTTP 200) nên notebook Kaggle clone được.
  - Đo thời gian CPU trên dữ liệu synthetic (chỉ đo thời gian): 3,85 s/epoch với 120 clip, 16 luồng. Trường hợp xấu nhất 96 fits × 50 epoch ≈ 3,9 giờ. GPU không bắt buộc về thời gian; chạy CPU vẫn cần cho phép vì M0 `allow_cpu_full=false`.
  - Viết `analyze_q0_q1.py` (DUC-04) trước mọi kết quả thật và khóa bằng addendum trong `m0_locked.json` (`addenda[0]`, SHA-256 file phân tích). Gồm delta cặp theo seed/subject, confusion, clip đổi kết quả, cơ chế attention–q', clip lỗi DUC-01 và kiểm tái lập Q0 chỉ để cảnh báo môi trường. Không đổi code train/config/input; `code_hash` M0 vẫn khớp.
  - Viết `check_analysis.py`: 7 kiểm dương, 5 kiểm âm. Kiểm quan trọng: nạp lại checkpoint Q1 smoke, tái tạo attention đã lưu và chứng minh q' của phân tích đúng là prior model đã áp (sai số 1e-5).
  - Notebook Kaggle: `RUN_NAME` cố định `q0_q1_full` (trước đây gắn ngày, phiên bị ngắt chạy lại khác ngày sẽ tạo run mới thay vì resume); thêm `check_analysis` vào bước test và `analyze_q0_q1` sau reload; pin `COMMIT` sang commit mới.
- File: mới `training/contribution_study/{analyze_q0_q1,check_analysis}.py`; sửa `kaggle_q0_q1.ipynb`, README training, `results/contribution_study/m0_locked.json` (thêm addendum), `docs/research/contribution-study.md`, báo cáo này, `docs/discussions/vinhduc.md`.
- Lệnh kiểm tra và PASS/FAIL/NOT_RUN:
  - `.venv/Scripts/python -m training.contribution_study.check_quality_pooling` PASS.
  - `.venv/Scripts/python -m training.contribution_study.check_review_fixes` PASS 50/50.
  - `.venv/Scripts/python -m training.contribution_study.check_analysis` PASS 12/12.
  - `code_hash()`/`make_config("full")` khớp `m0_locked.json`: PASS.
  - `python scripts/check_repository.py` PASS.
  - Smoke thật / full / phân tích thật: `NOT_RUN` (thiếu input).
- Environment: `.venv` Python 3.12.10, torch 2.10.0+cpu, numpy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2; Windows 11, CPU 16 luồng, không GPU. Synthetic seed 11 ở `.runtime/` ngoài Git, không phải kết quả.
- Kết quả tốt/âm/FN-FP: chưa có run thật.
- Blocker / main cần quyết định: `DUC-20261117-001` (cập nhật 18/11): bốn input; cho phép full CPU nếu không có GPU; checkpoint ~41 MB Đức sẽ commit trực tiếp theo checklist (file ~1,7 MB, dưới giới hạn GitHub).
- Bước tiếp: có input → chạy notebook (Kaggle) hoặc lệnh README trên máy → tải run về `results/contribution_study/runs/q0_q1_full/`, chạy lại `evaluate_models` + `analyze_q0_q1` trên máy, commit model/kết quả, viết kết luận DUC-04, mở PR. Mốc tự đặt: chưa có input tới 22/11 thì chuẩn bị bàn giao `NOT_RUN` đúng hạn 25/11.

## Bàn giao review

- Checklist nhiệm vụ: chưa hoàn tất.
- Lệnh tái lập / thời gian / tài nguyên đã dùng: chưa có.
- Kết quả Tùng (commit/run/PR): chưa có, không điền số thay bạn.
- Khác recipe/giới hạn/đóng góp đã hoặc chưa chứng minh: chưa có phân tích mới.
- PR về main: chưa mở; sau push main có thể đọc `git show origin/vinhduc:docs/reports/vinhduc.md` hoặc PR.
