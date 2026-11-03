# Báo cáo công việc - Đức

Người thực hiện: Đức. Nhánh: `vinhduc`. [Nhiệm vụ](../tasks/vinhduc.md).

## Snapshot

- Trạng thái: `IN_PROGRESS`.
- Cập nhật: 2026-11-03 13:09 UTC+7. Đã nhận phân công GB-20261030-001. Đã sửa DUC-REV-01..05, test hồi quy PASS 44/44. M0 giữ quality-aware và gửi xin mở rộng phạm vi kiến trúc.
- Commit code / run ID / PR:
  - `1cbb778` là code DUC-01..03 cũ (đã review).
  - Merge `origin/main` vào `vinhduc` cục bộ: `31af3cb`, chưa push.
  - Bản sửa DUC-REV chưa commit, chờ Đức yêu cầu.
  - Chưa có run thật, chưa mở PR.
- M0 duyệt bởi / ngày / config: chưa duyệt. Gửi `DUC-20261103-001` (`WAITING_FOR_MAIN`); review `DUC-20261028-001` vẫn `CHANGES_REQUESTED` đến khi main xem lại; ε = 0,1.
- Mốc:
  - DUC-01 `READY_FOR_REVIEW` (failure mode chưa xác minh bằng video).
  - DUC-02 `WAITING_FOR_MAIN`.
  - DUC-03 `BLOCKED`: chờ M0, bốn input, GPU.
  - DUC-04 `WAITING_FOR_TUNG` cho phần đối chiếu.
  - WEB-01 `CANCELLED` theo GB-20261030-001, không làm.

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

## Bàn giao review

- Checklist nhiệm vụ: chưa hoàn tất.
- Lệnh tái lập / thời gian / tài nguyên đã dùng: chưa có.
- Kết quả Tùng (commit/run/PR): chưa có, không điền số thay bạn.
- Khác recipe/giới hạn/đóng góp đã hoặc chưa chứng minh: chưa có phân tích mới.
- PR về main: chưa mở; sau push main có thể đọc `git show origin/vinhduc:docs/reports/vinhduc.md` hoặc PR.
