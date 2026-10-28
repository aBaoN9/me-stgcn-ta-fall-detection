# Trao đổi nhóm - Đức

Nhánh gửi: `vinhduc`. Người nhận mặc định: người điều phối. [Quy trình](README.md), [trả lời trên main](answers.md).

## Snapshot hiện tại

- Cập nhật: 2026-10-28 19:01 UTC+7.
- Trạng thái: 1 câu `WAITING_FOR_MAIN` (`DUC-20261028-001`).
- Đã đọc `origin/main:docs/discussions/answers.md` tại `1ab8164`: chưa có quyết định cho Đức.
- Giữ mọi câu chưa giải quyết khi cập nhật file; không đổi ID câu đang mở.

## DUC-20261028-001 - Duyệt M0 quality-aware pooling, cấp bốn input và GPU

- Thời gian: 2026-10-28 19:01 UTC+7.
- Người nhận / trạng thái: người điều phối / `WAITING_FOR_MAIN`.
- Nhiệm vụ liên quan: DUC-02 (M0), DUC-03 (Q0/Q1). Thiết kế đầy đủ: `docs/research/contribution-study.md` mục "DUC-02 - Đề xuất M0" trên nhánh `vinhduc`.
- Ngữ cảnh:
  - Công thức đề xuất, khóa trước train: `q_t` = trung bình 33 khớp của kênh 2 (visibility MediaPipe, đã 0 tại frame `masks=False` và khớp bị dropout), đọc từ input thô.
  - `q'` = avg-pool kernel 3, stride 2, padding 1 (bỏ padding), ánh xạ 32 → 16 bước attention, tâm `2t'`.
  - `w = softmax(a + log(0,1 + q'))`, ε = 0,1 cố định, không search.
  - All-missing thì trọng số về đúng attention gốc, không NaN. Không thêm tham số. Q0 = A3 multiplier 1,0, recipe 07, chạy lại trong runner mới. Q1 chỉ khác pooling.
- Đã thử:
  - DUC-01 audit OOF 07/08 PASS 22/22.
  - Code `training/contribution_study/{quality_pooling,runner,check_quality_pooling,benchmark_latency}.py`. Kiểm phần mềm trên tensor/dữ liệu giả PASS 13/13 (parity prior tắt = `SkeletonModel`, all-missing, q theo augmentation, gradient, smoke runner 8 outer / 32 fits).
  - Latency model-only CPU đã đo.
  - Không có thông tin bí mật trong log.
- Blocker:
  - Bốn input (`gmdcsa24_pose_32.npz`, `gmdcsa24_features_32.npz`, `gmdcsa24_pose_manifest.csv`, `gmdcsa24_cross_subject_folds.csv`) không có trên máy Đức; `data/local/` trống.
  - Máy không có GPU NVIDIA. Full 96 fits cần Kaggle GPU như 07.
- Cần nhóm quyết định:
  1. `APPROVED`/`CHANGES_REQUESTED` cho công thức q, ε = 0,1, ánh xạ stride và cặp Q0/Q1 như trên.
  2. Cấp nguồn bốn input có quyền sử dụng, đúng SHA-256 trong `results/component_ablation/locked_protocol.json` (ví dụ Kaggle dataset/output notebook 01 của nhóm).
  3. Xác nhận chạy full trên Kaggle GPU của nhóm/Đức.
- Đề xuất:
  - Duyệt nguyên công thức. Sau khi có input, chạy `validate_pipeline` rồi smoke thật, và báo phân bố q (số clip có frame thiếu, mean/min q) **trước** full.
  - Nếu q gần hằng trên đa số clip, vẫn chạy và báo kết quả, không đổi công thức sau khi xem outer.
  - Cost 2,25 cho Q1 và robustness nhiễu: chưa đề xuất.
- Ảnh hưởng nếu chờ: DUC-03 `NOT_RUN`. Trong lúc chờ, Đức chỉ làm việc độc lập: review DUC-01, chuẩn bị notebook Kaggle trong `training/contribution_study/`, WEB-01 nếu còn thời gian.
- Trả lời đã đọc (ID/commit main) / việc sau trả lời: chưa có.
