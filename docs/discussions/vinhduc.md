# Trao đổi nhóm - Đức

Nhánh gửi: `vinhduc`. Người nhận mặc định: người điều phối (Gia Bảo). [Quy trình](README.md), [trả lời trên main](answers.md).

## Snapshot hiện tại

- Cập nhật: 2026-11-03 13:09 UTC+7.
- Trạng thái: 2 câu `WAITING_FOR_MAIN` (`DUC-20261103-001`, `DUC-20261028-001`).
- Đã đọc `origin/main:docs/discussions/answers.md` tại `f6a23a5`: GB-20261030-001 (phân công mới, WEB-01 hủy), `DUC-20261028-001` `CHANGES_REQUESTED`.
- Giữ mọi câu chưa giải quyết khi cập nhật file; không đổi ID câu đang mở.

## DUC-20261103-001 - M0 theo phân công mới: giữ quality-aware pooling, xin mở rộng phạm vi kiến trúc

- Thời gian: 2026-11-03 13:09 UTC+7.
- Người nhận / trạng thái: Gia Bảo / `WAITING_FOR_MAIN`.
- Nhiệm vụ liên quan: DUC-02 (M0), DUC-03. Thiết kế: `docs/research/contribution-study.md`, mục "M0 theo phân công GB-20261030-001", trên `vinhduc`.
- Ngữ cảnh:
  - GB-20261030-001 cho Đức tự chọn thay đổi trên mốc `A3_motion_attention` / `fixed` (07, `training/component_ablation/` tại `c8f654b`).
  - Đức chọn giữ quality-aware pooling: `w = softmax(a + log(0,1 + q'))`, q từ visibility đã mask/augment, ε = 0,1 cố định, avg-pool kernel 3 / stride 2 / padding 1 bỏ padding. Công thức giữ nguyên như review 29/10, không thêm tham số.
  - Prior sửa forward của temporal attention, nên Đức **không gọi là giữ nguyên kiến trúc** và xin duyệt như mở rộng phạm vi.
- Giữ nguyên: dữ liệu/nhãn/split, bốn input đúng hash, feature v2 tám kênh, augmentation, loss class-balanced, multiplier 1,0, recipe 07, nested LOSO 4×3, seed 42/142/242, policy fixed 0,5, chọn epoch/ngưỡng chỉ bằng inner.
- Cặp so / ngân sách: Q0 = mốc A3 chạy lại trong runner mới (parity với `SkeletonModel` đã kiểm) vs Q1 = Q0 + prior. 24 outer / 96 fits, không search.
- Đã thử: mã `training/component_ablation/` hiện tại giống hệt `c8f654b`. Kiểm phần mềm PASS (xem `DUC-20261028-001`).
- Cần nhóm quyết định:
  1. `APPROVED` / `CHANGES_REQUESTED` cho M0 này như một mở rộng phạm vi kiến trúc có kiểm soát.
  2. Nếu không chấp nhận sửa pooling, chỉ hướng cho phương án thay thế không đụng kiến trúc (xử lý/huấn luyện/quyết định) để Đức soạn M0 mới.
- Đề xuất: duyệt M0, nhưng full vẫn chỉ sau review bản sửa và smoke thật như điều kiện 29/10.
- Ảnh hưởng nếu chờ: DUC-03 `NOT_RUN`; Đức chỉ làm việc độc lập (review/test, notebook Kaggle chuẩn bị). Không làm web.
- Trả lời đã đọc (ID/commit main) / việc sau trả lời: chưa có.

## DUC-20261028-001 - Nộp lại bản sửa DUC-REV-01..05; còn chờ bốn input và GPU

- Thời gian gửi đầu: 2026-10-28 19:01 UTC+7. Cập nhật: 2026-11-03 13:09 UTC+7.
- Người nhận / trạng thái: người điều phối / `WAITING_FOR_MAIN` (review lại bản sửa). Trả lời đã đọc: `CHANGES_REQUESTED` ngày 29/10 trên `origin/main` (`f6a23a5`).
- Phần ép hướng/công thức Q0/Q1 đã được thay bởi GB-20261030-001. Đức vẫn sửa vì tiếp tục dùng code này (xem `DUC-20261103-001`).
- Đã sửa:
  - REV-01: `runner.verify_inputs` dừng trước mọi fit nếu thiếu/thừa input hoặc sai hash so với `results/component_ablation/locked_protocol.json`. Synthetic chỉ được chạy smoke, `scientific_result=false`.
  - REV-02: `validate_fold` dùng cho cả resume và tổng hợp. Kiểm complete, chữ ký run, inner/fits, ba policy, ID/index/subject/label khớp nguồn, score/threshold hữu hạn, prediction khớp threshold, metric tái tính, artifact. Fold hỏng được chạy lại; `completion.json` chỉ ghi khi qua hết.
  - REV-03: `benchmark_latency.make_sync` đồng bộ đúng `cuda`/`cuda:N`.
  - REV-04: `audit_existing.compare` không bỏ NaN; NaN chỉ khớp NaN, khóa trùng/thiếu hàng/inf đều FAIL.
  - REV-05: thêm `audit_existing --run <run>` cho Q0→Q1; báo cáo/README cập nhật.
- Kiểm tra (03/11, `.venv` torch 2.10.0+cpu, CPU):
  - `check_review_fixes` PASS 44/44, gồm ca review tái hiện (8 báo cáo complete=false/fits=0/chỉ fixed/metrics rỗng) nay bị chặn, và resume chạy lại fold mất checkpoint.
  - `check_quality_pooling` PASS 13/13.
  - Audit 07/08 nguyên trạng PASS.
  - `scripts/check_repository.py` PASS.
  - Smoke thật / full / latency GPU `NOT_RUN`.
- Commit chứa bản sửa: xem commit đi kèm lần push này trên `vinhduc`; chi tiết hash file trong `docs/reports/vinhduc.md`.
- Còn blocker từ 28/10: bốn input (`gmdcsa24_pose_32.npz`, `gmdcsa24_features_32.npz`, `gmdcsa24_pose_manifest.csv`, `gmdcsa24_cross_subject_folds.csv`) đúng hash chưa có trên máy Đức; máy không GPU NVIDIA.
- Cần nhóm quyết định:
  1. Review lại bản sửa DUC-REV-01..05.
  2. Cấp nguồn bốn input có quyền sử dụng.
  3. Xác nhận GPU (Kaggle của nhóm/Đức) cho smoke thật rồi full sau khi duyệt.
- Không có token, dữ liệu riêng hoặc log bí mật trong nội dung này.
