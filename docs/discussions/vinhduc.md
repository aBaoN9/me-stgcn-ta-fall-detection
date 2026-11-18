# Trao đổi nhóm - Đức

Nhánh gửi: `vinhduc`. Người nhận mặc định: người điều phối (Gia Bảo). [Quy trình](README.md), [trả lời trên main](answers.md).

## Snapshot hiện tại

- Cập nhật: 2026-11-18 UTC+7 (bổ sung `DUC-20261117-001`).
- Đã đọc `origin/main:docs/discussions/answers.md` tại `8ab7b29`: `GB-20261116-002` (Đức tự chủ M0/smoke/full), `GB-20261116-001` (hạn 25/11/2026: model đã train + kết quả trên GitHub). Đã merge `origin/main` vào `vinhduc` (không xung đột).
- **Xác nhận đã nhận `GB-20261116-001` và `GB-20261116-002`, hạn thứ Tư 25/11/2026 (UTC+7).** Đức đã tự ghi/khóa M0 (`M0_DOCUMENTED`) và chuẩn bị đủ đường chạy; quyền tự chạy không đồng nghĩa code/kết quả đã nghiệm thu.
- Đã fetch `origin/main` tại `af9e7fb` (18/11): chưa có trả lời cho `DUC-20261117-001`.
- Trạng thái câu hỏi: `DUC-20261117-001` `WAITING_FOR_MAIN` (blocker dữ liệu, **ảnh hưởng hạn 25/11**). `DUC-20261028-001` còn mở phần input/GPU, chuyển theo dõi sang `DUC-20261117-001`. `DUC-20261103-001` đã được trả lời bởi `GB-20261116-002`.
- Giữ mọi câu chưa giải quyết khi cập nhật file; không đổi ID câu đang mở.

## DUC-20261117-001 - Cần nguồn bốn input đúng hash để chạy full trước hạn 25/11

- Thời gian: 2026-11-17 06:30 UTC+7.
- Người nhận / trạng thái: Gia Bảo / `WAITING_FOR_MAIN`. **Khẩn**: còn 8 ngày tới hạn `GB-20261116-001`.
- Nhiệm vụ liên quan: DUC-03 (full Q0/Q1), sản phẩm model + kết quả.
- Ngữ cảnh:
  - M0 đã khóa: `results/contribution_study/m0_locked.json`, code hash `6ba0420b4b2a…`, 24 outer / 96 fits, không search.
  - Đã chuẩn bị notebook `training/contribution_study/kaggle_q0_q1.ipynb`. Notebook clone repo public, kiểm code/config khớp M0, guard bốn hash input của 07, rồi chạy validate → phân bố q → test → smoke thật → full → audit → nạp lại checkpoint → zip.
  - Checkpoint lưu ở `<run>/models/`. Ước tính 24 file, khoảng 1,7 MB mỗi file (422.871 tham số float32), tổng khoảng 41 MB. Theo checklist "artifact nhỏ phù hợp Git thì commit trực tiếp", Đức định `git add -f` đích danh từng file `.pt` trong `results/contribution_study/<run>/models/` kèm `manifest.json` (SHA-256). Máy Đức không có `gh` CLI nên chưa tạo được Release asset.
- Đã thử: tìm `gmdcsa24_*` trên ổ C: và D: của máy Đức, không thấy. Máy không có GPU NVIDIA và không có cấu hình Kaggle API. Không tự tải lại video Zenodo để dựng lại input, vì MediaPipe khác phiên bản sẽ ra hash khác, guard sẽ chặn và không còn là cùng dữ liệu với mốc 07.
- Cần nhóm quyết định:
  1. **Cấp quyền truy cập bốn input đúng hash** của notebook 01/07. Ví dụ: share Kaggle Dataset (private) cho tài khoản Kaggle của Đức, hoặc link tải có quyền. Đức chạy trên GPU Kaggle miễn phí, không tiêu tiền.
  2. Nếu tới 22/11 vẫn không có GPU: có cho chạy full trên CPU không? Hiện `allow_cpu_full=false` theo 07; chạy CPU là đổi thiết bị so với mốc nên Đức không tự bật. Nếu cho phép, cả Q0 và Q1 cùng chạy trên CPU và báo cáo ghi rõ thiết bị.
  3. Xác nhận commit trực tiếp khoảng 41 MB checkpoint vào `vinhduc`, hay người điều phối tạo Release asset.
- Đề xuất: (1) share Kaggle Dataset trong hôm nay hoặc ngày mai. Đức chạy smoke thật ngay, full trong vòng 1 ngày, push model + kết quả trước 25/11.
- Ảnh hưởng nếu chờ: smoke thật/full/model `NOT_RUN`. Nếu không có input trước hạn, Đức nộp đúng hạn phần code/test/M0 và ghi rõ phần thiếu là `NOT_RUN`, không tạo số giả.
- Không có token, dữ liệu riêng hoặc log bí mật trong nội dung này.

### Bổ sung 2026-11-18 UTC+7

- Vẫn thiếu bốn input; đây là blocker duy nhất còn chặn smoke thật/full. **Đề xuất hạn cấp input: 20/11** để còn thời gian full, kiểm và phân tích trước 25/11.
- **GPU không còn là blocker cứng:** đo trên CPU máy Đức (16 luồng, dữ liệu synthetic, chỉ đo thời gian): 3,85 s/epoch trên 120 clip, xấu nhất ≈ 3,9 giờ cho 96 fits. Câu 2 đổi thành: **cho phép chạy full trên CPU (cả Q0 và Q1 cùng thiết bị, ghi rõ trong báo cáo) nếu input có trên máy mà không có GPU?** M0 vẫn `allow_cpu_full=false`, Đức không tự bật.
- Câu 3 (checkpoint): Đức tự quyết theo checklist `GB-20261116-001` mục 4 ("artifact nhỏ phù hợp Git thì commit trực tiếp"): 24 file ~1,7 MB, tổng ~41 MB, `git add -f` đích danh từng file + `models/manifest.json` SHA-256. Người điều phối chỉ cần trả lời nếu muốn Release asset thay thế.
- **Vì sao Đức không tự dựng lại từ video GMDCSA24 (GitHub `v2.1` MIT hoặc Zenodo 13354453):** notebook 01 ghi `"path": str(video_path)` (đường dẫn `/kaggle/input/<tên dataset>/...`) vào `gmdcsa24_pose_manifest.csv`, nên hash manifest phụ thuộc tên dataset Kaggle gốc; pose phụ thuộc phiên bản MediaPipe không được ghi và model URL `latest`. Dựng lại sẽ là **phiên bản dữ liệu mới**, guard chặn đúng; dùng nó là đổi input chung của cả Đức và Tùng, ngoài phạm vi Đức. (Đã kiểm: `np.savez_compressed` tự nó tái lập byte, nên vấn đề nằm ở nội dung, không ở định dạng.)
- **Kiểm trên video GMDCSA24 GitHub (commit `5abac76`, MIT) Đức đã tải, 18/11:** đủ 160 video; `video_id`/nhãn/subject khớp 160/160 với `results/component_ablation/predictions.csv`; thứ tự `sample_index` khớp 07. Dựng lại `gmdcsa24_cross_subject_folds.csv` theo code notebook 01 cho **SHA-256 trùng khớp `774f0ac2…`** (file gốc ghi dòng CRLF, tức đã qua máy Windows). Vậy video, nhãn, thứ tự và split giống hệt mốc; ba file còn lại (`pose_32.npz`, `features_32.npz`, `pose_manifest.csv`) phụ thuộc MediaPipe/đường dẫn Kaggle nên không tái lập được. Đã quét theo hash 2.574 file và 17 zip trong thư mục người dùng của Đức: không có bản gốc.
- **Cách cấp nhanh nhất (khoảng 5 phút, phía Gia Bảo):** mở notebook 01 đã chạy trên Kaggle → phiên bản đã lưu → tab **Output** → **New Dataset** (Private) từ bốn file `gmdcsa24_pose_32.npz`, `gmdcsa24_features_32.npz`, `gmdcsa24_pose_manifest.csv`, `gmdcsa24_cross_subject_folds.csv` → **Share** cho tài khoản Kaggle của Đức (gửi riêng, không ghi vào repo). Trước khi share, kiểm `sha256sum` khớp `results/component_ablation/locked_protocol.json`. Đức gắn dataset vào `kaggle_q0_q1.ipynb`, notebook tự kiểm hash.
- **Nếu bản gốc đã mất:** cần Gia Bảo quyết định có dùng bản dựng lại làm input mới cho cả hai hướng không. Khi đó Q0 và Q1 vẫn chạy lại cùng dữ liệu nên so sánh cặp hợp lệ, nhưng mất liên kết với số 07 lịch sử; phải ghi addendum M0 với hash mới trước khi chạy.
- Đã chuẩn bị thêm trong lúc chờ (không cần input): phân tích DUC-04 khóa trước kết quả thật (addendum M0 18/11, `check_analysis` PASS 12/12), notebook Kaggle resume đúng run khi phiên bị ngắt. Chi tiết trong `docs/reports/vinhduc.md`.

## DUC-20261103-001 - M0 theo phân công mới: quality-aware pooling, mở rộng phạm vi kiến trúc

- Thời gian: 2026-11-03 13:09 UTC+7. Cập nhật: 2026-11-17 06:30 UTC+7.
- Trạng thái: **đã được trả lời** bởi `GB-20261116-002` (được tự thay đổi thành phần kiến trúc, gồm quality-aware pooling; không chờ duyệt M0).
- Việc sau trả lời: M0 tự khóa ngày 17/11, `M0_DOCUMENTED` trong `results/contribution_study/m0_locked.json` và `docs/research/contribution-study.md`. Công thức giữ nguyên: `w = softmax(a + log(0,1 + q'))`, ε = 0,1, Q0 vs Q1, 24 outer / 96 fits. Không phải nghiệm thu code/kết quả.

## DUC-20261028-001 - Bản sửa DUC-REV-01..05; còn chờ bốn input và GPU

- Thời gian gửi đầu: 2026-10-28 19:01 UTC+7. Cập nhật: 2026-11-17 06:30 UTC+7.
- Trả lời đã đọc: `CHANGES_REQUESTED` (29/10), rồi `GB-20261116-002` bỏ điều kiện review lại trước full. Danh sách DUC-REV vẫn giữ, chưa được kiểm độc lập.
- Trạng thái: phần review lại không còn là cổng chặn. **Phần input/GPU còn mở**, theo dõi tiếp ở `DUC-20261117-001`. Không gắn `RESOLVED`.
- Bản sửa DUC-REV-01..05 (03/11) vẫn PASS khi kiểm lại 17/11, sau merge main và sau khi thêm kiểm tra nạp lại model: `check_review_fixes` PASS 50/50 (44 cũ + 6 reload), `check_quality_pooling` PASS 13/13, audit 07/08 PASS, `scripts/check_repository.py` PASS. Chi tiết trong `docs/reports/vinhduc.md`.
