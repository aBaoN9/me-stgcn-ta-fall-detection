# Trao đổi nhóm - Tùng

Nhánh gửi: `dinhtung`. Người nhận mặc định: người điều phối. [Quy trình](README.md), [trả lời trên main](answers.md).

## Snapshot hiện tại

- Cập nhật: 2026-11-17 22:35 UTC+7.
- Trạng thái: `OPEN` — **đề xuất chính thức `TUNG-20261117-001`**, Tùng đã kiểm tra và đồng ý commit/push lên `origin/dinhtung` để Gia Bảo xem xét; chưa được Gia Bảo duyệt.
- Đã đọc quyết định trên `origin/main` tại `8ab7b29`: `GB-20261030-001`, `GB-20261116-001`, `GB-20261116-002`. Ghi nhận hạn model đã train + kết quả trên GitHub là 25/11/2026; Tùng vẫn cần duyệt M0 trước full. Nhánh dinhtung đã nhận main 8ab7b29 bằng fast-forward; việc phát hành được xác minh bằng kết quả push/remote, không đồng nghĩa Gia Bảo đã đọc.
- Giữ mọi câu chưa giải quyết khi cập nhật file; không đổi ID câu đang mở.

## TUNG-20261117-001 — Xin duyệt M0: TCN đối chứng ME-STGCN-TA

- Thời gian soạn: 2026-11-17 22:16 UTC+7.
- Người hỏi: Nguyễn Đinh Tùng; người nhận: Nguyễn Gia Bảo.
- Trạng thái: **`OPEN` / đề xuất chính thức được Tùng đồng ý gửi ngày 17/11/2026**. M0: **`M0_PROPOSED — PENDING_APPROVAL`**; chờ quyết định của Gia Bảo.
- Nhiệm vụ: TUNG-01; bản đầy đủ: [M0 — TCN đối chứng](../research/backbone-benchmark.md).
- Đề xuất: **TCN nhỏ (mạng tích chập theo thời gian)** từ thiết kế Bai/Kolter/Koltun, bốn khối 32 kênh ẩn, dilation (giãn bộ lọc) 1/2/4/8, kernel3 (độ rộng3), lấy mốc cuối. Dùng cùng feature v2 (đặc trưng phiên bản2) tám kênh của 07, xếp 33 khớp thành vector mỗi thời điểm; không thêm dữ liệu/weights (trọng số) ngoài.
- Lý do: đã khảo sát TCN/GRU/CTR-GCN; TCN có nguồn chính thức gọn, không phải đổi graph (đồ thị khớp), khác rõ mạng đồ thị có attention (chú ý). Không mặc định TCN tốt hơn; GRU (mạng hồi tiếp có cổng) là dự phòng kỹ thuật cần xác nhận lại khi kích hoạt, không chạy thêm để chọn bằng test.
- Mốc xin xác nhận: **chạy lại A3 chuẩn/07**, multiplier Fall (hệ số phạt lớp ngã)1.0, fixed (ngưỡng cố định)0.5; không dùng checkpoint final06 hoặc ghép số lịch sử làm đối chứng mới. A3 được đề xuất vì cùng feature/recipe rõ, không tự lấy quy định mốc của Đức áp cho Tùng.
- Dữ liệu/đánh giá: cùng 160 video, 79 Fall/81 ADL, bốn người, bốn input đúng hash07; Nested LOSO (giữ một người làm test, chọn cấu hình trong ba người còn lại), seed42/142/242. Accuracy (độ chính xác) fixed0.5 chính, báo đủ độ nhạy/đặc hiệu/FN (bỏ sót)/FP (báo nhầm), từng người/seed và chi phí cùng điều kiện nếu đo. Không chọn bằng outer-test (kiểm tra vòng ngoài).
- Ngân sách đề xuất: một cấu hình mỗi bên; AdamW (cập nhật trọng số), lr0.001 (tốc độ học), batch16 (clip/lượt), tối đa50 epoch (lượt qua tập train), patience10 (dừng sớm); tổng **96 fits (lần train), 24 outer**, trong đó72 inner (vòng lựa chọn) +24 refit (train lại). Khóa trước outer. Đây là so sánh toàn bộ bộ phân loại, không quy cải thiện riêng cho graph/attention/TCN.
- Đã thử/kiểm: đọc code07/config06 và nguồn chính thức, kiểm mốc không khác main; fetch thành công, local dinhtung sạch trước viết. `data/` chỉ có README; `nvidia-smi` có RTX3050 Laptop4GB nhưng chưa kiểm PyTorch CUDA; có Kaggle CLI (lệnh), chưa xác minh tài khoản/quota/GPU. Chưa train hoặc smoke thật; chưa có metric (chỉ số) mới.
- Cần Gia Bảo xác nhận:
  1. Duyệt TCN, A3 chuẩn chạy lại và thiết kế/ngân sách mục8–10 trong M0, cho phép full sau khi kiểm dữ liệu/test/smoke thật đạt.
  2. Cấp nguồn bốn input có quyền và đúng hash; xác nhận GPU Kaggle được dùng. Nếu dùng RTX3050 local thay Kaggle, cần đồng ý thiết bị cho cả cặp sau kiểm runtime/bộ nhớ.
  3. Xác nhận nguyên tắc dự phòng GRU cần báo/duyệt lại trước outer khi TCN vướng kỹ thuật; không chuyển vì score thấp.
  4. Cho biết quyền nộp checkpoint qua GitHub/Release (tệp phát hành); chưa thử push/upload nên chưa xác minh quyền ghi.
- Ảnh hưởng nếu thiếu: M0 có thể review, nhưng validation/smoke thật và full còn NOT_RUN; không hứa kịp hạn khi chưa có input/tài nguyên/duyệt. Nộp phần thực có đúng hạn, không bịa kết quả.
- Trả lời đã đọc: quyết định phân công/hạn tại main `8ab7b29`, **chưa có câu trả lời cho ID này**. Không coi im lặng là đồng ý.
- Bước phát hành đã được Tùng cho phép: commit đúng ba tài liệu M0, push `origin/dinhtung`, kiểm commit trên remote rồi dừng. Sau đó chờ Gia Bảo trả lời đúng ID trên answers.md; không triển khai M1, smoke hoặc full train trong phiên này.

## Mẫu câu hỏi - chưa phải câu hỏi đã gửi

- ID: `TUNG-YYYYMMDD-001`.
- Thời gian: `YYYY-MM-DD HH:mm UTC+7`.
- Người nhận / trạng thái: người điều phối / `OPEN`.
- Nhiệm vụ liên quan / ngữ cảnh / file: điền rõ.
- Đã thử / lệnh / log đã khử bí mật: điền rõ.
- Cần nhóm quyết định điều gì: điền một câu cụ thể.
- Đề xuất / lựa chọn / ảnh hưởng nếu chờ: điền rõ.
- Trả lời đã đọc (ID/commit main) / việc sau trả lời: chưa có.

Mỗi lần hỏi thật, thay snapshot/mẫu bằng nội dung hiện tại, commit và push file này lên `dinhtung` theo quy trình. Không force-push.
