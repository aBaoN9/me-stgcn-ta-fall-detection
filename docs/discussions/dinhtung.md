# Trao đổi nhóm - Tùng

Nhánh gửi: `dinhtung`. Người nhận mặc định: người điều phối. [Quy trình](README.md), [trả lời trên main](answers.md).

## Snapshot hiện tại

- Cập nhật 19/11/2026. Tùng đã nhận quyết định `GB-20261118-001` trên `origin/main` tại `af9e7fb`.
- `TUNG-20261117-001`: `ANSWERED`. Gia Bảo cấp quyền `APPROVED_TO_PROCEED / AUTONOMOUS`. M0: `M0_DOCUMENTED`.
- TCN_v2_small và A3 chuẩn/07 đã tích hợp trong chương trình M1. Kiểm thử phần mềm trên dữ liệu tổng hợp PASS. Smoke thật BLOCKED vì thiếu bốn input chuẩn. M2 NOT_RUN.

## TUNG-20261117-001 — TCN đối chứng ME-STGCN-TA

- Đề xuất gốc ngày 17/11/2026 trong commit `d65e59c`. [Bản M0](../research/backbone-benchmark.md).
- Tùng xác nhận dùng TCN_v2_small và chạy lại A3 chuẩn/07, cùng feature v2, Nested LOSO theo người và ba seed. Kế hoạch M2 có 96 lần fit.
- Quyền triển khai và lựa chọn thiết bị đã được cấp. Hỗ trợ còn cần là nguồn có quyền của bốn input đúng hash 07. Tùng đã xác nhận CUDA hoạt động trên Kaggle và Colab, quyền truy cập dữ liệu nhóm chưa được xác minh.
- Cập nhật M1 đang ở máy local theo yêu cầu Tùng. Hạn sản phẩm 25/11/2026 UTC+7.

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
