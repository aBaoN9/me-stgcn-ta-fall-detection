# Trao đổi nhóm - Tùng

Nhánh gửi: `dinhtung`. Người nhận mặc định: người điều phối. [Quy trình](README.md), [trả lời trên main](answers.md).

## Snapshot hiện tại

- Cập nhật 19/11/2026. Tùng đã nhận quyết định `GB-20261118-001` trên `origin/main` tại `af9e7fb`.
- `TUNG-20261117-001`: `ANSWERED`. Gia Bảo cấp quyền `APPROVED_TO_PROCEED / AUTONOMOUS`. M0: `M0_DOCUMENTED`.
- TCN_v2_small và A3 chuẩn/07 đã tích hợp trong chương trình M1. Kiểm thử phần mềm trên dữ liệu tổng hợp PASS. Bốn tệp chuẩn đã nhận và xác minh. Smoke thật CPU PASS 8 lần fit, reload/coverage PASS. Smoke T4 và M2 NOT_RUN.

## TUNG-20261117-001 — TCN đối chứng ME-STGCN-TA

- Đề xuất gốc ngày 17/11/2026 trong commit `d65e59c`. [Bản M0](../research/backbone-benchmark.md).
- Tùng xác nhận dùng TCN_v2_small và chạy lại A3 chuẩn/07, cùng feature v2, Nested LOSO theo người và ba seed. Kế hoạch M2 có 96 lần fit.
- Quyền triển khai và lựa chọn thiết bị đã được cấp. Đã nhận GB-20261119-001 và GB-20261119-002 từ main e16ee1a. Bốn tệp chuẩn ở data/gmdcsa24 đã khớp SHA-256, validation và kiểm thử M1 đạt. Xem báo cáo tiến độ về smoke CPU và bước xác minh trên Kaggle T4.
- Bàn giao M1 và báo cáo kiểm tra trên nhánh dinhtung theo yêu cầu Tùng. Hạn sản phẩm 25/11/2026 UTC+7.

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

## Xác nhận GB-20261124-001

- Thời điểm xác nhận: 2026-11-27 12:34:25 +07:00 (UTC+7).
- Tùng đã nhận năm mô hình và A3 chạy lại theo hợp đồng tại docs/research/modern-model-benchmark.md, commit 6985ed7d96498793d1b6def75404bd4fad785ea5.
- Trạng thái: IN_PROGRESS. Đã đồng bộ nhánh dinhtung; đang kiểm nguồn, giấy phép và môi trường. Full mới: 0 lượt tại thời điểm xác nhận.
- Hạn 25/11 chỉ áp dụng nhiệm vụ cũ. Đợt mới chưa có hạn riêng.
