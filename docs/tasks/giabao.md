# Gia Bảo - Phát triển kỹ thuật và điều phối

Nhánh làm việc: `giabao`. Vai trò: chủ repo/người điều phối. [Báo cáo](../reports/giabao.md), [bảng việc](README.md), [bàn giao](../handoff.md).

## Hai vai trò, không nhầm hai nhánh

- `giabao`: sửa code, thử nghiệm theo yêu cầu, chuẩn bị bản tích hợp hoặc chỉnh Word trước review. Chưa có thay đổi nào tự được công nhận chỉ vì nằm trên nhánh chủ repo.
- `main`: bản chuẩn của nhóm sau review/kiểm tra; giữ quy tắc, bàn giao và quyết định chung tại `docs/discussions/answers.md`.
- Tài liệu điều phối có thể sửa trực tiếp trên main. Code/thử nghiệm làm trên giabao; tích hợp và publish main chỉ khi chủ repo yêu cầu. Tạo nhánh không đồng nghĩa đã bật branch protection.

## AI bắt đầu một phiên

1. Vai trò Gia Bảo đã rõ thì không hỏi lại tên. Đọc AGENTS, tài liệu bàn giao, bảng việc, protocol và yêu cầu hiện tại; không tự nhận nhiệm vụ nghiên cứu mới.
2. Kiểm status/nhánh/remote, fetch khi có mạng. Sửa code phải ở giabao; worktree có thay đổi chưa rõ nguồn gốc thì dừng hỏi, không reset/stash/ghi đè để chuyển nhánh.
3. Nếu giabao chưa có local nhưng đã có origin/giabao, dùng `git switch --track origin/giabao`; nếu đã có local dùng `git switch giabao`. Không chạy cả hai máy móc.
4. Đọc main mới và quyết định nhóm trước phần phụ thuộc. Nhận cập nhật bằng merge an toàn khi được yêu cầu, không reset nhánh có bài làm. Không cần đổi nhánh chỉ để đọc file từ remote.

## Review Tùng và Đức

```bash
git fetch origin
git show origin/dinhtung:docs/reports/dinhtung.md
git show origin/vinhduc:docs/reports/vinhduc.md
git show origin/dinhtung:docs/discussions/dinhtung.md
git show origin/vinhduc:docs/discussions/vinhduc.md
git diff origin/main...origin/dinhtung
git diff origin/main...origin/vinhduc
```

- Không cần merge hai nhánh vào giabao chỉ để xem code/câu hỏi. Ghi rõ commit đã review, phát hiện, kiểm tra, phần chấp nhận và phần yêu cầu sửa.
- Quyết định chính thức trả lời đúng ID trong `docs/discussions/answers.md` trên main, publish khi chủ repo yêu cầu và kiểm remote sau push. File local chưa phải thông báo đã gửi.
- Không tự push/sửa nhánh dinhtung/vinhduc hoặc làm thay phần đã giao. Nếu cần đổi phân công/phạm vi, ghi rõ quyết định chung trước khi triển khai.
- Phân công hiện hành ngày 30/10/2026: Tùng giữ dataset và tự đề xuất mô hình/phương pháp khác; Đức thay đổi có kiểm soát ME-STGCN-TA. Giao tiêu chí đánh giá chung, không áp sẵn mô hình/công thức. Duyệt M0 và mốc so sánh trước full; giữ bài làm và phát hiện lỗi đã có nếu tiếp tục dùng mã.
- PoseLab/WEB-01/WEB-02 đã hủy; không còn ngoại lệ UI/QA web. Giữ model, nguồn và bằng chứng cũ; không bật lại app nếu chưa có yêu cầu mới của Gia Bảo.

## Phát triển và tích hợp

- Phạm vi từng phiên theo yêu cầu chủ repo; không tự đổi protocol, train full, model/threshold triển khai, backend hay dữ liệu vì vai trò điều phối.
- Giữ bằng chứng/kết quả cũ; kết quả thử nghiệm mới tách thư mục, ghi input/code/config hash và giới hạn. Giữ quy tắc liêm chính nghiên cứu như hai thành viên.
- Trước tích hợp: review diff, chạy kiểm tra phù hợp và `python scripts/check_repository.py`, xử lý xung đột có chủ đích. Không gom code chưa duyệt từ nhiều nhánh rồi gọi là bản chuẩn.
- Báo cáo riêng trên giabao, có commit, lệnh/check và phần chưa chạy. Commit/push, mở PR hoặc merge theo yêu cầu chủ repo; không tự force-push hoặc nâng trạng thái kết quả.
- AI Tùng/Đức vẫn đọc quyết định từ origin/main, không cần theo dõi origin/giabao để biết được duyệt hay chưa.
