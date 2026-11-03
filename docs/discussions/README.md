# Kênh trao đổi nhóm qua Markdown

Áp dụng từ 28/10/2026. Chủ repo yêu cầu câu hỏi/đề nghị giữa nhóm đi qua file Markdown và push để chủ repo kiểm tra, không chỉ nằm trong chat AI riêng.

| Người | File đang dùng | Nhánh ghi | Ai trả lời |
|---|---|---|---|
| Tùng | [dinhtung.md](dinhtung.md) | `dinhtung` | Người điều phối |
| Đức | [vinhduc.md](vinhduc.md) | `vinhduc` | Người điều phối |
| Gia Bảo (người điều phối) | [answers.md](answers.md) | `main` | Trả lời theo ID câu hỏi |

Gia Bảo phát triển code trên `giabao`, nhưng quyết định chính thức cho nhóm vẫn ở `answers.md` trên main. Không tạo bản trả lời riêng trên giabao hoặc yêu cầu thành viên đọc giabao thay main. Muốn publish quyết định mới khi đang làm ở giabao, chỉ chuyển nhánh khi worktree an toàn; không mang theo code thử nghiệm vào main.

Mỗi người chỉ sửa file gửi của mình. Không tạo một file câu hỏi chung để hai nhánh ghi đè lẫn nhau. File trả lời do người điều phối giữ; nếu cần hỏi bạn còn lại thì ghi người nhận trong câu hỏi để nhóm cùng đọc.

## Thành viên gửi câu hỏi

1. Xác định danh tính/nhánh, kiểm worktree và đọc câu trả lời mới từ main. Câu xác định danh tính được hỏi trong chat; trao đổi với nhóm thì phải ghi file.
2. Cập nhật **cùng file Markdown cá nhân**, không tạo file mới mỗi lần hỏi. Giữ các câu còn mở; câu đã giải quyết có thể bỏ khỏi snapshot sau khi có ID câu trả lời/commit đối chiếu. Lịch sử nằm trong Git.
3. Mỗi câu có ID ổn định, ví dụ `TUNG-20261028-001` hoặc `DUC-20261028-001`; ngày/giờ UTC+7, người nhận, ngữ cảnh, file/lệnh/log đã khử bí mật, điều đã thử, đề xuất và quyết định cần xác nhận. Không đổi ID khi sửa cùng câu.
4. Stage đúng file, tạo commit thường và push nhánh cá nhân. Đây là quyền gửi trao đổi chủ repo đã cho sẵn, không cần xin lại mỗi lần. Không stage code/báo cáo khác nếu chưa được yêu cầu; kiểm outgoing commits để không push kèm code ngoài phạm vi.
5. Kiểm push thành công rồi báo trong chat ngắn gọn ID/nhánh. Nếu lỗi auth/quyền/network, ghi rõ chưa gửi được; không giả vờ nhóm đã nhận.

Ví dụ cho Tùng, chỉ chạy sau khi kiểm nhánh/diff và các commit chưa push:

```bash
git add -- docs/discussions/dinhtung.md
git diff --cached --check
git diff --cached --name-only
git commit -m "Update Tung team questions"
git push origin dinhtung
```

Đức thay tên nhánh/file tương ứng. **Ghi đè là cập nhật nội dung file bằng commit mới, không phải `git push --force`**, không amend/xóa lịch sử. Không có nội dung mới thì không tạo commit rỗng.

## Người điều phối đọc và trả lời

```bash
git fetch origin
git show origin/dinhtung:docs/discussions/dinhtung.md
git show origin/vinhduc:docs/discussions/vinhduc.md
```

Đọc ngay từ nhánh remote, không cần merge code để xem câu hỏi. Người điều phối cập nhật `docs/discussions/answers.md` trên main, dẫn ID, quyết định, phạm vi cho phép, lý do/điều kiện và thời gian; commit/push khi chủ repo yêu cầu trả lời/publish. Chỉ người điều phối ghi file này để tránh xung đột.

Thành viên kiểm mỗi đầu phiên và trước khi làm phần chờ phê duyệt:

```bash
git fetch origin
git show origin/main:docs/discussions/answers.md
```

Cập nhật trạng thái câu hỏi sau khi đã đọc; muốn nhận hướng dẫn/code main thì merge main vào nhánh mình khi worktree sạch, không reset. Không cần merge chỉ để đọc câu trả lời.

## Trạng thái và an toàn

- `OPEN`: câu mới; `WAITING_FOR_MAIN`: đã gửi, chờ quyết định; `ANSWERED`: đã có phản hồi nhưng có thể còn việc; `RESOLVED`: đã xử lý xong; `BLOCKED`: vướng cụ thể.
- Chưa có trả lời không phải được duyệt. Tiếp tục việc độc lập; không chạy full train chưa duyệt, đổi model hoặc sửa ngoài phạm vi.
- Không có thông báo/check tự động được cài. Chủ repo sẽ kiểm thủ công; không hứa thời gian phản hồi hay spam polling.
- Giữ báo cáo công việc trong `docs/reports/`; kênh này là hỏi-đáp/quyết định, không thay báo cáo/bằng chứng thực nghiệm.
