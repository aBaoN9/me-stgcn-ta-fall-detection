# Quy tắc làm việc cho AI và thành viên

Repo này là nghiên cứu Fall/ADL, không chỉ demo web. Đọc trước khi sửa; áp dụng toàn repo.

## Xác định người và đọc đúng nhiệm vụ

- Chưa xác định người trong cuộc trò chuyện: **hỏi "Bạn là Tùng, Đức hay người điều phối dự án?" trước khi sửa/chạy train**. Không đoán từ tài khoản Git/máy/nhánh.
- Tùng: nhánh `dinhtung`, [nhiệm vụ](docs/tasks/dinhtung.md), [báo cáo](docs/reports/dinhtung.md).
- Đức: nhánh `vinhduc`, [nhiệm vụ](docs/tasks/vinhduc.md), [báo cáo](docs/reports/vinhduc.md).
- Người điều phối: `main`, [bảng việc](docs/tasks/README.md). Người khác cần người điều phối giao việc, không tự nhận phần của hai bạn.
- Vai trò đã rõ thì không hỏi lại mỗi lượt. Sau khi đọc, tóm tắt hiện trạng, nhiệm vụ, phạm vi sửa, tiêu chí xong và bước đầu tiên.
- Thứ tự đọc: [README](README.md), [bàn giao](docs/handoff.md), [bảng việc](docs/tasks/README.md), nhiệm vụ cá nhân, [protocol](docs/research-protocol.md), [phương pháp](docs/methodology.md), [ablation](docs/component-ablation.md), [timing](docs/timing-ablation.md), rồi code/kết quả liên quan. Không bắt người dùng đọc hết thay AI.

<!-- CODEGRAPH_START -->
## CodeGraph

Nếu repo có `.codegraph/`, dùng `codegraph_explore` hoặc `codegraph explore` trước grep/find/đọc code để định vị/hiểu mã. Source trả về coi như đã đọc. Repo clone không có index thì bỏ qua, không tự index.
<!-- CODEGRAPH_END -->

## Git và ranh giới

- Trước khi làm: `git status --short`, `git branch --show-current`, `git remote -v`. Thay đổi không rõ nguồn gốc: hỏi trước; không reset/clean/stash/ghi đè hộ người khác.
- Thành viên phải ở đúng nhánh trước khi sửa; chưa chuyển an toàn được thì dừng sửa. Người điều phối được sửa main để bàn giao/tích hợp.
- Chỉ ghi vùng trong nhiệm vụ; nguồn cũ được đọc/import, không sửa loop train chung để tiện riêng mình.
- Thành viên không push trực tiếp main, force-push, tự merge hoặc thay model/threshold triển khai. Người điều phối duyệt tích hợp; publish main chỉ khi chủ repo yêu cầu.
- Không tự commit/push khi người dùng chưa yêu cầu. Khi được yêu cầu, cập nhật báo cáo, kiểm diff; không đưa token, video/tensor riêng, môi trường hoặc checkpoint lớn vào Git.
- Mở PR về main; báo cáo trên nhánh được đọc qua PR/fetch, không tự xuất hiện trên main.

## Liêm chính nghiên cứu

- 160 video, 4 subject; không chia ngẫu nhiên frame/clip để test, không gọi ba seed là 480 mẫu độc lập.
- Epoch/hyperparameter/cost/ngưỡng chỉ chọn bằng inner validation theo subject, không dùng outer-test hoặc clip lỗi đã biết.
- Kết quả 06/07/08 không phải kết quả code mới; không ghi đè bằng chứng cũ trong results.
- Không mặc định backbone mới hơn là tốt hơn; không lấy số paper trên dataset khác làm đối chứng trực tiếp. Ghi adaptation/khác biệt paper.
- Không hứa accuracy 95%, chọn seed đẹp, giấu FN/FP, đổi nhãn vì model sai hoặc dùng smoke như full.
- Chưa chạy: `NOT_RUN`; thiếu input/GPU: báo lý do/hỗ trợ cần thiết. Không tạo số liệu giả.
- Khóa phương pháp trước held-out. Outer folds cũ đã được xem trong phát triển; phải công bố giới hạn.

## Kiểm tra và bàn giao

- Kiểm tra nhẹ trước; full train cần protocol duyệt, dữ liệu/môi trường phù hợp. Không tự cài lại CUDA/PyTorch hệ thống, tiêu tiền dịch vụ hoặc tải dataset không rõ quyền.
- Chạy `python scripts/check_repository.py` trước bàn giao. Checker khóa 9 notebook cũ; notebook mới để trong training riêng, không sửa checker để bỏ lỗi.
- Code mới cần check shape/mask/split leakage/OOF coverage/tái tính metric. Ghi lệnh, phiên bản, hash input-code-config, seed, thiết bị, PASS/FAIL/NOT_RUN.
- Cập nhật báo cáo cá nhân mỗi phiên: đã làm, file/commit/run/PR, bằng chứng, kiểm tra, lỗi, còn thiếu, việc main cần quyết định.
- Chỉ `READY_FOR_REVIEW` khi đủ tiêu chí. Kết quả âm hợp lệ nếu đủ thí nghiệm, không phải lý do bỏ thực nghiệm.
