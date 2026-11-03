# Quy tắc làm việc cho AI và thành viên

Repo này tập trung nghiên cứu Fall/ADL. Đọc trước khi sửa; áp dụng toàn repo.

## Phân công hiện hành - Gia Bảo quyết định ngày 30/10/2026

- Tùng: giữ dataset, tự đề xuất mô hình hoặc phương pháp khác để xử lý bài toán và so sánh với ME-STGCN-TA. Không bắt buộc CTR-GCN hoặc chỉ được đổi backbone.
- Đức: lấy ME-STGCN-TA hiện tại làm mốc, tự đề xuất thay đổi có kiểm soát và so sánh với bản gốc. Không thay bằng mô hình khác như nhiệm vụ Tùng, không bắt buộc quality-aware pooling.
- Mốc Đức đã chốt: `A3_motion_attention` / policy `fixed` của notebook 07 và `training/component_ablation/`, tám kênh, attention bật, Fall multiplier 1.0; xem [nhận diện mốc](docs/tasks/vinhduc.md). Không lấy checkpoint final 06, A3_cost_225, T0/T1 hoặc Q1 thay mốc; chạy lại cặp gốc/thay đổi để đánh giá như nhau.
- Giao câu hỏi nghiên cứu và tiêu chí đối chiếu, không áp sẵn mô hình, công thức, siêu tham số hoặc ma trận thí nghiệm. Thành viên giải thích phần giữ nguyên/thay đổi, lý do và giới hạn; nộp phương án M0 để thống nhất trước full train.
- Hai hướng giữ cùng dataset, nhãn, split theo subject và giao thức đánh giá trong [protocol](docs/research-protocol.md). Cùng quy tắc chọn cấu hình/ngưỡng, seed, ngân sách tương đương và cách tính chỉ số; không dùng outer-test để chọn phương án.
- PoseLab đã dừng theo yêu cầu chủ repo. Hủy WEB-01/WEB-02, không tự làm UI/QA web hoặc bật lại app. Giữ nguồn cũ, model và kết quả để truy vết, không coi việc dừng app là xóa bằng chứng.
- Quyết định GB-20261030-001 trong [answers](docs/discussions/answers.md) thay phạm vi giao việc cũ, không tự cấp phê duyệt full train. Phát hiện kiểm tra dữ liệu/nghiệm thu cũ còn giá trị; CTR-GCN/B0-B3/quality-aware/Q0-Q1 không còn là lời giải bắt buộc.

## Xác định người và đọc đúng nhiệm vụ

- Chưa xác định người trong cuộc trò chuyện: **hỏi "Bạn là Gia Bảo (người điều phối), Tùng hay Đức?" trước khi sửa/chạy train**. Không đoán từ tài khoản Git/máy/nhánh.
- Tùng: nhánh `dinhtung`, [nhiệm vụ](docs/tasks/dinhtung.md), [báo cáo](docs/reports/dinhtung.md).
- Đức: nhánh `vinhduc`, [nhiệm vụ](docs/tasks/vinhduc.md), [báo cáo](docs/reports/vinhduc.md).
- Gia Bảo là chủ repo/người điều phối: phát triển kỹ thuật trên `giabao`, [nhiệm vụ](docs/tasks/giabao.md), [báo cáo](docs/reports/giabao.md). `main` giữ bản chuẩn đã duyệt và quyết định chung trong `docs/discussions/answers.md`; không phải nhánh thử nghiệm cá nhân. Người khác cần người điều phối giao việc, không tự nhận phần của hai bạn.
- Câu hỏi danh tính có thể hỏi ngay trong chat. Sau khi biết người, các câu hỏi cho nhóm/người điều phối phải ghi vào [kênh Markdown](docs/discussions/README.md), không chỉ nhắn trong chat rồi coi nhóm đã nhận.
- Vai trò đã rõ thì không hỏi lại mỗi lượt. Sau khi đọc, tóm tắt hiện trạng, nhiệm vụ, phạm vi sửa, tiêu chí xong và bước đầu tiên.
- Thứ tự đọc: [README](README.md), [bàn giao](docs/handoff.md), [bảng việc](docs/tasks/README.md), nhiệm vụ cá nhân, [protocol](docs/research-protocol.md), [phương pháp](docs/methodology.md), [ablation](docs/component-ablation.md), [timing](docs/timing-ablation.md), rồi code/kết quả liên quan. Không bắt người dùng đọc hết thay AI.

<!-- CODEGRAPH_START -->
## CodeGraph

Nếu repo có `.codegraph/`, dùng `codegraph_explore` hoặc `codegraph explore` trước grep/find/đọc code để định vị/hiểu mã. Source trả về coi như đã đọc. Repo clone không có index thì bỏ qua, không tự index.
<!-- CODEGRAPH_END -->

## Git và ranh giới

- Trước khi làm: `git status --short`, `git branch --show-current`, `git remote -v`. Thay đổi không rõ nguồn gốc: hỏi trước; không reset/clean/stash/ghi đè hộ người khác.
- Thành viên phải ở đúng nhánh trước khi sửa; chưa chuyển an toàn được thì dừng sửa. Gia Bảo sửa code/thử nghiệm trên `giabao`, chỉ tích hợp vào main sau review/kiểm tra và yêu cầu của chủ repo. Tài liệu bàn giao/quy tắc/quyết định điều phối được cập nhật trực tiếp trên main; publish vẫn cần yêu cầu của chủ repo.
- AI làm cùng Gia Bảo được đọc/fetch nhánh Tùng/Đức để review, không cần merge vào giabao chỉ để xem. Không tự sửa hoặc push hai nhánh cá nhân đó; giabao không tự đồng nghĩa code đã được duyệt vào main.
- Chỉ ghi vùng trong nhiệm vụ; nguồn cũ được đọc/import, không sửa loop train chung để tiện riêng mình.
- Thành viên không push trực tiếp main, force-push, tự merge hoặc thay model/threshold triển khai. Người điều phối duyệt tích hợp; publish main chỉ khi chủ repo yêu cầu.
- Không tự commit/push code khi người dùng chưa yêu cầu. **Ngoại lệ được chủ repo cho phép sẵn:** mỗi lần hỏi nhóm, cập nhật/ghi đè file trao đổi cá nhân hiện tại, tạo commit thường và push nhánh được giao, không phải hỏi lại quyền cho lần gửi Markdown đó. Chỉ stage file trao đổi; kiểm cả commit chưa push để không gửi kèm code ngoài yêu cầu. Không force-push, không commit hộ thay đổi khác. Không đưa token, video/tensor riêng, môi trường hoặc checkpoint lớn vào Git.
- Mở PR về main; báo cáo trên nhánh được đọc qua PR/fetch, không tự xuất hiện trên main.

## Trao đổi nhóm qua Markdown

- Tùng dùng `docs/discussions/dinhtung.md` trên `dinhtung`; Đức dùng `docs/discussions/vinhduc.md` trên `vinhduc`. Đọc [quy trình](docs/discussions/README.md) và giữ toàn bộ câu chưa giải quyết trong bản hiện tại, không xóa mất câu khác khi ghi đè.
- Câu hỏi phải có ID, thời gian UTC+7, ngữ cảnh/file/log, đã thử gì, lựa chọn đề xuất và quyết định cần nhóm trả lời. Không ghi mật khẩu, token, dữ liệu riêng hoặc log có bí mật.
- Mỗi lần gửi/cập nhật câu hỏi: commit/push bình thường và kiểm push thành công. Không có quyền/mạng/auth thì báo lỗi, không nói đã gửi; không vượt cơ chế phê duyệt của công cụ.
- Người điều phối trả lời trong `docs/discussions/answers.md` trên main, dẫn đúng ID. Đầu mỗi phiên và trước việc đang chờ duyệt, fetch và đọc file trả lời; không tự xem im lặng là đồng ý.
- Chưa trả lời thì đánh dấu `WAITING_FOR_MAIN`, tiếp tục việc độc lập. Thiếu phê duyệt M0/phạm vi/đổi model thì dừng đúng phần đó, không dừng mọi việc và không tự đoán quyết định.

## PoseLab đã dừng

- [WEB-01/WEB-02](docs/tasks/web-app.md) có trạng thái `CANCELLED`, không phải việc phụ khi chờ GPU. Ngoại lệ cho Đức sửa frontend và Tùng QA web đã bị thu hồi.
- `app/` chỉ giữ làm nguồn lịch sử; launcher và khởi động ASGI bị chặn. Không tự cài thư viện/tải model để chạy app, thay model/threshold hoặc xóa dữ liệu lịch sử.
- Chỉ Gia Bảo được quyết định mở lại; không tự bật lại vì tài liệu hoặc nhánh cũ còn hướng dẫn chạy.

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
