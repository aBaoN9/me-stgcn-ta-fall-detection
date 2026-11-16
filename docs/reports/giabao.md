# Báo cáo công việc - Gia Bảo

Người thực hiện: Gia Bảo. Nhánh phát triển: `giabao`. Vai trò: chủ repo/người điều phối. [Nhiệm vụ](../tasks/giabao.md).

## Snapshot

- Cập nhật: 16/11/2026, kiểm tra tiến độ remote và ra thông báo nhắc Tùng/Đức thực hiện nhiệm vụ; chưa chạy thí nghiệm mới.
- Tùng dùng mô hình/phương pháp khác trên cùng dataset; Đức thay đổi có kiểm soát ME-STGCN-TA. Không áp sẵn mô hình/công thức, cùng giao thức đánh giá và duyệt M0 trước full.
- Quyết định chính thức và các phản hồi cho Tùng/Đức vẫn ở `docs/discussions/answers.md` trên main; báo cáo này không thay kênh phê duyệt.
- Trạng thái tạo/publish nhánh phải đối chiếu Git/remote, không dùng nội dung báo cáo này làm bằng chứng push thành công.

## Nhật ký tiếp theo

Mỗi phiên nối thêm ngày, yêu cầu, phạm vi, file/commit/PR, lệnh kiểm tra và PASS/FAIL/NOT_RUN, giới hạn/blocker và bước tiếp. Khi review nhánh khác ghi đúng commit; khi tích hợp ghi phần được duyệt và bằng chứng, không ghi code chưa review thành DONE.

## 16/11/2026 - Nhắc Tùng và Đức thực hiện nhiệm vụ

- Gia Bảo bổ sung sản phẩm bàn giao: MODEL ĐÃ TRAIN + KẾT QUẢ ĐÁNH GIÁ phải có trên GitHub trước hạn 25/11/2026, không chỉ M0/tiến độ. Đồng bộ AGENTS, bảng việc và thông báo; thêm checklist model/load/OOF/metric/hash, vị trí riêng và quyền nộp trên nhánh cá nhân, Release asset cho artifact lớn. Không upload artifact thay thành viên, không duyệt full hay thay model triển khai; thiếu sản phẩm vẫn chưa hoàn tất và blocker phải báo sớm. Kiểm tra diff/checker trước publish; đối chiếu remote để xác nhận phát hành.
- Cập nhật tiếp theo theo yêu cầu Gia Bảo: chốt hạn đề xuất và báo cáo tiến độ cho cả hai là thứ Tư 25/11/2026 (UTC+7), không phải 18/11. Đồng bộ answers, AGENTS và bảng việc; bỏ câu chưa đặt deadline trong hướng dẫn hiện hành. Không duyệt M0/full hoặc thay yêu cầu input/GPU/review; phần chưa chạy vẫn phải báo đúng hạn với `NOT_RUN`/blocker. Kiểm tra diff và checker trước phát hành; xác nhận commit/push bằng Git/remote.
- Theo yêu cầu Gia Bảo, chuẩn bị thông báo `GB-20261116-001` trên main và thêm điểm đọc trong AGENTS/bảng việc; không sửa code, model, kết quả hoặc nhánh thành viên.
- Đã fetch: Tùng `1ab8164` (28/10), Đức `da5b552` (03/11). Ghi nhận Đức đã nộp bản sửa/đề xuất nhưng kiểm tra mới chỉ do Đức báo cáo; không nhận là đã độc lập nghiệm thu hoặc đã full train.
- Nhắc Tùng nộp M0, Đức tiếp tục bài đã nộp/cập nhật bằng chứng và blocker; cả hai xác nhận thông báo trên file trao đổi cá nhân ở phiên tiếp theo. Không tự duyệt M0/kiến trúc/full hay bỏ yêu cầu input/GPU/review. PoseLab tiếp tục dừng.
- `python scripts/check_repository.py` và `git diff --check`: PASS cho thông báo; full train và review lại code Đức: `NOT_RUN`. Phát hành qua commit thông báo trên main; đối chiếu Git/remote để xác nhận push, không suy ra hai bạn đã đọc từ việc publish.

## 30/10/2026 - Phân công hai hướng và dừng PoseLab

- Yêu cầu: Gia Bảo xác nhận mình là người điều phối; Tùng dùng mô hình/phương pháp khác, Đức thay đổi có kiểm soát mô hình hiện tại; không giao sẵn cách triển khai; tắt hoặc bỏ PoseLab.
- Thực hiện trên `giabao`, nền `c8f654b`: cập nhật AGENTS, README, bàn giao, protocol, bảng việc/nhiệm vụ, roadmap và quyết định `GB-20261030-001` trong answers. Giữ review cũ làm lịch sử, không xóa lỗi kỹ thuật hoặc bài làm Tùng/Đức; hủy WEB-01/WEB-02 và quyền frontend/QA.
- Tắt `app/start.cmd`, `app/start.ps1`; chặn ASGI lifespan trước Engine và middleware trả 503 kể cả không chạy lifespan. Giữ mã nguồn, dữ liệu lịch sử, model/threshold, notebook, kết quả và Word; bỏ phần quảng bá/hướng dẫn chạy PoseLab.
- `python scripts/check_repository.py`: **PASS**, hash model, 9 notebook, Word, dự đoán/tổng hợp và liên kết tài liệu không đổi/hợp lệ. `git diff --check`: **PASS**.
- Kiểm hành vi bằng Python tạm trong bộ nhớ: compile cú pháp toàn `app.py`; trích nguyên AST cờ/lifespan/middleware để chạy cùng FastAPI/TestClient, không import Engine hoặc tạo runtime. **PASS**: lifespan từ chối trước Engine; GET/POST/DELETE trả 503 khi không chạy lifespan. Đây là kiểm tra chặn app, không phải QA/inference toàn pipeline.
- Chạy thật hai launcher bằng subprocess: **PASS**, cùng thoát mã 1 và thông báo đã tắt; PowerShell thử cả cổng tùy chọn. Không cài thư viện/tải model/mở server. Không thấy tiến trình Python/uvicorn mang dấu hiệu PoseLab đang chạy lúc kiểm tra; không dừng tiến trình khác.
- Trạng thái giao: thay đổi local, **chưa commit/push/merge main**, không có PR; chưa tới máy hai thành viên. Publish main cần yêu cầu của Gia Bảo; sau đó hai bạn fetch/nhận cập nhật an toàn, không reset nhánh có bài làm. Full train: `NOT_RUN`.

## 30/10/2026 - Chốt mốc Đức và phát hành theo yêu cầu

- Yêu cầu tiếp: Gia Bảo cho phép commit/publish main sau khi làm rõ "hiện tại làm mốc" cho Đức. Dòng trạng thái local của mục trước là lịch sử phiên chuẩn bị, không mô tả trạng thái remote sau phát hành.
- Chốt `A3_motion_attention` / `fixed` của 07 tại nguồn `c8f654bf0f1943ffcfe375911c8b3e0015beadc5`, không phải checkpoint 06, A3_cost_225, T0/T1 hoặc Q1. Ghi nguồn model/features/recipe/input, kết quả lịch sử để nhận diện và yêu cầu tái lập bản gốc cùng bản thay đổi; không giao sẵn cách cải tiến.
- Đồng bộ vào AGENTS, README, bàn giao, protocol, bảng việc, nhiệm vụ Đức và quyết định GB-20261030-001. Không sửa helper, notebook, model, kết quả hoặc bài làm nhánh cá nhân.
- Phát hành bộ phân công/dừng PoseLab từ giabao qua PR về main sau kiểm tra; không force-push hoặc xóa nhánh. Kết quả thực tế phải xác nhận bằng trạng thái PR và SHA remote, không coi yêu cầu publish là bằng chứng push thành công. Hai bạn nhận main bằng fetch/merge an toàn; full train vẫn `NOT_RUN`.
- Kiểm tra lại trước phát hành: checker repo, diff check, nhận diện A3/feature/seed/metric gốc và kiểm chặn PoseLab đều **PASS**. Commit `777295926502207551d645f5a4bbbb2670ad14ba` đã push `origin/giabao`; [PR #1](https://github.com/aBaoN9/me-stgcn-ta-fall-detection/pull/1) về main đã mở, kiểm thấy `MERGEABLE` / `CLEAN`, không có status check CI được cấu hình. Các kiểm tra PASS là local, không gọi là CI PASS.
- Gia Bảo đã cho phép merge PR này lên main. Trạng thái merge/SHA cuối tra trực tiếp PR và `origin/main`; không tự cập nhật hoặc ghi đè nhánh `dinhtung`/`vinhduc` thay thành viên.
