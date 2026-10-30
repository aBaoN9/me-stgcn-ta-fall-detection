# Báo cáo công việc - Gia Bảo

Người thực hiện: Gia Bảo. Nhánh phát triển: `giabao`. Vai trò: chủ repo/người điều phối. [Nhiệm vụ](../tasks/giabao.md).

## Snapshot

- Cập nhật: 30/10/2026, đổi phân công theo yêu cầu trực tiếp của Gia Bảo và dừng PoseLab; chưa chạy thí nghiệm mới.
- Tùng dùng mô hình/phương pháp khác trên cùng dataset; Đức thay đổi có kiểm soát ME-STGCN-TA. Không áp sẵn mô hình/công thức, cùng giao thức đánh giá và duyệt M0 trước full.
- Quyết định chính thức và các phản hồi cho Tùng/Đức vẫn ở `docs/discussions/answers.md` trên main; báo cáo này không thay kênh phê duyệt.
- Trạng thái tạo/publish nhánh phải đối chiếu Git/remote, không dùng nội dung báo cáo này làm bằng chứng push thành công.

## Nhật ký tiếp theo

Mỗi phiên nối thêm ngày, yêu cầu, phạm vi, file/commit/PR, lệnh kiểm tra và PASS/FAIL/NOT_RUN, giới hạn/blocker và bước tiếp. Khi review nhánh khác ghi đúng commit; khi tích hợp ghi phần được duyệt và bằng chứng, không ghi code chưa review thành DONE.

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
