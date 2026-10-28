# Web app - nhiệm vụ phụ nếu còn thời gian

Ngày bổ sung: 28/10/2026. Trạng thái `OPTIONAL_NOT_STARTED`. Chủ repo cho phép làm lại web app nếu còn thời gian; không bắt buộc hoàn tất mục này để nghiệm thu phần nghiên cứu, không được dùng nó thay benchmark/ablation.

## Điều kiện bắt đầu

- Phần nghiên cứu chính đã đủ bằng chứng để review, hoặc đang chờ GPU/quyết định và còn việc độc lập có thể làm. Ghi rõ vào báo cáo, không bỏ train/audit để trang trí web.
- Đọc `app/README.md`, các luồng UI/API hiện có; ghi một bản thiết kế ngắn và phạm vi. Làm frontend trên backend hiện tại đã được cho phép; đổi API/backend/framework/dependency hoặc đụng model cần gửi câu hỏi Markdown và đợi duyệt riêng.
- Làm trên nhánh cá nhân hiện có, không tạo nhánh tên khác hoặc tự đẩy main.

## WEB-01 - Đức làm lại giao diện

Người thực hiện: Đức, nhánh `vinhduc`. Được sửa `app/static/**`, `app/README.md`; thêm bản thiết kế/hướng dẫn kiểm tra `docs/research/web-app-redesign.md`, ghi tiến độ vào báo cáo Đức. Đây là ngoại lệ có giới hạn cho quy tắc trước đây không sửa app; **không mở quyền sửa toàn bộ app**.

Mục tiêu: demo dễ hiểu và sử dụng hơn, không chỉ thay màu. Giữ ngôn ngữ/nhận diện PoseLab và các chức năng đã có; ưu tiên:

1. Luồng tải video -> đang xử lý -> kết quả có tiến trình, trạng thái rỗng/lỗi rõ, không bấm gửi trùng khi chạy.
2. Hiển thị Fall score/ngưỡng/chất lượng pose, trạng thái không đủ dữ liệu và cảnh báo giới hạn nghiên cứu. Không đổi nhãn/chỉ số để làm model có vẻ tốt hơn.
3. Timeline/video/khung xương, lịch sử sự kiện và batch evaluation vẫn truy cập và sử dụng được; không bỏ chức năng cũ để dễ làm UI.
4. Responsive desktop/mobile, tiếng Việt rõ, bàn phím/focus/contrast phù hợp, không thêm dịch vụ gửi video/dữ liệu ra ngoài.
5. Chạy lại bằng lệnh hiện có; giữ API contract. Không đổi `engine.py`, `features.py`, `motion.py`, model/config/threshold hoặc database/runtime riêng để làm giao diện hoạt động.

Nộp ảnh trước/sau, danh sách file, lệnh chạy, trình duyệt/viewport, các luồng đã test và lỗi còn tồn tại. Ảnh/log không chứa video riêng hoặc dữ liệu nhạy cảm; không nhận QA PASS khi chưa mở/test app.

## WEB-02 - Tùng review và QA độc lập

Người thực hiện: Tùng, nhánh `dinhtung`. Ghi `docs/research/web-app-qa.md` và báo cáo Tùng, **không cùng sửa `app/static/` với Đức**. Review bản preview hoặc commit đã được người điều phối cung cấp/duyệt; ghi đúng SHA, không tự merge thay main.

- Kiểm các luồng upload/result, timeline, batch, history, lỗi/không đủ pose và responsive, bàn phím.
- Xác nhận model/config/ngưỡng và ý nghĩa kết quả không bị đổi; ghi hành vi, bước tái hiện và mức ảnh hưởng nếu phát hiện lỗi.
- Dùng [file trao đổi Tùng](../discussions/dinhtung.md) để gửi phản hồi cho Đức/người điều phối, push mỗi lần hỏi theo [quy trình](../discussions/README.md).
- Cần thay backend: mô tả vấn đề/phạm vi qua Markdown và chờ duyệt; không tự sửa vì thấy thuận tiện.

## Tiêu chí và bàn giao

- [ ] Nghiên cứu chính không bị bỏ; lý do/thời gian làm web được ghi rõ.
- [ ] Các luồng cũ được giữ, UI mới usable trên desktop/mobile; trạng thái lỗi/thiếu pose đúng nghĩa.
- [ ] Không thay inference/model/preprocessing/API hoặc thêm dependency chưa duyệt.
- [ ] Có ảnh/bản thiết kế, kiểm tra thực tế, QA ghi đúng bản/commit và báo cáo phần chưa kiểm được.
- [ ] `python scripts/check_repository.py` PASS; PR gửi người điều phối review, không tự merge/deploy.

Nếu hết thời gian, ghi `SKIPPED_TIME_BUDGET` với lý do trong báo cáo. Không báo đã làm web hoặc đã kiểm thử chỉ vì có file nhiệm vụ này.
