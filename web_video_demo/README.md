# PoseLab — thử video trên máy

## Mở web

1. Mở `start.cmd` trong thư mục này. Lần đầu cần Internet để cài thư viện và tải MediaPipe Full.
2. Giữ cửa sổ terminal mở; truy cập `http://127.0.0.1:8765`.
3. Chọn video, chọn **Theo thời gian · thử nghiệm** (mặc định) hoặc **Chỉ phân loại toàn clip**, rồi bấm **Phân tích video**.
4. Chờ đọc hết clip. Phát video để xem khung xương, trạng thái theo thời gian và mục **Dựa vào đâu để đọc kết quả?**. Bấm mốc cảnh báo hoặc kéo thanh xem lại để đồng bộ các biểu đồ. Tải JSON/CSV để lưu kết quả.
5. **Xóa phiên / Chọn lại** xóa bản video upload trên backend. Ctrl+C dừng server.

Sau khi thư viện và model đã có, phân tích không cần mạng, GPU hoặc Kaggle. Không mở cổng này ra Internet. Python 3.11 là môi trường đã dùng để kiểm tra.

## Đầu vào và model

- Model: `../models/deployment_06/`, gồm ONNX và `model_config.json` trích nguyên từ bundle 06. Không cần thư mục Kaggle trên máy tác giả. Hash kết quả mới là SHA-256 ONNX; báo cáo cũ dùng hash bundle, xem `artifact_manifest.json`.
- MediaPipe Full v1, VIDEO mode, 1 pose, ngưỡng detection/presence/tracking = 0.35 như notebook 01.
- Đọc mọi frame, nội suy pose thiếu theo chỉ số frame rồi lấy đều 32 mẫu bằng `linspace(...).round()`.
- Chuẩn hóa tâm hông, scale thân trung vị; vận tốc chia `duration / 31`; góc thân đúng công thức của notebook 01.
- 8 kênh: relative x/y, visibility, velocity x/y, torso angle, hip displacement x/y.
- Ngưỡng Fall đọc từ bundle, hiện 0.425. Không cho chỉnh ngưỡng trên giao diện để giữ phép thử đã chốt.
- Clip tối đa 120 giây, 7.200 frame, 200 MB và 4K. MP4 H.264 thuận tiện để phát trong trình duyệt.

## Cách đọc kết quả

Khung **Toàn clip · tham chiếu** luôn là nhãn dùng cả video, không phải kết luận tại vị trí đang phát. Khung xương chỉ là pose trực tiếp; frame mất pose không vẽ người giả. 32 ô màu cho biết các mẫu trực tiếp/nội suy.

Chế độ **Theo thời gian** chạy cửa sổ gần nhất 2–6 giây, mặc định 3 giây, bước 0,5 giây. Backend dự đoán ngay khi đọc đến mốc tương ứng, không đợi đọc frame tương lai để tính đặc trưng cửa sổ đó. Frontend nhận tiến độ và quyết định gần nhất khi xử lý; sau khi hoàn tất có thể phát lại timeline. Đây là mô phỏng trên file, chưa phải webcam hoặc hệ thống cam kết realtime.

Hai cửa sổ liên tiếp có điểm >= ngưỡng tạo cảnh báo; hai cửa sổ liên tiếp dưới ngưỡng mới kết thúc cảnh báo. Khi không đủ pose hoặc frame tại mốc quyết định mất pose, tạm ngừng kết luận và ngắt chuỗi xác nhận, không gán ADL. Cần tích lũy đủ cửa sổ đầu tiên. Đoạn video ngắn hơn cửa sổ không có quyết định timeline; đây không phải kết quả âm tính. Các tham số cửa sổ/xác nhận chưa được kiểm định.

Mục **Dựa vào đâu** có hai lớp: (1) quy tắc điểm so với ngưỡng và trạng thái xác nhận; (2) số đo đầu vào gồm vận tốc khớp tương đối, góc thân, dịch chuyển hông, chất lượng pose. Không có SHAP, phân tích nhân quả hay tỷ lệ đóng góp của từng kênh. Bản triển khai nhận 8 kênh, không nhận kênh gia tốc. Vận tốc biểu đồ là trung bình độ lớn trên 33 khớp, không phải m/s; model vẫn nhận vận tốc riêng từng khớp.

Phạm vi biểu đồ mặc định là cửa sổ gần nhất đã kết thúc ở vị trí phát, không chọn cửa sổ tương lai. Có thể chuyển sang toàn clip nhưng giao diện sẽ ghi rõ có sử dụng dữ liệu sau vị trí phát. Chấm cam là mẫu nội suy/bù, không phải pose phát hiện trực tiếp.

Ô **Mốc bắt đầu ngã** là nhãn do người dùng tự xác nhận cho một sự kiện. Kết quả chỉ tính thời gian từ mốc đó đến đợt cảnh báo mới đầu tiên sau mốc; hiển thị riêng số đợt bắt đầu trước mốc. Không tự suy ra mốc ngã, sensitivity, false-alarm rate hay độ trễ thực thi trên máy. JSON và CSV cửa sổ đính kèm mốc nếu hợp lệ; mốc này không lưu bền khi tải lại trang.

Điểm Fall không được coi là xác suất đã hiệu chuẩn. ADL không có nghĩa loại trừ chắc chắn té ngã. Video mẫu thuộc train, không dùng để đánh giá người mới.

Guard chất lượng: không kết luận nếu không có pose, dưới 50% frame phát hiện hoặc dưới 16/32 mẫu trực tiếp, hoặc scale thân suy biến. Đây là ngưỡng kỹ thuật đề xuất, chưa được validation. Chỉ hỗ trợ một người. Camera di chuyển, nhiều người và FPS thay đổi có thể làm sai đặc trưng. Thời gian sử dụng FPS metadata; chưa xử lý timestamp VFR riêng.

## Quyền riêng tư

### Đánh giá hàng loạt

Mở `http://127.0.0.1:8765/batch.html` hoặc chọn **Đánh giá hàng loạt** ở thanh điều hướng. Chọn tối đa 20 video (200 MB/file, 600 MB/lượt). Chọn ADL/FALL cho từng file; ứng dụng không suy nhãn từ tên. Dùng **Xem / ghi mốc** để phát video trước khi chạy; với FALL, có thể lấy vị trí phát làm mốc bắt đầu ngã. Nếu chưa biết, để trống: clip đó vẫn được phân tích nhưng không được chấm TP/FN. Chỉ hỗ trợ một sự kiện ngã mỗi clip.

Mỗi lượt cố định cửa sổ, ngưỡng và nhãn khi bắt đầu. Model chạy tuần tự, không đồng thời với phép thử một video. **Dừng sau video hiện tại** giữ kết quả xong và đánh dấu các video chưa chạy là cancelled, không phải ADL.

Kết quả chấm cảnh báo theo thời gian, không lấy nhãn toàn clip làm kết quả thay thế. Chỉ đưa vào TP/FP/FN/TN nếu mọi cửa sổ đã đánh giá đủ pose; FALL còn cần mốc ngã hợp lệ và quyết định tại/sau mốc đó. Với FALL, ghép đợt cảnh báo MỚI đầu tiên bắt đầu từ mốc ngã tới hết clip. Cảnh báo trước mốc không tính là TP. Chưa áp thời hạn cảnh báo tối đa, vì vậy TP không chứng minh hệ thống đáp ứng kịp thời. Mẫu số bằng 0 hiển thị “—”. Các tỷ lệ chỉ phản ánh tập con đủ điều kiện; giao diện luôn hiển thị độ bao phủ và các clip bị loại. Độ trễ trung bình chỉ trên TP, không gồm bỏ sót. Chưa tính false alarms/hour.

Báo cáo có cấu hình, hash model, nhãn người dùng, các mốc cảnh báo và kết quả từng cửa sổ. Tải JSON/CSV ngay cả khi mới xử lý một phần; trạng thái/pending ghi rõ. Báo cáo tự lưu ở `%LOCALAPPDATA%\PoseLab\batch_reports`, ngoài OneDrive, có thể mở lại sau khi khởi động lại web. Danh sách hiện 10 lượt gần nhất nhưng không tự xóa file báo cáo cũ. Video upload tạm được xóa sau từng phép xử lý hoặc khi dừng hàng đợi bình thường; tắt cưỡng bức có thể để lại file trong runtime. Bộ video gốc của bạn không bị sửa hoặc xóa.

Mục **Tập giữ lại** chỉ là khai báo của người dùng, không tự xác minh trùng video, trùng người hay rò rỉ dữ liệu huấn luyện. Không thêm nhiều bản sao cùng một clip để tăng cỡ mẫu; hệ thống chưa kiểm tra trùng nội dung. Sáu clip URFD đã xem thuộc tập phát triển, không phải test độc lập. Các lượt có ghi chú QA kỹ thuật chỉ để kiểm tra phần mềm.

Không tải video lên cloud. Upload tạm ở `%LOCALAPPDATA%\PoseLab\runtime`, ngoài OneDrive. Bản upload được xóa khi xóa phiên hoặc shutdown bình thường. Nếu tắt cưỡng bức, có thể còn file tạm; người dùng có thể xóa sau khi dừng web. Tối đa một video đang chạy và năm phiên trong bộ nhớ. JSON/CSV chứa thống kê, tên file và cấu hình, không chứa video hoặc toàn bộ pose.

## Giới hạn

- Model/ngưỡng được xây dựng trên toàn clip; dùng cửa sổ trượt là thay đổi đầu vào chưa được validation. Trong lượt thử ngày 30/09/2026, URFD ADL-01 đúng ở toàn clip nhưng báo nhầm khi dùng cửa sổ 3 giây. Không điều chỉnh ngưỡng chỉ để khớp video này.

- ONNX final học cả 160 video, không có test độc lập riêng cho model này.
- Metric LOSO là của model theo fold, không phải cam kết realtime.
- Builder đặc trưng được đối chiếu tensor Kaggle trên 160 mẫu. MediaPipe Windows/CPU vẫn có thể cho pose khác Kaggle; không suy ra bit-for-bit parity từ video.
- Đây là demo nghiên cứu, không phải thiết bị y tế/cấp cứu.

## Cấu trúc

`features.py`: tiền xử lý; `motion.py`: số đo, cửa sổ nhân quả và quy tắc cảnh báo; `engine.py`: MediaPipe + ONNX; `app.py`: API/job worker; `static/`: giao diện không CDN; `prepare.py`: tải model pose chính thức; `.venv/`: môi trường Python riêng, không nằm trong Git. Xem `../docs/methodology.md` để đọc phương pháp và giới hạn.

`batch_evaluation.py`: hàng đợi, đối chiếu nhãn và lưu báo cáo; `static/batch.html`, `batch.js`, `batch.css`: trang đánh giá hàng loạt. `static/research.html`: số liệu notebook 07 đã hoàn tất, không thay model đang dùng.

`../results/external_evaluation/`: bảng đối chiếu sáu clip, mốc ngã do AI ước lượng chưa duyệt độc lập. Không dùng kết quả này như test độc lập. Video và script thử một lần không nằm trong repo.

Có thể dùng `start.ps1 -Port 8766` nếu 8765 bị chiếm. `POSELAB_RUNTIME_DIR` chọn thư mục runtime riêng; mặc định giữ LocalAppData như trước. `POSELAB_EXAMPLE_VIDEO` chọn video mẫu local được phép sử dụng; mặc định không có video và nút mẫu bị vô hiệu. Không có webcam hoặc cơ chế tự tải video dataset từ Internet.
