# Hướng dẫn notebook và Kaggle

## Bản nên chạy

- Dùng **06** khi cần đọc/tái lập quy trình corrected-angle và model triển khai hiện tại.
- Dùng **07** để kiểm chứng motion, attention và cost, không tự xuất model dùng trên web.
- **01** tạo dữ liệu pose; **02/03** là baseline lịch sử. **04/04b/05** được giữ vì chúng giải thích quá trình hình thành phương pháp, không phải các bước bắt buộc trước 06/07.
- Các notebook lịch sử có thể còn giả định tên biến, input hoặc API của môi trường tại thời điểm viết. Đọc từng cell và gắn đúng input. Không tuyên bố tất cả notebook lịch sử đã được chạy lại trong lượt đóng gói GitHub.

## Chạy 07

1. Tạo Kaggle notebook **Private**, import `notebooks/07-gmdcsa24-controlled-ablation.ipynb`. Tên notebook: `07-gmdcsa24-controlled-ablation`.
2. Trong thanh phải **Notebook → Input → Add Input**, gắn output 01 đủ bốn file. Nếu có gói input trên máy, dùng **Upload**, tạo dataset Private rồi gắn. Không upload ZIP chứa video vào GitHub.
3. Cần `gmdcsa24_pose_32.npz`, `gmdcsa24_features_32.npz`, `gmdcsa24_pose_manifest.csv`, `gmdcsa24_cross_subject_folds.csv`. Lượt đã lưu có 160 mẫu, 79 Fall, 81 ADL và subject 1–4. Loader tìm đệ quy dưới `/kaggle/input` và từ chối các bản trùng khác hash.
4. Bật GPU trong Settings. Giữ `RUN_MODE = "full"`, không cài lại PyTorch nếu Kaggle đã cung cấp. Chạy Run All. Cấu hình đã chốt là 50 epoch tối đa, patience 10, ba seed 42/142/242; 240 lần fit mạng tính cả inner/refit và 76 lần fit SVM trong grid full.
5. Kết thúc có `Full run completed` và `07_controlled_ablation_full.zip` trong `/kaggle/working`. Không lấy smoke làm số liệu nghiên cứu. `completion.json` full phải có `outer_runs=64`.
6. **Đã Run All xong:** Save Version → Version Name `07_gmdcsa24_full_3seed_YYYY-MM-DD` → Type **Quick Save** để lưu notebook hiện tại mà không chạy lại. Tải ZIP output riêng trước khi session mất; không dựa vào Quick Save để bảo đảm lưu mọi file working.
7. **Muốn chạy một phiên version mới và lưu output của phiên ấy:** Type **Save & Run All (Commit)**. Việc này chạy lại notebook, không phải chỉ giữ bản train đang có. Không đổi sang Public.

Nếu ngắt session, chỉ resume cùng code/config/hash. Session mới cần gắn output version cũ và chép nguyên thư mục run về `/kaggle/working`; input read-only không phải thư mục output. Outer chưa hoàn tất sẽ chạy lại các inner. Không trộn hai cấu hình trong cùng run.

## Bản dữ liệu và model không được trộn

06 dùng pipeline deployment cũ đã đối chiếu và corrected torso angle index 9. 07 đặt visibility bằng 0 tại mốc thiếu pose và augment raw pose trước khi tính motion. Không đưa feature v2 vào checkpoint 06 rồi gọi đó là đánh giá đúng. Web trong repo giữ đúng feature/ONNX 06.

Output cell của notebook GitHub đã được xóa để tránh các bảng debug, ảnh dữ liệu và trạng thái không tái lập. Mã cell không bị thay thuật toán trong lượt đóng gói. Kết quả chạy full đã có trong `results/notebook_07/`; những notebook chỉ có output tổng hợp cũ không được tự gán kết quả 07.
