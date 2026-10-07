# Notebooks và cách chạy trên Kaggle

## Chọn đúng notebook

| Chức năng | Notebook |
|---|---|
| Chuẩn bị pose và đặc trưng | [01 · Chuẩn bị dữ liệu](../notebooks/01-prepare-pose-data.ipynb) |
| Baseline Motion-SVM | [02 · Motion-SVM](../notebooks/02-baseline-motion-svm.ipynb) |
| Baseline ST-GCN | [03 · ST-GCN](../notebooks/03-baseline-stgcn.ipynb) |
| Mô hình ứng dụng, góc thân đã hiệu chỉnh | [06 · Nested LOSO và xuất mô hình](../notebooks/06-final-model-nested-loso.ipynb) |
| Kiểm chứng đặc trưng, chú ý và chi phí lớp | [07 · Kiểm chứng thành phần](../notebooks/07-component-ablation.ipynb) |
| Kiểm chứng mẫu số thời gian của vận tốc | [08 · Kiểm chứng thời gian](../notebooks/08-timing-ablation.ipynb) |

[04, 04b và 05](../notebooks/history/README.md) lưu các giai đoạn ban đầu. Không cần chạy các bản này trước 06–08. Notebook 06 tạo mô hình đang dùng trong ứng dụng; 07 và 08 đánh giá các biến thể, không tự thay mô hình triển khai.

## Input và thiết lập

1. Tạo notebook trên Kaggle và import tệp tương ứng. Bước 01 cần video gốc; các bước sau cần bốn tệp đầu vào trong [hướng dẫn dữ liệu](../data/README.md).
2. Trong thanh **Notebook → Input → Add Input**, gắn output của bước chuẩn bị dữ liệu hoặc dataset chứa đủ bốn tệp. Loader tìm đệ quy dưới `/kaggle/input` và kiểm tra căn chỉnh/hash.
3. Bật GPU cho các notebook mạng. Không cài lại PyTorch khi môi trường Kaggle đã cung cấp phiên bản tương thích.
4. Với kiểm chứng thành phần/thời gian, giữ `RUN_MODE = "full"` rồi chọn **Run All**. Cấu hình dùng ba seed 42/142/242, tối đa 50 epoch và patience 10; dừng sớm có thể kết thúc trước 50 epoch.
5. Kiểm tra trạng thái hoàn tất: kiểm chứng thành phần có 64 lượt ngoài tính cả SVM; kiểm chứng thời gian có 24 lượt ngoài và 96 lần fit mạng. Smoke chỉ kiểm tra phần mềm, không dùng làm kết quả nghiên cứu.

Notebook kiểm chứng thời gian tự chứa mã và bằng chứng về chỉ số lấy mẫu đã kiểm tra; không cần input thứ năm. Nếu bốn input khác bản đã xác minh, notebook dừng thay vì tự đoán metadata.

## Lưu phiên bản và tải kết quả

- **Chạy xong và muốn giữ trạng thái hiện tại:** tải ZIP output trước, rồi **Save Version → Type: Quick Save**. Không dựa vào Quick Save để bảo đảm lưu toàn bộ tệp working.
- **Muốn có một phiên chạy hoàn chỉnh được lưu cùng output:** **Save Version → Type: Save & Run All (Commit)**. Chế độ này chạy lại notebook.
- Tên phiên bản gợi ý: `component-ablation-3seeds-v1` hoặc `timing-ablation-3seeds-v1`.
- ZIP tương ứng: `07_controlled_ablation_full.zip` và `gmdcsa24_08_timing_controlled_ablation_full.zip`.

Chỉ tiếp tục lượt bị ngắt khi mã, input, cấu hình và môi trường phù hợp. Không trộn kết quả của các cấu hình khác nhau trong một thư mục.

## Không trộn tiền xử lý với trọng số

Mô hình ứng dụng giữ cách tạo đặc trưng của quy trình hiệu chỉnh góc thân. Kiểm chứng thành phần xử lý visibility tại mốc thiếu và tăng cường raw pose trước khi tính chuyển động; kiểm chứng thời gian tiếp tục kiểm tra mẫu số vận tốc. Không đưa biểu diễn mới vào trọng số cũ rồi xem là hệ thống tương đương đã được kiểm chứng.

Các notebook được công bố không chứa output cell. Dự đoán, bảng kết quả và cấu hình hoàn tất được lưu riêng trong [results](../results/).
