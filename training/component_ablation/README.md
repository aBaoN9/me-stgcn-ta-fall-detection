# Kiểm chứng thành phần của ME-STGCN-TA

Các module xây dựng đặc trưng, mạng và vòng huấn luyện tương ứng với thí nghiệm trong [notebook kiểm chứng thành phần](../../notebooks/07-component-ablation.ipynb). Tên package đã được đổi cho rõ chức năng; thuật toán, cấu hình và cách đánh giá không thay đổi. Mã gốc của lượt Kaggle vẫn được nhúng trong notebook để đối chiếu với hash đã lưu.

Năm cấu hình: A0 ba kênh/trung bình, A1 tám kênh/trung bình, A2 ba kênh/chú ý, A3 tám kênh/chú ý và A3 tăng hệ số lớp ngã ×2,25. Tăng cường chỉ dùng trên tập huấn luyện; không chọn seed tốt nhất hoặc xuất mô hình triển khai từ kết quả kiểm tra ngoài.

Dùng môi trường riêng với các thư viện trong `requirements.txt`. Từ thư mục gốc của repo:

`python -m pip install -r training/component_ablation/requirements.txt`

`python -m training.component_ablation.validate_pipeline --input-root data/local`

Lệnh kiểm tra căn chỉnh dữ liệu, đặc trưng, tăng cường và kích thước đầu vào; không huấn luyện toàn bộ thí nghiệm. Báo cáo kiểm tra được lưu trong `.runtime/validation/`, ngoài Git. Để huấn luyện trên Kaggle, dùng notebook tự chứa mã và đọc [hướng dẫn notebook](../../docs/notebook-guide.md).
