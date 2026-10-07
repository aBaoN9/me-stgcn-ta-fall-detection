# Dữ liệu đầu vào

Repo không chứa video gốc hoặc tensor pose/features. Mô hình triển khai có sẵn nên chạy ứng dụng không cần dữ liệu huấn luyện.

Để chạy lại các thí nghiệm, gắn output bước chuẩn bị dữ liệu trên Kaggle gồm:

| Tệp | Nội dung |
|---|---|
| `gmdcsa24_pose_32.npz` | Pose, mặt nạ phát hiện, nhãn, chủ thể và mã video |
| `gmdcsa24_features_32.npz` | Đặc trưng lưu trữ và thời lượng video |
| `gmdcsa24_pose_manifest.csv` | Thông tin video và chất lượng pose |
| `gmdcsa24_cross_subject_folds.csv` | Căn chỉnh mẫu và thông tin chia chủ thể |

Khi kiểm tra trên máy, đặt các tệp trong `data/local/` hoặc truyền `--input-root`. Dữ liệu trong thư mục này được Git bỏ qua. Đọc NPZ với `allow_pickle=False`; các bản cũ đổi đuôi sang ZIP/TXT chỉ được chấp nhận khi nội dung và cấu trúc phù hợp.

Nguồn GMDCSA-24: https://zenodo.org/records/13354453. Chỉ dùng dữ liệu có quyền sử dụng và tuân thủ điều kiện của tác giả. Để bật video mẫu trong ứng dụng, đặt `POSELAB_EXAMPLE_VIDEO` trỏ tới một video trên máy; ứng dụng không tự tải video dataset.
