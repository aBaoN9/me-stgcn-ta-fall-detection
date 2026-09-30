# Dữ liệu private — không nằm trong Git

Repo không chứa video train, ảnh người tham gia, tensor pose/features hoặc gói `07-inputs-verified.zip`. Model web đã có sẵn nên chạy web không cần những file này.

Để tái chạy 07, gắn private dataset/output notebook 01 trên Kaggle gồm:

| File | Vai trò |
|---|---|
| `gmdcsa24_pose_32.npz` | poses (160,32,33,4), masks (160,32), labels, subjects, video IDs |
| `gmdcsa24_features_32.npz` | feature gốc và duration phục vụ kiểm tra |
| `gmdcsa24_pose_manifest.csv` | ID video, nhãn, subject, thông tin pose |
| `gmdcsa24_cross_subject_folds.csv` | sample index và subject split |

Không dùng pickle để đọc NPZ. Loader hỗ trợ đuôi `.zip` cũ khi nội dung thật là NPZ và hai CSV cũ mang tên TXT, nhưng phải giữ đúng schema/hash. Khi kiểm tra local, đặt input riêng trong `data/private/` (đã gitignore) hoặc truyền `--input-root`.

Không có quyền tự tái phân phối dataset chỉ vì repo Private. Dùng dữ liệu của mình hoặc được chủ sở hữu cho phép và tuân thủ điều kiện nguồn. Để bật video mẫu của web, dùng `POSELAB_EXAMPLE_VIDEO` trỏ tới video có quyền sử dụng; mặc định nút mẫu bị vô hiệu nếu không có video.
