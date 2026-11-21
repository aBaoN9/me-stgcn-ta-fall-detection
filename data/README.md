# Dữ liệu đầu vào

**Tool kiểm/tải độc lập:** `python scripts/training_inputs.py --input-root data/gmdcsa24` (chỉ kiểm, Python chuẩn); thêm `--download` nếu thiếu file. Nguồn pin commit dữ liệu, kiểm size/hash và không ghi đè file có sẵn. Xem [hướng dẫn local/Kaggle và notebook pin code cũ](../docs/training-start.md); không cần đổi hash hoặc chạy lại notebook 01.

**Đã bổ sung bốn input chuẩn ngày 19/11/2026 theo yêu cầu của Gia Bảo (`GB-20261119-001`).** Tùng và Đức lấy trực tiếp trong [gmdcsa24/](gmdcsa24/), dùng `--input-root data/gmdcsa24`; không cần chờ gửi file riêng hoặc chạy lại notebook 01. Đây là dữ liệu đầu vào đã có, không phải model/kết quả train mới. Repo vẫn không chứa video gốc.

Các tệp là output bước chuẩn bị dữ liệu đã dùng cho protocol 07:

| Tệp | Nội dung |
|---|---|
| [gmdcsa24_pose_32.npz](gmdcsa24/gmdcsa24_pose_32.npz) | Pose, mặt nạ phát hiện, nhãn, chủ thể và mã video |
| [gmdcsa24_features_32.npz](gmdcsa24/gmdcsa24_features_32.npz) | Đặc trưng lưu trữ và thời lượng video |
| [gmdcsa24_pose_manifest.csv](gmdcsa24/gmdcsa24_pose_manifest.csv) | Thông tin video và chất lượng pose |
| [gmdcsa24_cross_subject_folds.csv](gmdcsa24/gmdcsa24_cross_subject_folds.csv) | Căn chỉnh mẫu và thông tin chia chủ thể |

Tổng dung lượng bốn file: **15.100.639 byte**. [input_manifest.json](gmdcsa24/input_manifest.json) ghi tên, dung lượng, SHA-256 và nguồn. Cả bốn hash phải khớp `inputs` trong [locked_protocol.json của 07](../results/component_ablation/locked_protocol.json). Không thay hash chuẩn theo file khác. Hai CSV được đặt `-text` trong `.gitattributes` để Git không đổi xuống dòng/hash khi checkout trên Windows.

## Nhận dữ liệu và kiểm tra

Trên nhánh cá nhân, kiểm worktree rồi nhận main bằng merge bình thường; không reset hoặc ghi đè bài đang làm:

```bash
git fetch origin
git merge origin/main
python -m training.component_ablation.validate_pipeline --input-root data/gmdcsa24
```

Lệnh validation cần môi trường có dependency huấn luyện hiện hành; không phải full train và không tự xác nhận môi trường/GPU của thành viên. Trên Kaggle, gắn đủ bốn tệp và truyền input-root tương ứng; giữ nguyên byte/tên, không mở rồi lưu lại CSV. Bộ dữ liệu có 160 video, 79 Fall/81 ADL, bốn subject, pose `(160,32,33,4)`; CSV folds có 640 hàng của bốn outer fold. Runner nested LOSO vẫn phải dựng đủ ba inner theo subject, không lấy một val cố định trong CSV thay cả nested LOSO.

## Nguồn và phạm vi chia sẻ

- Nguồn: Ekram Alam, **GMDCSA-24**, bản `v2.1`, DOI [10.5281/zenodo.13354453](https://doi.org/10.5281/zenodo.13354453), [Zenodo record](https://zenodo.org/records/13354453). Metadata nguồn ghi **CC BY 4.0**; ghi công và tuân thủ [giấy phép nguồn](https://creativecommons.org/licenses/by/4.0/).
- Đây là các tensor/CSV dẫn xuất từ notebook 01 của nhóm, không phải video gốc của tác giả. Bản export lưu từ trước có đuôi `.zip`/`.txt`; chỉ chép sang tên `.npz`/`.csv` chuẩn, **không giải nén/đóng gói lại, không thay nội dung/nhãn/split**. SHA-256 giữ nguyên.
- Cột `path` trong CSV manifest giữ đường dẫn Kaggle gốc để truy vết; không phải đường dẫn bắt buộc phải có trên máy mới và không chứa video đi kèm. Loader dùng ID/thứ tự/nhãn để căn chỉnh tensor.
- Chỉ năm tệp được chỉ định trong `data/gmdcsa24/` (bốn input và manifest metadata) được mở ignore để publish. `data/local/`, video, dữ liệu riêng khác vẫn bị Git bỏ qua; không force-add toàn bộ thư mục dữ liệu.
- Mã PoseLab đã gỡ ngày 21/11/2026, demo mới để sau. Chia sẻ input không cho phép triển khai app, đổi dataset/giao thức hoặc xem test để chọn cấu hình; Tùng/Đức vẫn tự test/smoke thật rồi train theo quyền tự chủ đã cấp.
