# Code kiểm chứng 07

`features.py`, `models.py`, `training.py` là các module đã được nhúng trong notebook 07 full. Chúng được giữ nguyên bytes khi đóng gói để hash code của lượt Kaggle vẫn đối chiếu được. `validate_pipeline.py` chỉ được đổi đường dẫn input/output để không phụ thuộc thư mục máy tác giả.

Các model: A0 pose-average, A1 motion-average, A2 pose-attention, A3 motion-attention và cost ×2,25. 32 mốc, 33 khớp, 4 block 64/64/128/128. Chỉ augment tập train; tính lại feature sau crop/flip/noise. Không có bước chọn seed tốt nhất hoặc tự xuất ONNX v2 trong full notebook.

Môi trường nghiên cứu phải tách khỏi môi trường web:

```bash
python -m venv .venv-research
```

Kích hoạt môi trường, cài `python -m pip install -r research07/requirements.txt`, rồi từ root repo:

```bash
python -m research07.validate_pipeline --input-root data/private
```

Lệnh cần bốn file input private, kiểm tra parity, mask, augmentation và shape; không train full. Output kiểm tra nằm ở `.runtime/validation/` và không được đưa vào Git. Kaggle notebook dùng môi trường Kaggle hiện có, không yêu cầu cài lại toàn bộ requirements local này.
