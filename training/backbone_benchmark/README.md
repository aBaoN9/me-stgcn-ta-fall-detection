# M1 · TCN_v2_small và A3/07

Từ thư mục gốc repo, chạy kiểm thử phần mềm bằng Python có PyTorch, NumPy, pandas và scikit-learn:

```bash
python -B -m training.backbone_benchmark.test_benchmark
```

Kiểm thử dùng thư mục tạm và dữ liệu tổng hợp, với 8 lần fit (huấn luyện kiểm thử) trên CPU. Dữ liệu tổng hợp không đi qua lệnh chạy dữ liệu thật.

## Kaggle GPU T4

1. Tạo notebook riêng, chọn bộ tăng tốc GPU T4, dùng một GPU. Đưa bản mã M1 lên phiên notebook bằng bản sao được phép của repo có đủ thư mục mới; trước khi commit, mã M1 chỉ có ở workspace local. Mở thư mục gốc repo trong notebook.
2. Kiểm tra nhận GPU và ghi phiên bản. Giữ môi trường sẵn có nếu tương thích.

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
python -B -m training.backbone_benchmark.test_benchmark
```

3. Gắn nguồn dữ liệu riêng do nhóm cấp quyền. Dùng đường dẫn thực dưới `/kaggle/input`; chương trình sẽ dừng nếu thiếu tệp hoặc SHA-256 khác cấu hình 07.
4. Chạy thử nhỏ theo M0. Chọn thư mục đầu ra chưa tồn tại.

```bash
python -B -m training.backbone_benchmark.runner --input-root /kaggle/input/INPUT_DUOC_CAP --output /kaggle/working/tung-smoke-01 --mode smoke --device cuda
```

5. Giữ thư mục đầu ra làm bằng chứng nội bộ. Kiểm `completion.json` có 8 lần fit thực tế, hai lượt đánh giá ngoài, reload/coverage PASS, `synthetic=false`, `scientific_result=false`. Hồ sơ lưu dấu vân tay dữ liệu/mã, thiết bị, phiên bản, lịch sử, hai checkpoint và kết quả kiểm nạp lại. `peak_allocated_bytes` đo bộ nhớ CUDA cấp phát qua PyTorch cho phiên chạy.

M2 dùng `--mode full --smoke-evidence <thu-muc-smoke-that>`, cùng dữ liệu, mã và môi trường GPU. Trước khi chạy, chương trình đọc từng vòng smoke, kiểm lịch sử 6 inner và 2 refit, nạp lại hai checkpoint và chạy suy luận trên các mẫu đã giữ riêng, rồi đối chiếu dự đoán, manifest và trạng thái hoàn tất. Full dùng 3 seed, 4 người đánh giá ngoài và 96 lần fit cho cả cặp.

## Tiếp tục khi phiên Kaggle gián đoạn

- Lưu toàn bộ thư mục đầu ra ở nơi có quyền và có thể khôi phục, gồm `locked_protocol.json` và `folds/`. Có thể sao lưu khi một thư mục vòng đã xuất hiện trong `folds/`; các thư mục `.pending-*` chỉ chứa phần đang ghi. Nếu thư mục làm việc của Kaggle mất trước khi sao lưu, các vòng chưa sao lưu phải chạy lại.
- Khôi phục thư mục vào vùng ghi được và thêm `--resume` vào cùng lệnh, với cùng `--output`, mode, input, mã và môi trường. Chỉ chạy một tiến trình ghi trên mỗi thư mục. Full tiếp tục yêu cầu `--smoke-evidence` hợp lệ.
- Mỗi vòng được ghi và kiểm trong thư mục tạm, rồi đổi tên sang `folds/<variant>_seed<seed>_subject<subject>/`. Checkpoint nằm ở `model.pt`, lịch sử và dự đoán nằm ở `fold.json`. `models/manifest.json` dẫn đến các checkpoint này.
- Vòng chưa được công bố sẽ chạy lại từ đầu. Nếu vòng đã công bố bị hỏng, chương trình dừng và giữ bằng chứng. Khôi phục bản sao hợp lệ hoặc dùng thư mục run mới. Định dạng cũ và khóa khác cấu hình không thể nối tiếp.
- Sau resume, số đo bộ nhớ đỉnh chỉ phản ánh tiến trình kết thúc, có thể gồm kiểm nạp lại và các vòng còn thiếu. Đo tài nguyên huấn luyện bằng một lượt smoke liên tục trước khi quyết định dùng GPU đó cho M2.

TCN tham khảo thiết kế [Bai và cộng sự](https://arxiv.org/abs/1803.01271) và [mã tác giả](https://github.com/locuslab/TCN/blob/2f8c2b817050206397458dfd1f5a25ce8a32fe65/TCN/tcn.py), triển khai bằng các lớp PyTorch. Đệm trái tương đương đệm rồi cắt bên phải. Khởi tạo bộ lọc trước khi áp dụng chuẩn hóa trọng số bằng API parametrizations. A3 dùng trực tiếp lớp và hàm fit từ mã 07.
