# Kết quả contribution study (Đức)

| Thư mục | Nội dung | Trạng thái |
|---|---|---|
| [audit/](audit/README.md) | DUC-01: tái tính và tách yếu tố từ OOF 07/08 | Đã chạy, PASS 22/22 |
| `latency/model_only_pytorch_cpu.json` | Latency model-only Q0/Q1, PyTorch CPU 1 thread, batch 1, trọng số khởi tạo | Đã đo trên máy Đức (AMD, Windows 11) |
| `runs/` | Run Q0/Q1 nested LOSO | `NOT_RUN`: chờ duyệt M0, bốn input và GPU |

Latency đo trên máy cá nhân, không đại diện cho Kaggle/ONNX/thiết bị triển khai. Không gồm MediaPipe, trích feature hay toàn pipeline. Smoke trên dữ liệu giả nằm ở `.runtime/` ngoài Git và không phải kết quả.

[Bản thảo phân tích](../../docs/research/contribution-study.md).
