# Benchmark sơ bộ trên máy tác giả

Ba clip × ba lượt, có warmup, tách decode/MediaPipe/resample/features/ONNX và tổng. Tổng trung vị khoảng 3,8–5,7 giây cho các clip ngắn đã đo; ONNX inline khoảng 3,4–3,8 ms. ONNX-only 100 lượt sau 10 warmup: p50 5,829 ms, p95 6,857 ms. Các phép đo riêng có điều kiện CPU khác nhau, không thay thế tốc độ cả hệ thống.

MediaPipe là nút thắt quan sát được. Chưa tính mạng/upload/render browser, chưa đo webcam end-to-end, ba clip không đại diện mọi máy hoặc video. `benchmark.json` và `timings.csv` giữ thông tin đo/runtime, không kèm video.
