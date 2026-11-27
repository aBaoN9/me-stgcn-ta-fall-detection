# Benchmark năm mô hình hiện đại — Tùng

Phạm vi theo [GB-20261124-001](../discussions/answers.md#gb-20261124-001---giao-10-model-cùng-đầu-vào-và-thang-đánh-giá-a3) và [hợp đồng chung](modern-model-benchmark.md). Mốc là A3_motion_attention nguyên bản notebook 07, huấn luyện lại trong cùng môi trường. Mã và kết quả đợt này được tách khỏi các nghiên cứu trước.

## Thiết kế khóa trước full

Mỗi mô hình có một cấu hình cố định trong [designs](../../training/modern_benchmark_tung/designs), được đưa vào khóa cùng mã chuyển thể, runner, mã tạo đặc trưng, bốn SHA-256 đầu vào, cấu hình và môi trường. Mỗi mô hình học từ đầu qua 36 lượt inner và 12 refit. Dùng chung GMDCSA-24 gồm 160 clip từ bốn người, tám kênh v2, 32 mốc và 33 khớp MediaPipe. Giữ batch 16, FP32, AdamW, weighted CE, dropout 0,3, tối đa 50 epoch inner, patience 10, cải thiện lớn hơn 1e-6, seed 42/142/242. Epoch và ngưỡng phụ chỉ chọn từ inner validation. Cấu hình duy nhất, không tìm siêu tham số qua outer-test.

Giữ nguyên thứ tự MediaPipe và quy ước bị thiếu của v2. Loader giữ tọa độ đã nội suy và đặt visibility bằng 0 ở mốc thiếu theo helper chuẩn. Các mô hình không thêm padding đầu vào. Nút ảo của Hyper-GCN là tham số bên trong, không phải khớp quan sát.

| Mô hình | Nguồn khóa | Cơ chế giữ lại và thay đổi |
| --- | --- | --- |
| CascadeFormer | [Paper](https://arxiv.org/abs/2509.00692) · [847efac](https://github.com/Yusen-Peng/CascadeFormer/tree/847efac9bec015d2d1d8f2104d255a0258ad76d7) | Dựa trên cấu hình 1.0 joint. Chiếu cả 264 giá trị mỗi mốc, T1 → T2 rồi cross-attention dùng T1 làm truy vấn. Thu gọn 48 kênh, bốn head, một block mỗi tầng, pooling theo thời gian. Bỏ pretraining tái dựng khớp bị che, giữ CE có nhãn. |
| DHA-eGCN | [Paper](https://www.mdpi.com/1424-8220/26/12/3932) · [307b508](https://github.com/oskar-ika-adi-nugroho/DHA-eGCN/tree/307b5086175964d7781c7e070483c5f4c0e0c7ce) | Hai khối 24 kênh với hai bản đồ chú ý trừ theo hệ số học được, thiên lệch theo số bước trên đồ thị, và token ngữ cảnh từ năm nhóm khớp. Một nhánh đồ thị giới hạn trên cạnh MediaPipe có trọng số và tích chập thời gian dilation 1/2. Không RICH4/RICH5 hoặc hợp nhất nhiều mô hình. |
| Hyper-GCN | [Paper](https://arxiv.org/abs/2411.14796) · [cefe70b](https://github.com/6UOOON9/Hyper-GCN/tree/cefe70b9c62ab825722cc8ece3fa26ecf5c334a1) | Bản theo Zhou và cộng sự với virtual connections. Hai khối 24 kênh, bốn head, sáu nút ảo theo vùng cơ thể. Học siêu cạnh thích ứng từ chín lân cận trong không gian biểu diễn, kết hợp đồ thị cạnh xương. Loại nút ảo khỏi biểu diễn trả về, không bỏ khớp thật. |
| ProtoGCN | [Paper](https://arxiv.org/abs/2411.18941) · [ddf7f27](https://github.com/firework8/ProtoGCN/tree/ddf7f274f9f5d9e45a2fcfeb299bfb3fd7c2303d) | Hai khối 24 kênh tạo topo động theo kênh. Dùng 16 prototype (mẫu biểu diễn học được) tái dựng topo cuối, kết hợp với đặc trưng đã tổng hợp để phân loại. Mỗi fit tạo bộ prototype mới từ đầu. Bỏ contrastive loss theo lớp, nên không đo đóng góp từ cơ chế phân tách prototype của phương pháp gốc. |
| InfoGCN++ | [Paper](https://arxiv.org/abs/2310.10547) · [8c44085](https://github.com/stnoah1/infogcn2/tree/8c44085ac134f35dc85f683dcc775fdb43172ad1) | Đồ thị động, bộ mã hóa thời gian nhân quả, lấy mẫu ẩn khi train, dùng trường vi phân đồ thị và một bước RK4 (tích phân số) để ngoại suy biểu diễn. Nối hiện tại và ngoại suy, phân loại toàn clip. Bỏ các mục tiêu phụ và đầu phân loại prefix (đoạn đầu clip), dùng CE và PyTorch sẵn có. Không dùng Apex hoặc torchdiffeq trong bản thu gọn. |

Đã kiểm tra cây công khai ở các SHA được giao, README, mã cơ chế và cấu hình tương ứng. DHA-eGCN có LICENSE Apache-2.0, ProtoGCN có LICENSE MIT. Ba cây CascadeFormer, Hyper-GCN, InfoGCN++ không thấy LICENSE/NOTICE cấp kho ở snapshot giao. Các chuyển thể được viết lại từ cơ chế đã công bố, có ghi nguồn và không phát hành mã tải từ kho tác giả hoặc trọng số bên ngoài. Thiết kế so sánh full-clip với CE, không tái lập pipeline nguyên gốc hoặc số đã công bố trên dữ liệu khác.

Trong bản phát hành InfoGCN++ tại SHA đã giao, đoạn lấy mẫu ẩn trong `SODE.forward` được để dưới dạng chú thích. Bản chuyển thể ở đây có lấy mẫu khi train và dùng trung bình khi eval theo thiết kế đã khóa, nên khác cả đường forward trong mã phát hành ở điểm này.

Giới hạn nghiên cứu: bốn người và từng lượt seed dùng lại cùng 160 clip. Các outer fold đã được xem trong các nghiên cứu trước. Bảng chính là so sánh bản thu gọn theo chế độ huấn luyện chung, chưa có tập độc lập.
