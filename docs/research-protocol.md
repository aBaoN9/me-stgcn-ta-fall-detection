# Giao thức nghiên cứu tiếp và yêu cầu học thuật

Phạm vi cập nhật theo Gia Bảo ngày 30/10/2026. Đây là khung so sánh, không giao sẵn mô hình hoặc công thức. Cấu hình cụ thể cần thống nhất M0 trước full train; tiến độ đã có theo báo cáo và review riêng. Không sửa hồi tố protocol để hợp thức hóa số đẹp.

## 1. Hiểu yêu cầu của giảng viên và backbone

Diễn giải có khả năng phù hợp, **không phải lời trích nguyên văn/xác nhận của thầy**: chọn backbone phù hợp, đưa đối chứng vào chính dữ liệu nhóm, đề xuất thay đổi kỹ thuật và chứng minh bằng thực nghiệm công bằng rằng nó giúp ở đâu, đánh đổi gì. Chạy một model sẵn rồi báo accuracy hoặc thêm giao diện chưa trả lời được câu hỏi đóng góp.

**Backbone** là phần mạng học biểu diễn đặc trưng chính. Trong `training/component_ablation/models.py`, `SkeletonModel.blocks` gồm bốn ST-GCN block là backbone; `temporal_attention` tổng hợp theo thời gian, `classifier` là đầu phân loại. MediaPipe trích pose đầu vào, không phải backbone GCN đang so sánh. Pretrained là cách khởi tạo/huấn luyện trước, không đồng nghĩa backbone.

Đổi backbone, đặc trưng đầu vào, pooling, loss và ngưỡng là các trục khác nhau; không đổi hết rồi quy thành công cho một thành phần.

## 2. Câu hỏi và đối chứng

| Câu hỏi | Phép đo | Người |
|---|---|---|
| Mô hình/phương pháp khác xử lý cùng bài toán ra sao? | Giữ dataset và giao thức đánh giá; tự đề xuất phương pháp đối chứng và so với ME-STGCN-TA | Tùng |
| Thay đổi có kiểm soát giúp mô hình hiện tại ở đâu? | Giữ ME-STGCN-TA làm mốc; so bản gốc với thay đổi đã giải thích và khóa trước | Đức |
| Có đáng dùng model phức tạp hơn? | Sensitivity/specificity/FN/FP theo người/seed, tham số, latency/tài nguyên | Cả hai |

Không bắt buộc CTR-GCN, quality-aware pooling, công thức hay ma trận B/Q cũ. Hai bạn tự nghiên cứu và trình lý do, phần giữ nguyên/thay đổi, tính khả thi và nguồn/quyền sử dụng trước khi chạy full. Không cam kết đề xuất thắng mọi metric; giảm FN nhưng tăng FP hoặc nhẹ hơn nhưng accuracy thấp hơn đều là trade-off cần lượng hóa.

Tùng có thể so phương pháp hoàn chỉnh, khác cả biểu diễn/cách học, nhưng phải công bố khác biệt; chỉ gọi so sánh riêng kiến trúc khi các yếu tố khác được kiểm soát. Đức giữ kiến trúc hiện tại trong hướng chính; mở rộng sửa kiến trúc cần duyệt riêng và không được gọi là giữ nguyên. Đổi chính sách ngưỡng không phải đổi kiến trúc. Với Đức, mỗi cặp chỉ thay yếu tố/nhóm yếu tố được chốt, không gộp rồi gán tác dụng riêng.

## 3. Khóa dữ liệu và protocol

- **Mốc gốc đã chỉ định cho Đức:** `A3_motion_attention` / `fixed` của notebook 07, `SkeletonModel(channels=8, attention=True, dropout=0.3)`, Fall multiplier 1.0, feature v2. Nguồn/recipe và phân biệt với checkpoint 06, A3_cost_225, timing/Q1 ghi trong [nhiệm vụ Đức](tasks/vinhduc.md). Tái lập cặp gốc/thay đổi theo mốc này; nếu muốn đổi mốc/recipe cần duyệt track riêng, không tự ghép số lịch sử với run mới. Hai bên thống nhất cùng mốc khi tổng hợp so sánh liên nhánh.
- Cùng 160 ID/nhãn/subject, bốn input theo [data](../data/README.md); SHA-256 đối chiếu `results/component_ablation/locked_protocol.json`. Khác hash/căn chỉnh -> dừng hỏi, không tự đổi dataset.
- M0 xác nhận tái lập mốc đã chỉ định cho Đức, phiên bản dữ liệu và cặp so sánh; nguồn pose/feature v2 của 07 là tài nguyên tham chiếu hiện có. Nếu phương pháp cần đầu vào/biểu diễn khác, giải thích và xin duyệt, giữ nguyên video/nhãn/split và truy vết nguồn. Không âm thầm trộn pipeline 06/07/08 hoặc suy kết quả khác đầu vào là tác dụng riêng kiến trúc.
- **Nested LOSO:** 4 outer subject; ba người còn lại chia 3 inner theo người để chọn epoch/cấu hình/ngưỡng; refit ba người outer-train rồi đánh giá người outer-test. Scaler/class weight/thống kê học chỉ fit trên train đúng split.
- Seed 42/142/242, cùng split/ID và seed list. Primary policy `fixed`, neural threshold 0,5; balanced/sensitivity là phụ và chỉ chọn bằng inner. SVM cũ dùng decision margin fixed 0, không phải xác suất 0,5.
- Không ép mọi phương pháp dùng cùng optimizer hay siêu tham số bất chấp tính phù hợp. Chốt nguyên tắc chọn, ngân sách tinh chỉnh tương đương, điều kiện dừng và cấu hình trước khi chạy; chỉ inner chọn. Những yếu tố không nghiên cứu trong cặp của Đức phải giữ nguyên. Giá trị trong recipe 07 là tham chiếu để tái lập mốc cũ, không phải lời giải áp sẵn cho phương án mới.
- Primary fixed và chính sách ngưỡng phụ phải áp dụng cùng quy tắc cho các bên; không lấy mỗi mô hình một chính sách có lợi khác nhau. Nếu nghiên cứu chính sách quyết định, báo thành cặp/track riêng đã khóa trước, không gộp với thay đổi mô hình.
- Pretrain hoặc nguồn dữ liệu/weights bổ sung cần duyệt riêng và công bố, không tự xem là cùng điều kiện. Mục tiêu recall inner không bảo đảm đạt outer hoặc chứng nhận an toàn.
- Ghi trước variants, metric chính, tie-break, ngân sách/dừng trong `locked_protocol.json` ở results riêng. Run mới dùng thư mục mới, lưu environment/git SHA/hash code (cả file uncommitted nếu có), input/config.
- Kết quả 07 là mốc cũ. Muốn gọi so sánh mới cùng recipe/môi trường: chạy lại đối chứng liên quan trong runner mới hoặc chứng minh parity và công bố điều kiện khác; không chỉ ghép số cũ/mới rồi gọi hoàn toàn công bằng.

## 4. Artifact và chỉ số bắt buộc

Mỗi run phải có `locked_protocol.json`, `predictions.csv`, `fold_metrics.csv`, `pooled_by_seed.csv`, `seed_mean_std.csv`, `errors.csv`, `completion.json`, README tái lập và log/metadata. Đây là **hợp đồng đầu ra cần triển khai**, chưa có runner tự tạo.

Schema trao đổi giữ giống OOF cũ:

```text
sample_index,video_id,subject,label,score,prediction,threshold,variant,seed,policy
```

- `0=ADL`, `1=Fall`; score neural là softmax Fall, không tự gọi đã hiệu chuẩn. Mỗi `(variant,seed,policy,sample_index)` duy nhất; mỗi nhóm đủ 160 ID/79 Fall/81 ADL. Các policy dùng cùng score, chỉ khác threshold.
- Mỗi seed pooled OOF đúng 160 clip, rồi mean/sample std (`ddof=1`) qua ba seed. Báo từng subject; không tráo trung bình fold thành pooled, không coi seed là mẫu độc lập.
- Accuracy, balanced accuracy, precision, F1/F2, sensitivity/specificity, FN/FP/TP/TN. `FP/(FP+TN)` là false positive rate mức clip, **không phải báo nhầm/giờ**. Delta tỷ lệ ghi bằng điểm phần trăm.
- Chỉ bootstrap/kiểm định nếu giải thích đơn vị độc lập theo người, seed và clip lặp; bốn subject rất hạn chế. Không tạo p-value từ 480 hàng lặp để tuyên bố ý nghĩa thống kê.
- Completion ghi mode/complete/variants/seeds, số outer/inner/refit kỳ vọng/thực tế. Mỗi neural variant có 12 outer và 48 fits nếu mỗi inner một cấu hình: 36 inner + 12 refit. Search thêm phải ghi số fits tăng.
- `complete=true` chỉ khi đủ coverage/run/metric checks. NOT_RUN/smoke/partial tách thư mục/nhãn, không tổng hợp như full.
- Checkpoint/log lớn để ở artifact Kaggle hoặc nơi nhóm có quyền; Git chỉ giữ code/bảng gọn/hash/hướng dẫn truy cập. Không upload dịch vụ công khai để lách giới hạn Git.

## 5. Hiệu suất và độ vững

Đo cùng máy/backend, batch 1, eval/no-gradient; ghi CPU/GPU, phiên bản/threads/shape/precision. Warmup >=20 lượt, đo >=100 lượt, median/p95; đồng bộ GPU khi đo GPU. Báo parameters và peak memory nếu hỗ trợ. FLOPs cần công cụ/quy ước, không bịa khi op không được hỗ trợ.

Tách model-only, MediaPipe, feature extraction và toàn pipeline. PyTorch CPU không được coi tương đương ONNX CPU mà không chỉ rõ backend; không suy realtime hệ thống từ latency backbone. Chưa tích hợp model vào pipeline thì chỉ báo model-only và phần chưa đo.

Nhiễu/mất pose tổng hợp chỉ là robustness phụ, không test ngoài. Khóa mức nhiễu/seed/mask/cách tái tạo feature trước đo; cùng nhiễu cho đối chứng, vẫn báo tập sạch, không bỏ clip khó. Dữ liệu 32 mốc đã nội suy không khôi phục đầy đủ khoảng mất pose gốc.

## 6. Giới hạn và kết luận

- Outer-test GMDCSA-24 đã xem trong phát triển; run mới là so sánh phát triển có kiểm soát, không phải xác nhận chưa từng dùng test.
- Sáu clip URFD đã xem không là held-out mới. Chưa có tập độc lập thì ghi chưa có; nếu bổ sung phải gán nhãn/tách theo người-cảnh/khóa trước dự đoán.
- Không sửa deployment từ bảng này. Sau review mới xét all-subject train/export/parity; metric trên train final không chứng minh tổng quát hóa.
- Kết luận phải nêu đối chứng/điều kiện/delta/FN-FP/chi phí/giới hạn. Thay đổi không cải thiện vẫn là kết quả hợp lệ nếu đủ thí nghiệm. Không có nhiệm vụ PoseLab/UI/QA web trong kế hoạch hiện hành.

## 7. Nguồn tham khảo lịch sử, không chỉ định mô hình

- ST-GCN, Yan/Xiong/Lin, AAAI 2018: [paper](https://arxiv.org/abs/1801.07455), [repo tác giả](https://github.com/yysijie/st-gcn).
- CTR-GCN, Chen và cộng sự, ICCV 2021: [paper](https://arxiv.org/abs/2107.12213), [repo chính chủ](https://github.com/Uason-Chen/CTR-GCN).

Các nguồn này là bối cảnh đã có, không bắt Tùng chọn CTR-GCN hoặc Đức chọn quality-aware. Với phương án thực chọn, đọc paper/code chính chủ, ghi commit/giấy phép/adaptation. Local `SkeletonModel` là ST-GCN-style giản lược, không mặc định tái hiện nguyên bản paper. Score trên dataset khác không thay score GMDCSA-24; đối chiếu related work trước khi bàn tính mới.
