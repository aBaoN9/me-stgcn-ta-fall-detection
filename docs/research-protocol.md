# Giao thức nghiên cứu tiếp và yêu cầu học thuật

Ngày đề xuất: 28/10/2026. Đây là **kế hoạch chưa chạy**, cần người điều phối duyệt M0 trước full train. Không sửa hồi tố protocol để hợp thức hóa số đẹp.

## 1. Hiểu yêu cầu của giảng viên và backbone

Diễn giải có khả năng phù hợp, **không phải lời trích nguyên văn/xác nhận của thầy**: chọn backbone phù hợp, đưa đối chứng vào chính dữ liệu nhóm, đề xuất thay đổi kỹ thuật và chứng minh bằng thực nghiệm công bằng rằng nó giúp ở đâu, đánh đổi gì. Chạy một model sẵn rồi báo accuracy hoặc thêm giao diện chưa trả lời được câu hỏi đóng góp.

**Backbone** là phần mạng học biểu diễn đặc trưng chính. Trong `training/component_ablation/models.py`, `SkeletonModel.blocks` gồm bốn ST-GCN block là backbone; `temporal_attention` tổng hợp theo thời gian, `classifier` là đầu phân loại. MediaPipe trích pose đầu vào, không phải backbone GCN đang so sánh. Pretrained là cách khởi tạo/huấn luyện trước, không đồng nghĩa backbone.

Đổi backbone, đặc trưng đầu vào, pooling, loss và ngưỡng là các trục khác nhau; không đổi hết rồi quy thành công cho một thành phần.

## 2. Câu hỏi và đối chứng

| Câu hỏi | Phép đo | Người |
|---|---|---|
| Backbone tham chiếu nâng cao có hợp dataset nhỏ không? | Giữ input/augmentation/pooling/loss/split, đổi local ST-GCN-style sang CTR-GCN adapted | Tùng |
| Motion giúp với backbone nào? | Cặp 3/8 kênh trong mỗi backbone, cùng recipe | Tùng |
| Chất lượng pose giúp pooling bớt chịu mốc nhiễu không? | A3 gốc vs A3 + quality-aware pooling, chỉ đổi pooling | Đức |
| Đóng góp hiện tại thực sự là gì? | Tái tính A0/A1/A2/A3/cost và T0/T1 từ OOF đã lưu, báo cả kết quả âm | Đức |
| Có đáng dùng model phức tạp hơn? | Sensitivity/specificity/FN/FP theo người/seed, tham số, latency/tài nguyên | Cả hai |

CTR-GCN là **đối chứng tham chiếu nâng cao**, không được gọi là mới nhất/SOTA hiện tại. Muốn thay đối chứng: trình paper/code chính chủ, lý do/khả thi/ngân sách trước khi đổi. Không cam kết đề xuất thắng mọi metric; giảm FN nhưng tăng FP hoặc nhẹ hơn nhưng accuracy thấp hơn đều là trade-off cần lượng hóa.

## 3. Khóa dữ liệu và protocol

- Cùng 160 ID/nhãn/subject, bốn input theo [data](../data/README.md); SHA-256 đối chiếu `results/component_ablation/locked_protocol.json`. Khác hash/căn chỉnh -> dừng hỏi, không tự đổi dataset.
- Feature v2 của 07; không trộn pipeline 06 hoặc timing T1 vào track chính mới. 33 node, 32 mốc, 3 kênh cơ sở/8 kênh motion. Input `(N,C,T,V)`; adapter cần chiều người thì thêm `M=1` có mô tả.
- **Nested LOSO:** 4 outer subject; ba người còn lại chia 3 inner theo người để chọn epoch/cấu hình/ngưỡng; refit ba người outer-train rồi đánh giá người outer-test. Scaler/class weight/thống kê học chỉ fit trên train đúng split.
- Seed 42/142/242, cùng split/ID và seed list. Primary policy `fixed`, neural threshold 0,5; balanced/sensitivity là phụ và chỉ chọn bằng inner. SVM cũ dùng decision margin fixed 0, không phải xác suất 0,5.
- Recipe khởi đầu 07: max 50 epoch, patience 10, batch 16, LR 0,001, weight decay 0,0001, dropout 0,3, không scheduler. Giữ optimizer, class-balanced CE, epoch selection và outer refit của 07; multiplier Fall chính 1,0. Multiplier 1,0 **không có nghĩa tắt cân bằng lớp**.
- Ghi optimizer/implementation chính xác trong config. Thành phần kiến trúc không hỗ trợ/recipe không tương thích thì công bố và xin duyệt, không âm thầm chỉnh riêng đối thủ cho yếu đi.
- Track paper recipe/search là phụ: cùng số lượt tìm kiếm/ngân sách mỗi model, chỉ inner chọn, báo riêng fixed recipe. Pretrain/multi-stream là track riêng kèm nguồn weights/quyền sử dụng/mapping.
- `target_sensitivity=0.95`, `minimum_specificity=0.75` là mục tiêu chọn ngưỡng inner của recipe cũ, không hứa đạt trên outer hoặc chứng nhận an toàn.
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
- Kết luận phải nêu đối chứng/điều kiện/delta/FN-FP/chi phí/giới hạn. "Attention chưa cải thiện ổn định" hay "quality-aware không giúp" hợp lệ nếu đủ thí nghiệm.

## 7. Nguồn mô hình và trích dẫn

- ST-GCN, Yan/Xiong/Lin, AAAI 2018: [paper](https://arxiv.org/abs/1801.07455), [repo tác giả](https://github.com/yysijie/st-gcn).
- CTR-GCN, Chen và cộng sự, ICCV 2021: [paper](https://arxiv.org/abs/2107.12213), [repo chính chủ](https://github.com/Uason-Chen/CTR-GCN).

Đọc paper/code chính chủ, ghi commit/giấy phép/adaptation. Local `SkeletonModel` là ST-GCN-style giản lược (adjacency chuẩn hóa/edge importance, bốn block), **không mặc định tái hiện nguyên bản paper**. Score trên NTU/NW-UCLA không thay score GMDCSA-24. Không gọi quality-aware pooling là ý tưởng chưa từng có; đối chiếu related work trước khi bàn tính mới.
