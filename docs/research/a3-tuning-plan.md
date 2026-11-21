# Kế hoạch tuning A3 so với TCN và GRU

Ngày lập: **21/11/2026 (UTC+7)**. Chủ track: **Gia Bảo**, nhánh `giabao`.
Trạng thái: **`PLANNED / NOT_RUN`** — chỉ có thiết kế; chưa viết runner, smoke hoặc train mới. Theo `GB-20261121-002` trong [quyết định nhóm](../discussions/answers.md).

## 1. Mục tiêu và phạm vi

Hiểu “hai cái của Tùng” là **TCN và GRU**: hai đối chứng có Accuracy trung bình cao hơn A3 rerun trong kết quả đã nhận. CTR-GCN adapted vẫn xuất hiện như tham chiếu phụ; không loại bỏ nó khỏi báo cáo. Mục tiêu là kiểm chứng A3 có cải thiện thật dưới cùng ngân sách không, **không cam kết thắng** và không tìm seed/metric có lợi để phủ nhận bài Tùng.

- Ưu tiên: tăng Accuracy tại policy `fixed`, ngưỡng **0,5**, so với cả TCN và GRU; báo cáo toàn bộ ba seed và bốn subject.
- An toàn: Sensitivity, specificity, FN/FP không được giấu. Muốn nói “tốt hơn cho phát hiện ngã” phải xét đồng thời bỏ sót và báo động giả, không chỉ Accuracy.
- Tốc độ: GRU đang có lợi thế model-only. Tuning recipe không đồng nghĩa giảm tham số hoặc đủ sức vượt latency GRU; tốc độ là mục tiêu riêng.
- Phiên này không triển khai demo, không đổi deployment/threshold hoặc sửa Word. PoseLab đã gỡ; demo mới để sau.
- Đây là track kế hoạch của Gia Bảo, **không thay đề xuất/mốc/nhiệm vụ hoặc hạn 25/11/2026 của Đức**, không yêu cầu Tùng train lại. Khi có phiên triển khai riêng, Gia Bảo thực hiện hoặc điều phối minh bạch; không âm thầm lấy phần việc đang giao Đức.

## 2. Mốc đã đo, không phải kết quả tuning

Nguồn: [TCN–A3 đã nhận](../../results/backbone_benchmark/tung-m2-rtx3050-20261120-165523/README.md) và [GRU/CTR-GCN đã nhận](../../results/backbone_benchmark/tung-gru-ctr-rtx3050-20261120-233932/README.md). Latency dùng cùng lần đo lại bốn model trong supplement, không trộn lần đo cũ.

| Model | Accuracy mean ± sample SD (%) | Sensitivity mean (%) | FN / FP qua 480 clip-seed | Tham số | Median model-only (ms) |
|---|---:|---:|---:|---:|---:|
| A3 rerun | 89,38 ± 2,50 | 92,41 | 18 / 33 | 422.871 | 2,549 |
| TCN | 91,04 ± 0,72 | 94,09 | 14 / 29 | 56.434 | 2,178 |
| GRU | 89,58 ± 1,57 | 91,14 | 21 / 29 | 64.018 | 0,366 |
| CTR-GCN adapted | 88,54 ± 2,01 | 89,03 | 26 / 29 | 24.044 | 2,906 |

A3 cần bù khoảng **1,67 điểm phần trăm** so TCN và **0,21 điểm** so GRU (tính từ số chưa làm tròn). Đây không phải khoảng cách chắc chắn có thể bù bằng tuning. A3 notebook 07 lịch sử **89,58%** không thay cho A3 rerun **89,38%**. 480 clip-seed là 160 clip lặp qua ba seed, không phải 480 mẫu độc lập.

## 3. Giả thuyết ưu tiên

1. **Recipe phù hợp hơn cho A3:** thử learning rate thấp hơn, regularization và lịch học; kiến trúc giữ nguyên để không gán nhầm cải thiện cho graph/attention mới.
2. **Khả năng tổng quát của recipe:** xem kết quả trên từng subject/seed, không chỉ mean. Chưa biết dropout thấp hơn hoặc smoothing có giúp; đây là giả thuyết cần bác bỏ nếu không đạt.
3. **Nhánh kiến trúc gọn (chỉ dự phòng):** nếu sau nghiên cứu recipe vẫn không cạnh tranh, cân nhắc giảm width/bottleneck trong một addendum riêng. Không thêm vào search sau khi xem outer để cố thắng. Đây không còn là “tuning nguyên A3” và chưa được giao triển khai.

Không tuning theo các clip S4 đã biết hoặc “sửa cho đúng” lỗi test. Không thêm augmentation, cost multiplier, optimizer khác, ensemble, TTA hoặc seed mới vào ma trận này; mỗi thay đổi ngoài kế hoạch phải thành phiên bản nghiên cứu khác, công bố đã biết test cũ.

## 4. Phần giữ nguyên

- Dữ liệu: 160 clip, 79 Fall/81 ADL, bốn subject; bốn input trong [data](../../data/README.md) đúng SHA-256 của protocol 07, giữ nguyên ID/nhãn/split.
- Feature v2, tám kênh, 32 mốc × 33 khớp, mask/interpolation/augmentation như baseline; chỉ train fold có augmentation, preprocessing không fit vào validation/test.
- A3: `SkeletonModel(channels=8, attention=True)`, bốn ST-GCN block width 64/64/128/128, temporal attention, 422.871 tham số. Dropout là biến recipe, không thay số tầng/width/graph.
- Seeds **42/142/242**; nested LOSO **4 outer × 3 inner** theo subject. AdamW, batch **16**, max epochs **50**, patience **10** như baseline; không tăng riêng ngân sách A3.
- Training CE có class weight tính **chỉ từ train fold**, Fall multiplier **1,0**. Multiplier 1,0 không có nghĩa bỏ class balance. Label smoothing chỉ thay ở C5; không dùng nhãn validation/test để tính class weight.
- Epoch tốt nhất theo **minimum unweighted, unsmoothed validation CE**, giữ epoch sớm hơn khi hòa; outer refit bằng median ba best epoch của **recipe thắng**. Không train final all-subject trong thí nghiệm này.
- Không sửa nguồn `training/component_ablation/`, `training/backbone_benchmark/`, supplement, notebooks, lock hoặc results đã nhận. Runner tương lai ở `training/a3_tuning/`, output mới ở `results/a3_tuning/<run-id>/`; hai thư mục này **chưa tạo** trong phiên lập kế hoạch.

## 5. Sáu recipe định trước

Áp dụng cùng C0–C5 cho **A3, TCN, GRU**; mỗi model giữ nguyên kiến trúc nguồn và semantics dropout sẵn có (GRU dropout ở classifier, không tự thêm recurrent dropout). C0 là cấu hình mốc trong mỗi fold. Đây là so recipe, không phải sáu thí nghiệm đủ tách riêng tác dụng nhân quả của từng siêu tham số.

| ID | Learning rate | Weight decay | Dropout | Scheduler | Label smoothing |
|---|---:|---:|---:|---|---:|
| C0 | 0,001 | 0,0001 | 0,30 | None | 0 |
| C1 | 0,0003 | 0,0001 | 0,30 | None | 0 |
| C2 | 0,001 | 0,001 | 0,30 | None | 0 |
| C3 | 0,0003 | 0,001 | 0,15 | None | 0 |
| C4 | 0,0003 | 0,001 | 0,30 | Cosine, T_max=50, eta_min=0,00001 | 0 |
| C5 | 0,0003 | 0,001 | 0,30 | Như C4 | 0,05 |

Với C4/C5, scheduler bước một lần sau mỗi epoch đã cập nhật optimizer, không warm restart; `T_max=50` cả inner và outer refit. Không nén lịch cosine về median epoch vì như vậy đổi recipe refit so inner. Lưu LR từng epoch; `scheduler.step()` không chạy trước optimizer. Validation CE dùng `weight=None, label_smoothing=0` cho mọi recipe kể cả C5.

Nguồn API/phương pháp: [PyTorch 2.10 CrossEntropyLoss](https://docs.pytorch.org/docs/2.10/generated/torch.nn.CrossEntropyLoss.html), [CosineAnnealingLR](https://docs.pytorch.org/docs/2.10/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html), [scikit-learn: nested vs non-nested CV](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html). Các nguồn giải thích cách dùng, không chứng minh các recipe này sẽ cải thiện A3. Không tự nâng PyTorch/CUDA hệ thống; runtime thí nghiệm phải khóa và ghi đúng phiên bản thực dùng.

## 6. Quy tắc chọn cấu hình, ngưỡng và ngân sách

Với từng model × outer subject × seed, chỉ dùng ba subject outer-train:

1. Train đủ sáu recipe ở ba inner folds; ở mỗi fold chọn epoch bằng validation CE như mục 4.
2. Gom dự đoán inner held-out của ba fold cho từng recipe, mỗi clip đúng một lần. Chọn theo thứ tự: **Accuracy ở 0,5 cao nhất → balanced accuracy cao nhất → unweighted unsmoothed CE pooled thấp nhất → ID C0…C5 sớm nhất**. CE pooled tính từ logits/log-softmax của checkpoint chọn epoch, theo số clip, không trung bình mean-fold có kích thước khác nhau. Không dùng thông tin outer-test ở bất kỳ tiêu chí nào.
3. Refit recipe thắng trên ba subject outer-train bằng median best epoch tương ứng; class weight tính lại từ outer-train. Đánh giá outer-test một lần sau khóa quyết định. Mỗi outer/seed có thể chọn recipe khác nhau; **không chọn một “cấu hình vô địch” theo outer mean** để chạy ngược lại.
4. Policy chính luôn fixed 0,5. Nếu xuất thêm balanced/sensitivity thì dùng đúng helper/grid/điều kiện khả thi đã khóa của baseline, chọn từ inner predictions của recipe thắng; xuất riêng, không lấy best-of-policy làm kết quả chính.

| Hạng mục | Số fit dự kiến |
|---|---:|
| Inner: 3 model × 6 recipe × 4 outer × 3 inner × 3 seed | 648 |
| Outer refit: 3 model × 4 outer × 3 seed | 36 |
| Tổng ma trận chính | **684** |

Sản phẩm chính dự kiến **36 outer checkpoint**, 1.440 dòng OOF fixed (3 model × 3 seed × 160 clip), thêm bảng config/epoch/inner scores và log đủ 684 fits. Chưa có các sản phẩm này. Fit smoke, kiểm parity C0, lỗi/rerun, ablation chẩn đoán hoặc CTR-GCN mới nếu có phải đếm riêng; không giấu vào “cùng ngân sách”.

**Hai cách so phải tách nhãn:** tuned A3 vs các mốc đã nhận là *so với frozen references, ngân sách tuning không bằng nhau*; tuned A3 vs tuned TCN/GRU theo ma trận trên mới là *cùng ngân sách search*. Không bắt Tùng chạy lại: runner track Gia Bảo tương lai import/reuse model của Tùng, giữ credit/source hash. Chỉ áp cùng số fits/max epoch không có nghĩa bằng FLOPs hoặc GPU-hours; ghi cả thời gian/peak RAM/VRAM thực đo cho từng model.

Không chạy riêng một seed rồi chỉ công bố seed thắng. Nếu thiếu tài nguyên, trước outer-test phải ghi addendum hạ **cùng** ngân sách cho cả ba model hoặc giữ trạng thái pilot/incomplete; không âm thầm cắt ma trận rồi tuyên bố full. Thời gian hoàn tất chỉ ước tính sau smoke đo throughput thật, không đoán số giờ GPU hoặc tự thuê dịch vụ.

## 7. Các bước khi có phiên triển khai

| Bước | Đầu ra / điều kiện | Hiện trạng |
|---|---|---|
| P0: lập kế hoạch | Mục tiêu, sáu recipe, ngân sách, giới hạn, phối hợp Đức | Có tài liệu này, chưa là run lock |
| P1: runner và khóa | Code/config/input hashes, runtime, guard chống leakage; parity C0 với nguồn gốc | NOT_RUN |
| P2: test và smoke thật | Shape/mask, split, selection, coverage, resume, CE/scheduler; smoke không báo như full | NOT_RUN |
| P3: full | Đủ ma trận khóa trước outer, không đổi grid sau xem test; log tài nguyên | NOT_RUN |
| P4: review | OOF, lỗi cặp, theo subject/seed, hash/checkpoint, timing cùng máy | NOT_RUN |
| P5: xác nhận mới | Subject/dataset độc lập hợp quyền, thiết kế/nhãn khóa trước xem score | Chưa có input/xác nhận độc lập |

P1/P2 phải kiểm không dùng outer labels/scores để early stop, chọn recipe/ngưỡng hoặc quyết định tiếp tục search; test round-trip config/checkpoint, cả A3/TCN/GRU, C0/C4/C5 và resume. Lưu trạng thái optimizer/scheduler/RNG khi resume để không vô tình đổi recipe. Input/hash/split hoặc test sai thì dừng sửa, không bỏ guard để chạy.

Trước P3, ghi code/config hash bất biến và kiểm nguồn C0, input, seed, augmentation; chỉ dùng tài nguyên miễn phí/đã có quyền, không tự cài lại môi trường hệ thống hoặc phát sinh chi phí. **Tài liệu này không khởi động P1–P3 trong phiên hiện tại.** Không dựng lại cổng duyệt từng lượt cho Tùng/Đức; quyền tự chủ bài hiện tại của hai bạn giữ nguyên.

## 8. Tiêu chí báo cáo, không hứa thắng

- Kết quả chính: OOF Accuracy mean ± sample SD qua ba seed, paired delta từng seed so hai đối chứng cùng run; bảng subject, Sensitivity/specificity/F1/balanced accuracy/FN/FP và lỗi ghép cặp đầy đủ. Mọi fail/incomplete phải công bố.
- Chỉ gọi “vượt cả hai về Accuracy trong thí nghiệm này” nếu tuned A3 cao hơn **cả tuned TCN và tuned GRU** ở mean ba seed, không chọn seed hoặc policy đẹp. Đặt mục tiêu an toàn phụ không giảm Sensitivity và không tăng FP so TCN cùng run; nếu không đạt, phải ghi rõ trade-off, không gọi tốt hơn toàn diện. Mốc lịch sử TCN 94,09%, FN14/FP29 chỉ là tham chiếu, không thay cho đối chứng mới.
- Không đặt 95% thành số phải đạt. Nếu A3 không thắng, giữ kết quả âm và giải thích giới hạn. Mean thắng nhưng không ổn định theo subject/seed chỉ cho kết luận mô tả, không tự gắn ý nghĩa thống kê.
- Tham số, median/p95 latency model-only đo lại cùng thiết bị/runtime/batch1/input/warmup/synchronization cho ba model; không cộng thời gian MediaPipe/demo, không dùng latency lịch sử khác máy như phép đo mới. Báo rõ search cost và inference cost.
- Với chỉ **bốn subject**, không gọi seed hoặc 480 clip-seed là người độc lập để suy p-value/độ tin cậy. Outer folds cũ đã được nhóm xem: dù chọn hoàn toàn bằng inner trong run mới, **đây vẫn là nghiên cứu thăm dò trên dữ liệu đã dùng phát triển**. Muốn tuyên bố tổng quát hơn cần xác nhận trên subject/dataset mới; không thu clip nguy hiểm hoặc tải dữ liệu sai quyền.
- Equal-FP/robustness pose có thể là nghiên cứu tiếp đăng ký riêng, không phải điều kiện buộc A3 thắng hoặc điều kiện nhận lại bài Tùng đã nghiệm thu.

## 9. Bàn giao dự kiến

Giữ riêng runner, test, manifest/source locks, lệnh tái lập, config lựa chọn từng fold/seed, raw inner/outer predictions, checkpoint, metric tính lại, FN/FP, timing và log tài nguyên. Báo cáo phải phân biệt **NOT_RUN / SMOKE / FULL / INCOMPLETE**, mốc lịch sử với run mới và tuned-vs-frozen với tuned-vs-tuned. Không ghi đè nguồn/kết quả cũ hoặc tự đổi model triển khai sau khi có bảng đẹp.
