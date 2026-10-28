# Kiểm chứng đóng góp - bản thảo của Đức

Nhánh `vinhduc`. Bản Markdown để người điều phối đưa vào Word sau review; chưa phải kết luận chốt. [Nhiệm vụ](../tasks/vinhduc.md) · [Báo cáo](../reports/vinhduc.md).

## DUC-01 - Tái tính từ OOF đã lưu

Ngày chạy: 28/10/2026. Không train lại, không dùng GPU. Chỉ đọc `results/component_ablation/` (notebook 07) và `results/timing_ablation/` (notebook 08).

```bash
python -m training.contribution_study.audit_existing
```

Script tự cài lại công thức metric bằng numpy, không import code train cũ, rồi đối chiếu với bảng đã lưu. Đầu ra ở [results/contribution_study/audit](../../results/contribution_study/audit/README.md); hash input/code/môi trường ở `audit_checks.json`.

### Kiểm tính toàn vẹn: PASS 22/22

| Kiểm tra | 07 component | 08 timing |
|---|---|---|
| Khóa `(variant,seed,policy,sample_index)` duy nhất | PASS, 7.680 hàng | PASS, 2.880 hàng |
| Mỗi nhóm đủ 160 ID, 79 Fall / 81 ADL | PASS, 48 nhóm | PASS, 18 nhóm |
| Nhãn/subject/video_id không đổi giữa các nhóm | PASS | PASS |
| `prediction == (score >= threshold)`; fixed = 0,5; các policy chung score | PASS | PASS |
| `pooled_by_seed.csv` tái tính từ dự đoán | PASS, sai lệch tối đa 1,1e-16 | PASS, 1,1e-16 |
| Confusion matrix theo subject khớp `fold_metrics.csv` | PASS, 192 hàng | PASS, 72 hàng |
| `errors.csv` đúng bằng các hàng dự đoán sai | PASS, 965 | PASS, 367 |
| `seed_mean_std.csv` (sample std, `ddof=1`, qua 3 seed) | PASS, 1,2e-16 | PASS, 3,6e-15 |

**Phát hiện phụ:** score của T0 (08) **giống hệt từng giá trị** score của A3_cost_225 (07) trên 480 cặp seed×clip; 0 dự đoán khác nhau. Như vậy T0 là cùng một đối chứng với A3_cost_225, không phải lần lặp độc lập. Không được tính 07 và 08 thành hai bằng chứng riêng cho cấu hình này. Script chỉ ghi nhận sự trùng khớp, không xác định nguyên nhân (dùng lại kết quả hay chạy lại tất định).

### Tách từng yếu tố, policy fixed (ngưỡng 0,5)

Delta = treatment − baseline trên cùng seed và cùng 160 clip, đơn vị **điểm phần trăm (pp)**. Mean ± SD qua 3 seed. "Seed tăng/giảm" đếm theo accuracy. Đây là 3 lần lặp trên cùng dữ liệu, không phải 480 mẫu độc lập. Không tính p-value.

| Yếu tố | Cặp | ΔAccuracy | ΔSens | ΔSpec | ΔFN | ΔFP | Seed tăng/giảm/hòa |
|---|---|---:|---:|---:|---:|---:|:---:|
| Motion | A0 → A1 (average) | +4,79 ± 3,66 | +5,49 | +4,12 | −4,3 | −3,3 | 3/0/0 |
| Motion | A2 → A3 (attention) | +1,88 ± 2,86 | −1,27 | +4,94 | +1,0 | −4,0 | 2/1/0 |
| Attention | A0 → A2 (pose) | +3,12 ± 4,51 | +7,59 | −1,23 | −6,0 | +1,0 | 2/1/0 |
| Attention | A1 → A3 (motion) | +0,21 ± 2,53 | +0,84 | −0,41 | −0,7 | +0,3 | 2/1/0 |
| Cost 2,25 | A3 → A3_cost_225 | +0,00 ± 3,12 | +4,64 | −4,53 | −3,7 | +3,7 | 1/1/1 |
| Timing | T0 → T1 | −1,88 ± 2,25 | −2,53 | −1,23 | +2,0 | +1,0 | 1/2/0 |

Từng seed (FN gốc→mới; FP gốc→mới), seed 42 / 142 / 242:

- A0→A1: FN 3→9, 17→5, 14→7; FP 13→6, 13→13, 14→11. Seed 42 tăng FN dù accuracy vẫn tăng nhẹ.
- A2→A3: FN 5→3, 5→8, 6→8; FP 14→8, 14→9, 15→14.
- A0→A2: FN 3→5, 17→5, 14→6; FP 13→14, 13→14, 14→15.
- A1→A3: ΔAcc +2,50 / +0,62 / −2,50 pp.
- A3→A3_cost_225: FN 3→3, 8→3, 8→2; FP 8→13, 9→14, 14→15. Số clip sửa đúng/làm sai: 5/10, 5/5, 8/3.
- T0→T1: FN 3→5, 3→3, 2→6; FP 13→10, 14→18, 15→17.

Theo subject: motion trong nhánh attention (A2→A3) làm giảm trung bình ở S3 (−4,7 pp accuracy) và giảm sensitivity S4 (−11,8 pp). T1 giảm nhiều nhất ở S4 (−6,3 pp). Chi tiết: `paired_delta_by_subject.csv`.

Kiểm độ nhạy với policy chọn ngưỡng ở inner: A0→A1 vẫn tăng 3/3 seed ở cả `balanced` và `sensitivity`. A1→A3 thành **−0,83 pp** ở cả hai policy. Cost với policy `sensitivity` có ΔFP +4,7, ΔSpec −5,76 pp, 0 seed tăng accuracy. Xem `paired_delta_summary_{balanced,sensitivity}.csv`.

### Điều có thể và không thể kết luận

Có thể nói, kèm giới hạn 4 subject / 3 seed / outer đã xem khi phát triển:

1. **Motion có bằng chứng giúp khi pooling average.** A0→A1 tăng ở cả 3 seed và cả 3 policy. Tuy vậy SD lớn (±3,66 pp), một phần do A0 không ổn định (FN 3/17/14).
2. **Motion trong nhánh attention yếu hơn và đổi tính chất.** A2→A3 chủ yếu tăng specificity (+4,94 pp) và giảm FP, nhưng sensitivity giảm nhẹ, 1 seed giảm accuracy. Không nên nói "motion luôn tăng mọi chỉ số".
3. **Attention trên motion chưa có bằng chứng.** A1→A3 là +0,21 ± 2,53 pp, dấu không nhất quán qua seed, âm ở policy balanced/sensitivity. Attention trên pose (A0→A2) tăng sensitivity, nhưng chủ yếu ở 2 seed A0 bỏ sót nhiều; seed 42 giảm.
4. **Cost 2,25 là trade-off.** Đổi khoảng 3,7 FN lấy 3,7 FP, accuracy trung bình không đổi. Nó thay ranh giới quyết định: ở seed 42 sửa đúng 5 clip nhưng làm sai 10 clip.
5. **Timing T1 chưa cải thiện.** Trung bình giảm cả sensitivity và specificity, 2/3 seed giảm.

Không thể nói:

- Không suy nhân quả từ heatmap attention; audit này không dùng heatmap.
- **06 vs 07 không là cặp chỉ khác một yếu tố**: khác pipeline feature (v1/v2), quy trình train/final refit và ngưỡng. Không dùng 06 để đo đóng góp motion/attention.
- Không có ý nghĩa thống kê. 3 seed trên cùng 160 clip của 4 người; SD nhỏ không đồng nghĩa tổng quát hóa sang người/cảnh mới.
- Kết quả 07/08 là mốc cũ. So sánh mới (Q0/Q1) phải chạy lại đối chứng trong runner mới.

### Lỗi lặp lại (policy fixed, số seed sai trên 3)

Có 63/160 clip sai ít nhất một lần trên 7 cấu hình mạng. Toàn bảng ở `clip_error_frequency_fixed.csv`.

| Clip | A0 | A1 | A2 | A3 | A3_cost | T0 | T1 | Quan sát từ score |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| S4_FALL_05 | 0 | 3 | 0 | 2 | 3 | 3 | 3 | Hai biến thể chỉ pose đúng mọi seed; mọi biến thể có motion đều sai ít nhất 2/3 |
| S3_FALL_06 | 0 | 1 | 1 | 3 | 0 | 0 | 1 | Chỉ A3 multiplier 1,0 sai ổn định; cost sửa được |
| S2_FALL_03 | 1 | 2 | 1 | 2 | 1 | 1 | 1 | Sai rải rác mọi cấu hình, gần ranh giới |
| S2_FALL_15 | 2 | 1 | 0 | 1 | 1 | 1 | 1 | Cost làm sai mới ở seed 42 |
| S4_ADL_05, S4_ADL_06 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | FP ở **mọi** cấu hình và mọi seed (21/21) |
| S1_ADL_03 / S1_ADL_02 | 3/3 | 2/2 | 3/3 | 3/2 | 3/3 | 3/3 | 3/3 | FP gần như mọi lần (20/21, 19/21) |
| S3_ADL_05 | 3 | 1 | 1 | 3 | 3 | 3 | 3 | Biến thể average/pose-attention ít sai hơn |

Mô tả failure mode, **chưa xác minh bằng video**: máy này không có video/pose/NPZ, chỉ có `data/README.md`.

- `S4_FALL_05`: mẫu "đúng khi chỉ dùng pose, sai khi thêm motion" gợi ý kênh vận tốc của clip này bất thường (ví dụ ngã chậm, mất pose quanh lúc ngã, hoặc frame lấy mẫu bỏ qua pha ngã). Đây là giả thuyết cần xem video/mask, không phải kết luận.
- `S4_ADL_05/06`, `S1_ADL_02/03`: sai bất kể kiến trúc nên khó quy cho motion/attention/cost. Cần đối chiếu hành động thực tế (ví dụ ngồi/nằm nhanh) trước khi quy cho lỗi pose. **Không đổi nhãn** vì model sai.
- Không điều chỉnh thiết kế/ngưỡng DUC-02 theo score outer-test của các clip này.

## DUC-02 - Đề xuất M0: quality-aware temporal pooling

Trạng thái: **đề xuất chờ người điều phối duyệt**, chưa có kết quả, chưa xác nhận tính mới. Câu hỏi duyệt: `DUC-20261028-001` trong [file trao đổi](../discussions/vinhduc.md).

### Giả thuyết

Attention pooling của A3 có thể dồn trọng số vào bước thời gian có pose bị thiếu/visibility thấp. Thêm prior chất lượng dương vào attention có thể giảm ảnh hưởng các bước đó. Kết quả âm (Q1 không hơn Q0) vẫn được báo.

### Công thức khóa trước train

Ký hiệu từ code hiện có (`training/component_ablation/models.py`, `features.py`):

- Input `x` có shape `(N, 8, 32, 33)`. Kênh 2 là visibility MediaPipe `clip(pose[...,3], 0, 1)`, **đã gán 0** tại frame `masks=False` (không phát hiện pose). Augmentation dropout gán 0 cho các khớp bị bỏ.
- Chất lượng frame: `q_t = (1/33) Σ_v clip(x[:,2,t,v], 0, 1)` ∈ [0,1], lấy từ input thô **trước** `data_bn`. Vì đọc từ chính tensor đầu vào, q sau augmentation (crop nội suy mask bằng AND, flip, dropout khớp) luôn khớp với input; không dùng q sạch.
- Ánh xạ sang bước attention: backbone có một block stride 2, nên 32 frame thành 16 bước. Conv thời gian tại block đó có kernel 9, padding 4, nên output `t'` có tâm tại frame `2t'`. Lấy `q'_{t'} = mean(q_{2t'-1}, q_{2t'}, q_{2t'+1})`, không tính padding. Cụ thể: `avg_pool1d(kernel 3, stride 2, padding 1, count_include_pad=False)`.
- Pooling: `w_{t'} = softmax_{t'}(a_{t'} + log(ε + q'_{t'}))`, tương đương `w ∝ exp(a)·(ε + q')`. `a` là điểm attention gốc (Conv1d-Tanh-Conv1d).
- **ε = 0,1**, chọn trước, không tìm kiếm. Bước mất hoàn toàn bị giảm tối đa 11 lần so với bước visibility 1, không bị xóa hẳn. Đổi ε cần duyệt riêng và tính là ngân sách tìm kiếm.
- **All-missing**: `q' = 0` mọi bước, prior hằng số, softmax bất biến, trọng số trở về đúng attention gốc. Không NaN và không cần nhánh riêng. Đây chỉ là fallback số học: model vẫn cho score trên input không đủ pose. **Không** coi score đó là xác suất tin cậy, không đổi cơ chế abstain của app, và không tính abstain là TN.
- q không dùng label/subject/outer prediction; prior không có tham số học (số tham số Q1 = Q0).

### Giới hạn dữ liệu đã biết

- Pose nguồn đã được nội suy về 32 frame. Mask chỉ là cờ theo frame, không theo khớp; visibility theo khớp lấy từ MediaPipe. Khoảng mất pose gốc không khôi phục được (xem ghi chú trong `validate_pipeline.py`).
- Chưa kiểm phân bố thực của q trên 160 clip vì máy hiện tại không có bốn input. Nếu đa số clip có q gần hằng, Q1 gần như trùng Q0. Đây là kết quả hợp lệ, không phải lý do đổi công thức sau khi xem outer.
- Visibility MediaPipe không phải xác suất đã hiệu chuẩn; q là proxy, không phải confidence đo được.

### Related work và đóng góp

- ST-GCN (Yan et al., AAAI 2018, [arXiv 1801.07455](https://arxiv.org/abs/1801.07455)) biểu diễn mỗi khớp bằng `(X, Y, C)`, trong đó C là confidence của OpenPose. Dùng confidence làm **đặc trưng đầu vào** đã có từ trước; A0-A3 cũng đưa visibility vào kênh 2.
- Attention-based MIL pooling (Ilse, Tomczak, Welling, ICML 2018, [arXiv 1802.04712](https://arxiv.org/abs/1802.04712), [code](https://github.com/AMLab-Amsterdam/AttentionDeepMIL)) là dạng attention pooling Tanh-MLP mà `temporal_attention` của A3 tương tự.
- RA-GCN (Song, Zhang, Shan, Wang, IEEE TCSVT, [arXiv 2008.03791](https://arxiv.org/abs/2008.03791)) xử lý skeleton bị che khuất/không đầy đủ bằng nhiều luồng GCN, đánh giá trên tập che khuất tổng hợp từ NTU.
- PeGCN (Yu, Yoon, Jeon, Applied Intelligence 2022, [arXiv 2003.07514](https://arxiv.org/abs/2003.07514)) học bền với skeleton nhiễu/thiếu bằng predictive coding.
- PoseConv3D (Duan et al., CVPR 2022, [arXiv 2104.13586](https://arxiv.org/abs/2104.13586)) dùng heatmap thay graph, báo bền hơn với nhiễu pose.
- STEP ([arXiv 2608.19987](https://arxiv.org/abs/2608.19987)) dùng trọng số theo confidence pose trong cửa sổ thời gian để giảm báo nhầm khi che khuất, nhưng cho bài toán phát hiện bất thường. Mới chỉ đọc tóm tắt qua tìm kiếm, chưa đọc toàn văn.

Đánh giá: dùng confidence/visibility để làm bền mô hình skeleton là **ý tưởng đã có**. Đóng góp đồ án chỉ là **adaptation** (đưa prior visibility vào attention pooling của A3 trên GMDCSA-24) **và kiểm chứng có kiểm soát** Q0/Q1. Không tuyên bố phương pháp mới. Chưa tìm thấy bài fall detection dùng đúng dạng prior này, nhưng tìm kiếm chưa đầy đủ nên không suy ra tính mới.

### Code và kiểm tra phần mềm (M1, không cần dữ liệu thật)

- `training/contribution_study/quality_pooling.py`: `QualityAwareSkeletonModel` kế thừa `SkeletonModel`, không sửa code cũ.
- `training/contribution_study/runner.py`: nested LOSO Q0/Q1 cùng recipe 07 (import helper 07, sao chép `fit_model` chỉ để thay factory model). Xuất đủ hợp đồng artifact của protocol, trừ `paired_*`; audit DUC-01 tái sử dụng được cho cặp Q0/Q1.
- `python -m training.contribution_study.check_quality_pooling`: tensor tổng hợp, smoke runner trên dữ liệu giả. **PASS 13/13** (28/10/2026, torch 2.10.0+cpu):
  - Q0 (prior tắt) **bằng hệt** `SkeletonModel` cùng seed: cùng khởi tạo, logits và attention. Số tham số bằng nhau.
  - Attention có shape 16 sau stride, tổng 1. Ánh xạ q có tâm `2t'`. `w_Q1 = w_Q0·(ε+q')/Σ`.
  - q hằng hoặc all-missing thì Q1 = Q0, không NaN. q khớp mask/visibility sau 200 lần augment (crop, flip, dropout khớp).
  - Gradient hữu hạn và tới được attention.
  - Smoke runner trên dữ liệu giả: 8/8 outer, 32/32 fits, 160 OOF/nhóm, không trùng khóa, metric tái tính khớp, không rò subject.
  - Kiểm đột biến: thay prior bằng hằng số thì kiểm tra tích prior bắt được lỗi. Bước thiếu pose bị giảm trọng số từ khoảng 0,062 xuống khoảng 0,013 trên tensor ví dụ.
- Chi phí: Q0 và Q1 cùng **422.871 tham số**. Latency model-only PyTorch 2.10 CPU, 1 thread, batch 1, 5×(20 warmup + 100 đo), xen kẽ thứ tự trên máy Đức (AMD, Windows 11): Q0 median 7,45 ms / p95 9,32 ms; Q1 median 7,90 ms / p95 10,94 ms. Không phải latency ONNX/pipeline/Kaggle. Chênh lệch nhỏ cỡ nhiễu đo, không khẳng định.

## DUC-03 - Thực nghiệm Q0/Q1

| Mã | Cấu hình | Trạng thái |
|---|---|---|
| Q0_A3_attention | 8 kênh, attention gốc, multiplier 1,0, recipe 07 | `NOT_RUN`: chờ M0 + bốn input + GPU |
| Q1_A3_quality_prior | Q0 + prior `log(0,1 + q')` | `NOT_RUN`: như trên |

Kỳ vọng full: 2 variant × 3 seed × 4 outer = 24 outer, 96 fits. Q0 chạy lại trong runner mới, không lấy số A3 cũ làm đối chứng. Lệnh sau khi duyệt (Kaggle GPU, thư mục mới):

```bash
python -m training.contribution_study.runner --input-root <kaggle-input> --mode full --output results/contribution_study/runs/q0_q1_full_<YYYYMMDD>
python -m training.contribution_study.runner --input-root data/local --mode smoke --output .runtime/contribution_study/smoke_real   # smoke thật, không phải kết quả
```

Cost 2,25 cho Q1 và robustness mất pose tổng hợp: chưa đề xuất, chỉ làm sau duyệt riêng.

## DUC-04

`WAITING_FOR_TUNG`.
