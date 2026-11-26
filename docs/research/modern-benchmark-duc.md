# Benchmark năm mô hình hiện đại của Đức — M0

Quyết định giao việc: **GB-20261124-001** ([hợp đồng chung](modern-model-benchmark.md)). Người làm: Đức, nhánh `vinhduc`.
Trạng thái: **M0_DOCUMENTED** — khóa ngày 26/11/2026 (UTC+7), **trước mọi kết quả outer-test của track này**. Đây là đăng ký phương án theo quyền tự chủ, không phải đơn xin duyệt và không phải nghiệm thu.

Code khóa: `training/modern_benchmark_duc/` (runner, 4 model port, graph, test). `code_sha256 = 43e2e1c4190ee62e938e2c59a2cbe22c15861885c8fe3f033ef8330bba1e8b4a` (SHA-256 của bảng hash từng file, CRLF→LF; bảng đầy đủ trong `locked_protocol.json` của mỗi run và trong báo cáo test). Sửa bất kỳ file `.py` nào trong gói hoặc `training/component_ablation/` làm đổi khóa: phải test, smoke lại và dùng thư mục run mới.

## 1. Câu hỏi

Trên cùng 160 clip GMDCSA-24, cùng tensor v2 `(N,8,32,33)`, cùng recipe CE và nested LOSO, năm backbone được giao so với A3 (`A3_motion_attention` của notebook 07) cho Accuracy/F1 lớp Fall/FN/FP thế nào ở ngưỡng cố định 0,5? Đây là bảng **controlled-backbone, CE-adapted**, train từ đầu, một cấu hình cố định mỗi model. Không phải tái lập score/recipe paper; không chứng minh phương pháp gốc tốt/kém phổ quát.

## 2. Mô hình, nguồn và giấy phép

| ID | Mô hình | Nguồn @ commit đã đọc | Giấy phép | Tham số | Trạng thái M0 |
|---|---|---|---|---:|---|
| `A3_motion_attention` | A3/07 (đối chứng) | `training/component_ablation/` (nguyên `SkeletonModel`, `fit_model`) | repo | 422.871 | chạy |
| `skateformer_v2_ce` | SkateFormer | KAIST-VICLab/SkateFormer @ `0249e19` | MIT; README: chỉ nghiên cứu/giáo dục | 1.963.242 | chạy |
| `block_gcn_v2_ce` | BlockGCN | ZhouYuxuanYX/BlockGCN @ `7b45971` | Apache-2.0 | 1.466.066 | chạy |
| `de_gcn_v2_ce` | DeGCN (mã công bố) | WoominM/DeGCN_pytorch @ `a0a261f` | MIT | 1.381.570 | chạy |
| `jt_graphformer_v2_ce` | JT-GraphFormer | ZhengYaolin/JT-GraphFormer @ `71fb81d` | MIT | 7.421.402 | chạy |
| `ds_gcn_v2_ce` | DS-GCN | davelailai/DS-GCN @ `ffc42f0` | **không có file LICENSE** | — | **BLOCKED_LICENSE / NOT_RUN** |

Đã tải từng repo đúng SHA vào thư mục tạm (không cài, không chạy mã tác giả), đọc model/config/LICENSE. Văn bản giấy phép và danh sách file port ở [THIRD_PARTY_NOTICES](../../training/modern_benchmark_duc/THIRD_PARTY_NOTICES.md). Không tải checkpoint, pretrained hay dữ liệu ngoài.

**DS-GCN:** repo chỉ ghi "generated based on pyskl", không có LICENSE; phần backbone riêng (`DGSTGCN` + `gcn_type='dgphgcn1'` trong `pyskl/models/gcns/utils/gcn.py`) là mã của tác giả DS-GCN chưa cấp quyền. Theo mục 6 của hợp đồng, Đức **không chép/port** mã này, ghi blocker và làm tiếp bốn model còn lại. Runner nhận ID `ds_gcn_v2_ce` nhưng dừng với `BLOCKED`, không fallback sang model khác. Cần điều phối quyết định ở `DUC-20261126-001`.

**DeGCN:** module không gian biến dạng DeSGC **không có trong bản công bố** (tác giả ghi "not included in the open-source release"). Bản port là đúng model công bố: CTR-GC cho không gian + DeTGC (temporal biến dạng) + hai stream. Báo cáo gọi là "DeGCN (released code)", không gọi là DeSGC của paper.

## 3. Phần giữ nguyên (y hệt A3)

- Bốn input `data/gmdcsa24/` đúng SHA-256 của lock 07; 160 clip, 81 ADL/79 Fall, 4 subject; `load_verified` kiểm hash, nhãn, mask, split.
- `make_loader` + `augment_pose` gốc, recipe tám kênh của A3, feature `v2-mask-visibility-raw-pose-augmentation`; augmentation chỉ ở train; loader A3 và đối thủ cho cùng tensor ở cùng chỉ số/RNG (test).
- AdamW lr 0,001, wd 0,0001, batch 16, tối đa 50 epoch inner, patience 10 (cải thiện > 1e-6), chọn epoch bằng CE inner không trọng số, refit bằng median ba best epoch; CE trọng số lớp `n/(2·count)`, Fall multiplier 1,0; không scheduler, label smoothing, AMP, pretrained, loss phụ.
- Nested LOSO 4 outer × 3 inner, seed 42/142/242; seed inner `seed+outer·1000+inner·10`, refit `seed+outer·1000+999`.
- Policy chính `fixed` ngưỡng 0,5 (`score >= 0,5`); `balanced`/`sensitivity` chọn bằng `select_thresholds` gốc trên inner OOF, cùng score.
- `metrics` gốc; F1 là F1 lớp Fall, không phải Macro-F1. Pooled 160 clip/seed rồi mean và SD mẫu (ddof=1).
- A3 chạy bằng nguyên `original.fit_model`. Vòng train của đối thủ (`fit_generic`) là bản sao chỉ khác hàm tạo model; test `test_generic_loop_matches_original_a3` chứng minh chạy với `SkeletonModel` cho lịch sử và trọng số **trùng bit** với `fit_model`.

## 4. Adaptation từng model (khóa trước outer-test)

Nguyên tắc: bắt đầu từ cấu hình tác giả, chỉ đổi phần bắt buộc vì shape/đầu vào/hợp đồng. Không thu nhỏ kiến trúc (không có bản "small"). Graph mọi model dùng đúng danh sách cạnh 33 khớp của A3; MediaPipe có ba thành phần liên thông (mặt 0–8, miệng 9–10, thân 11–32) và điều này được xử lý tường minh.

**SkateFormer** (cấu hình N-UCLA một người: depth 2-2-2-2, kênh 96/192/192/192, embed 96, 32 head, kernel 7, mlp_ratio 1, rel bias, index_t):
- Stem nhận 8 kênh theo công thức tác giả (C→2C→3C→96); embedding khớp 33 cột; mã hóa thời gian sin/cos với t = 0..31.
- Skate-MSA cần phần bằng nhau: sáu phần × sáu chỗ (tay phải, tay trái, chân phải, chân trái, mặt phải, mặt trái), **3 chỗ đệm**. Chỉ nhánh attention dùng bố cục 36; khóa đệm bị `-inf`, truy vấn đệm bị bỏ, mọi phép khác (G-Conv 33×33, T-Conv, LayerNorm, MLP, BN downsample, pooling) chạy trên 33 khớp thật. Test chứng minh đổi giá trị ở chỗ đệm không đổi output khớp thật.
- Kích thước phân vùng thời gian N = M = 4 (tác giả N-UCLA: 8) vì T = 32 giảm còn 4 ở stage cuối.

**BlockGCN** (10 lớp 128/256, stride ở lớp 5 và 8, BlockGC 8 head, mã hóa hop, Topo + TopoTrans mỗi lớp):
- Embedding Linear(8,128); một người (nhánh N-UCLA của tác giả, không repeat).
- Hop encoding: cặp khớp khác thành phần liên thông có bucket riêng (max hop + 1); mã gốc giả định đồ thị liên thông và sẽ gộp chúng với chính khớp đó.
- Topology lấy từ kênh 0–1 (vị trí tương đối x, y) của cùng mẫu đã augmentation, như tác giả dùng tọa độ khớp.
- Tác giả chuẩn hóa ma trận khoảng cách theo min/max **của cả batch**; ở đây theo từng clip để score test không phụ thuộc clip khác trong batch (test xác nhận).
- Không thêm dependency `torch_topological`: đồng điều bậc 0 của phức Vietoris–Rips = (sinh 0, chết = cạnh cây khung nhỏ nhất), tính chính xác bằng Prim; như mã tác giả, đám mây điểm là các hàng của ma trận khoảng cách đã chuẩn hóa. `StructureElementLayer` viết lại theo nguồn BSD-3, trả (B,64) cho mọi batch.
- `TopoTrans` dùng BatchNorm1d trên (B,C): batch 1 ở chế độ train không xác định (thiết kế tác giả). Kích thước train thật cho phần dư 0/5/11 theo batch 16, không bao giờ có batch 1 (test).

**DeGCN (released code)** (10 block 64/128/256, eta 4, hai stream, num_scale 4):
- Kiểm `in_channels == 3` của tác giả dùng để nhận block đầu (ST_GC, num_scale 1) được thay bằng cờ `first` cho 8 kênh; `base_frame = 32` để DeTGC kẹp chỉ số theo T thật.
- `gather` theo trục thời gian được thay bằng `index_select` tương đương (chỉ số chỉ đổi theo thời gian) để chạy nhanh trên CPU.
- Tác giả train mỗi head stream bằng một CE và chấm bằng tổng logit; ở đây model trả tổng logit và recipe chung áp **một** CE có trọng số lớp lên tổng đó.

**JT-GraphFormer** (8 block [[64,64,16],[64,64,16],[64,128,32],[128,128,32],[128,256,64],[256,256,64]×3], 3 head, kernel [3,5], use_pes, head fc như cấu hình NTU, không Koopman):
- num_channels 8, num_frames 32, len_parts 8 (giá trị N-UCLA; 32/8 = 4 nhóm), đồ thị khớp–quỹ đạo 33 × 8 = 264 nút.
- Ma trận Dijkstra = khoảng cách hop trên cạnh A3 + |i−j| giữa khung trong phần, đúng cách tác giả dựng; cặp không liên thông có trọng số tiên nghiệm exp(−∞) = 0 (tham số học `b`, `att0s` vẫn nối chúng).
- Tiên nghiệm là buffer thay vì `.cuda()` trong forward.

**Dropout (mọi model):** mọi vị trí dropout mà mã tác giả có tham số đều đặt p = 0,3; drop-path/stochastic depth = 0. SkateFormer: attention, projection/MLP, head. BlockGCN: head. DeGCN: head mỗi stream. JT-GraphFormer: attention, Dropout2d trước pooling, head. A3: như 07.

**Không thay:** không bỏ/ghép khớp về NTU/COCO, không bỏ visibility/motion, không thêm z/RGB, không bone/joint multi-stream, TTA, ensemble, pretrained hay loss phụ.

## 5. Thiết bị, ngân sách và thứ tự

- Máy Đức: AMD Ryzen 7 5800H (8 nhân/16 luồng), RAM 15,4 GB, Windows 11, **CPU float32**, không GPU (GB-20261119-002 cho phép full CPU). Python 3.12.10, torch 2.10.0+cpu, numpy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2 trong `.venv` riêng.
- Mọi run (smoke, full, A3 và đối thủ) dùng **cùng 4 luồng torch**; số luồng nằm trong lock. Có thể chạy hai làn song song, mỗi làn một tiến trình; thời gian train không phải chỉ số khoa học. Latency đo riêng trên máy rảnh.
- Một model: 3 seed × 4 outer × (3 inner + 1 refit) = 48 fit, 12 checkpoint outer, 480 dòng fixed / 1.440 dòng ba policy. Bốn đối thủ chạy được + A3 = **240 fit / 60 checkpoint**. DS-GCN 48 fit `NOT_RUN` (blocker).
- Thứ tự chốt trước outer-test: A3 → SkateFormer → BlockGCN → JT-GraphFormer → DeGCN (rẻ trước để có kết quả sớm). Model nào lỗi không chặn model khác.
- Ước tính từ đo một bước train (batch 16, 4 luồng, CPU đang bận Q0/Q1): A3 0,6 s; SkateFormer 4,4 s; BlockGCN 4,8 s; JT-GraphFormer 9,5 s; DeGCN 11,5 s. Track này cần cỡ 1,5–3 ngày CPU; đây là ước tính, không phải cam kết.

## 6. Kiểm tra bắt buộc trước full (runner thực thi)

1. `test_modern` PASS (21 test: shape/finite/backward batch 1 và 16, train/eval, độc lập batch, pose thiếu/visibility 0/khớp bị drop, padding SkateFormer, graph 33 khớp, không rò rỉ split, parity tensor, augmentation tất định, class weight, metric biên, parity vòng train với A3, reload checkpoint, resume/chữ ký, và test âm: sai hash input, thiếu fold, trùng ID, sai nhãn/subject, NaN, score khác giữa policy, sửa tay metric/score, thư mục rỗng, model lạ, DS-GCN bị chặn, full không có smoke). Báo cáo JSON gắn với hash code; runner full từ chối báo cáo khác code.
2. Smoke thật cho từng model: seed 42, outer subject 1, 3 inner + 1 refit, tối đa 2 epoch (4 fit / 1 checkpoint), `synthetic=false`, `scientific_result=false`. Không dùng score smoke để chọn gì.
3. Full chỉ chạy khi `--smoke-evidence` (cùng model/code/input/env/luồng) và `--test-report` hợp lệ; khóa ghi trước fit đầu tiên; mỗi fold được ghi vào thư mục tạm, kiểm lại (chọn epoch, ngưỡng, OOF, checkpoint SHA, nạp lại và chấm lại) rồi mới công bố; `completion.json` chỉ ghi khi đủ 12 fold/1.440 dòng và mọi CSV tái tính khớp.

## 7. Phân tích khóa trước kết quả

- Bảng chính: policy `fixed` 0,5 — Accuracy, F1 lớp Fall, Sensitivity, Specificity, Precision, FN, FP: mean ± SD (ddof=1) qua 3 seed, mỗi seed pooled 160 clip. CSV giữ tỷ lệ [0,1]; bảng trình bày %.
- Delta cặp so A3 **cùng seed** (điểm phần trăm) và số seed tốt hơn/kém hơn; delta theo subject; confusion; danh sách FN/FP và clip đổi kết quả so A3. Không p-value trên 3 seed của cùng 160 clip; ba seed không phải 480 mẫu độc lập.
- `balanced`/`sensitivity` chỉ là bảng phụ. Latency model-only batch 1, FP32, eval/no-grad, 30 warmup + 200 lần, median/p95, số tham số; đo lại A3 và các model trên cùng máy.
- Kết quả kém A3 vẫn nộp đầy đủ. Không đổi cấu hình, chọn seed hay tinh chỉnh sau khi xem outer. Mã phân tích (`training/modern_benchmark_duc/tools/`, ngoài khóa train) được khóa hash bằng addendum trước khi có kết quả full.

## 8. Giới hạn

Bốn người, 160 clip; outer fold đã được xem trong phát triển 06/07/08, không phải test mù. CPU float32 có thể lệch nhỏ so với GPU. Một recipe chung không phải recipe tối ưu của từng paper (lr, scheduler, label smoothing, epoch, augmentation riêng đều bị bỏ). DeGCN thiếu DeSGC chưa công bố; BlockGCN đổi chuẩn hóa batch thành theo clip; SkateFormer có đệm 3 chỗ; DS-GCN chưa chạy vì giấy phép.

## 9. Nộp sản phẩm

`results/modern_benchmark_duc/<run-id>/`: `tests/`, `smoke/<model>/`, `full/<model>/` (lock, lịch sử 36 inner + 12 refit, raw inner OOF, `predictions.csv`, `fold_metrics.csv`, `pooled_by_seed.csv`, `seed_mean_std.csv`, `errors.csv`, `completion.json`, `models/manifest.json`), `audit/`, `latency/`. Checkpoint lớn (tổng ~0,6 GB) đưa lên GitHub Release asset với manifest SHA-256 và lệnh tải/nạp đã kiểm; file `.pt` không commit vào Git.

## Addendum A1 — 26/11/2026, trước mọi kết quả full

- Smoke thật: A3 và SkateFormer PASS (mỗi model 4 fit, 96 dòng, reload PASS); BlockGCN, JT-GraphFormer, DeGCN đang chạy. Full chỉ bắt đầu khi smoke cùng model/code/env có `completion.json` hợp lệ (runner kiểm).
- Khóa mã phân tích ngoài khóa train (SHA-256 sau CRLF→LF): `tools/audit.py` `6f8f5d61bf92d3604b47ffe20d6a001c8020016305d54935b2240fab252d2d61`, `tools/latency.py` `8af6d8df54588c65642d25829f610bde4a440df79a145518eae468ef7d8a3ac0`, `tools/run_lane.sh` `d253e0e8fc60e39f12159ef14651a875f8036aefe818d8fc44b41032c9c265f2`. Audit chỉ được tự kiểm trên bản sao smoke (kiểm phần mềm, không phải kết quả). Nếu phải sửa lỗi phần mềm sau khi có kết quả, giữ bản cũ, ghi lý do và không đổi bảng/quy tắc ở mục 7.
- Hai làn chạy, mỗi tiến trình 4 luồng: làn A `A3 → SkateFormer → BlockGCN`, làn B `JT-GraphFormer → DeGCN` (thứ tự mục 5 giữ nguyên trong từng làn). CPU còn chia với run full Q0/Q1 tới khi run đó xong; thời gian train vì vậy không dùng để so tốc độ.
