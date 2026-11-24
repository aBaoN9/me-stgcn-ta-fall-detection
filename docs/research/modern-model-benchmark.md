# Benchmark 10 mô hình hiện đại với ME-STGCN-TA

Quyết định điều phối: **GB-20261124-001**, ngày 24/11/2026, UTC+7.
Trạng thái bàn giao: **ASSIGNED / IMPLEMENTATION_REQUIRED / TRAIN_NOT_RUN**.
Đây là lệnh giao triển khai và train cho Tùng/Đức, không chỉ là danh sách tham khảo.
Chưa có runner chung cho 10 mô hình; không dùng lệnh runner cũ rồi tuyên bố đã train mô hình mới.

## 1. Phân công chính thức

| STT | Người / nhánh | Mô hình | ID thí nghiệm mới | Trạng thái ban đầu |
|---|---|---|---|---|
| 1 | Tùng / `dinhtung` | CascadeFormer | `cascadeformer_v2_ce` | NOT_RUN |
| 2 | Tùng / `dinhtung` | DHA-eGCN | `dha_egcn_v2_ce` | NOT_RUN |
| 3 | Tùng / `dinhtung` | Hyper-GCN | `hyper_gcn_v2_ce` | NOT_RUN |
| 4 | Tùng / `dinhtung` | ProtoGCN | `proto_gcn_v2_ce` | NOT_RUN |
| 5 | Tùng / `dinhtung` | InfoGCN++ | `infogcnpp_v2_ce` | NOT_RUN |
| 6 | Đức / `vinhduc` | SkateFormer | `skateformer_v2_ce` | NOT_RUN |
| 7 | Đức / `vinhduc` | BlockGCN | `block_gcn_v2_ce` | NOT_RUN |
| 8 | Đức / `vinhduc` | DeGCN | `de_gcn_v2_ce` | NOT_RUN |
| 9 | Đức / `vinhduc` | JT-GraphFormer | `jt_graphformer_v2_ce` | NOT_RUN |
| 10 | Đức / `vinhduc` | DS-GCN | `ds_gcn_v2_ce` | NOT_RUN |

- **Cả hai tự thiết lập, triển khai adapter, test, smoke thật, khóa M0 rồi full train**, không chờ duyệt từng bước. AI hỗ trợ nhưng phải kiểm bằng lệnh và artifact thật.
- Nhiệm vụ mới mở rộng quyền của Đức sang năm backbone trên; câu cũ "Đức không được thay backbone" không áp dụng track này. Bài Q0/Q1 và kết quả đã có giữ riêng, không xóa hoặc gọi là kết quả của track mới.
- Tùng đã hoàn thành các benchmark trước; đây là đợt nghiên cứu mới, không thu hồi nghiệm thu cũ. SkateFormer adapted small của Tùng không thay cho bản mới của Đức. Không đổi tên checkpoint cũ thành sản phẩm mới.
- Hạn bài cũ 25/11/2026 giữ nguyên. Chưa chốt hạn riêng cho toàn bộ 10 mô hình mới; cập nhật phần đã làm, thời gian dự kiến và blocker ở mốc 25/11. Không tự hứa đủ 10 mô hình trong một ngày hoặc cắt folds/seeds để kịp.
- Thứ tự triển khai trong năm model của mình được tự chọn trước khi xem outer-test. Model bị blocker không ngăn làm các model còn lại; đổi tên mô hình được giao cần quyết định mới, không tự thay bằng mô hình dễ hơn.

## 2. Vùng làm việc, không đụng nhau

| Người | Mã / notebook / tests mới | Kết quả mới | Thiết kế và báo cáo |
|---|---|---|---|
| Tùng | `training/modern_benchmark_tung/` | `results/modern_benchmark_tung/<run-id>/` | `docs/research/modern-benchmark-tung.md`, báo cáo/trao đổi `dinhtung.md` |
| Đức | `training/modern_benchmark_duc/` | `results/modern_benchmark_duc/<run-id>/` | `docs/research/modern-benchmark-duc.md`, báo cáo/trao đổi `vinhduc.md` |

Các thư mục mới là **phạm vi được phép tạo**, chưa phải code đã tồn tại. Được import helper chung; không sửa `training/component_ablation/`, runner benchmark cũ, checker, input, artifact đã khóa, notebook gốc, demo, deployment hoặc Word. Nếu cần helper mới thì đặt trong vùng riêng. Không sửa/push nhánh của bạn kia hoặc tự merge main.

## 3. Hợp đồng đầu vào bắt buộc

Mốc: **`A3_motion_attention` của notebook 07**, `SkeletonModel(channels=8, attention=True, dropout=0.3)`, Fall multiplier 1.0. Không dùng model final 06, A3_cost_225, Q1 hoặc bản tuning chưa nghiệm thu làm mốc thay thế.

| Thuộc tính | Cố định cho cả 10 model và A3 |
|---|---|
| Dữ liệu | 160 clip GMDCSA-24, 4 subject; 81 ADL / 79 Fall |
| Nhãn | `0=ADL`, `1=Fall`; nguyên `sample_index`, `video_id`, `subject` |
| Nguồn | Bốn file trong [data/gmdcsa24](../../data/README.md), SHA-256 theo [lock 07](../../results/component_ablation/locked_protocol.json) |
| Feature version | `v2-mask-visibility-raw-pose-augmentation` |
| Tensor trước backbone | float32 hữu hạn, **`(N, 8, 32, 33)`** |
| Thời gian / người | 32 mốc của cùng clip, 33 khớp MediaPipe, một người |
| Kênh theo thứ tự | relative_x, relative_y, visibility, joint_velocity_x, joint_velocity_y, torso_angle, hip_displacement_x, hip_displacement_y |
| Tạo đặc trưng | `make_features_v2` và `make_loader` gốc với recipe A3 tám kênh |
| Augmentation | Cùng `augment_pose` trên raw pose trước khi tính feature; chỉ train |
| Output | Logits `(N,2)`; `score = softmax(logits, dim=1)[:,1]` |

**Không lấy trực tiếp tensor feature lưu trữ cũ làm v2**: dùng `load_verified` và loader gốc để tái tạo v2 từ pose/mask/duration. Tham khảo [features.py](../../training/component_ablation/features.py), [models.py](../../training/component_ablation/models.py), [runner.py](../../training/backbone_benchmark/runner.py).

Adapter chỉ được đổi layout, thêm trục người `M=1`, sửa input projection nhận đủ tám kênh và classifier hai lớp, graph/embedding 33 khớp, temporal shape 32. Được padding nội bộ nếu thuật toán cần chia block, nhưng phải mask padding trong attention/pooling và công bố, không biến nó thành quan sát mới. Không bỏ khớp về NTU 25/COCO 17, bỏ visibility/motion, thêm z/RGB, đổi YOLO/MediaPipe, lấy lại frame hoặc tạo split mới.

Không dùng nguyên loader NTU/N-UCLA, bone/joint multi-stream, test-time augmentation hoặc ensemble của repo tác giả. Graph/hyperedge phải định nghĩa trên đúng 33 khớp; không âm thầm dùng adjacency NTU. BatchNorm và mọi thống kê học chỉ cập nhật từ train; không normalize bằng cả tập. Kiểm parity tensor A3 và đối thủ trên cùng chỉ số mẫu và cùng trạng thái RNG.

## 4. Recipe chung và giới hạn của phép so sánh

**Bảng chính là controlled-backbone / CE-adapted**, train từ đầu, một cấu hình cố định mỗi model, không search. Đây không phải tái lập đầy đủ score/recipe paper và không chứng minh phương pháp gốc tốt/kém hơn phổ quát.

| Thành phần | Giá trị chung |
|---|---|
| Optimizer | AdamW; learning rate `0.001`, weight decay `0.0001` |
| Batch | 16, không tự giảm riêng model vì OOM |
| Epoch / early stop | Tối đa 50 epoch inner, patience 10; cải thiện loss > `1e-6` |
| Chọn epoch | Minimum **unweighted inner-validation cross entropy**, median của ba best epoch để refit |
| Loss train | Cross-entropy; weight lớp = `n_train / (2 * count_class_train)`, Fall multiplier `1.0` |
| Dropout | `0.3` cho các vị trí dropout áp dụng; ghi các vị trí, drop-path/stochastic-depth tắt |
| Scheduler / smoothing | Không scheduler, không label smoothing |
| Precision | FP32, không AMP; ghi thiết lập TF32/determinism, giữ giống A3 trong mỗi môi trường |
| Dữ liệu bổ sung | Không pretrained, pretraining tự giám sát, teacher hoặc dữ liệu ngoài |
| Loss bổ sung | Không auxiliary/prototype/contrastive/future-prediction loss trong bảng CE chung |

Import [default_config, fit_model, select_thresholds, metrics](../../training/component_ablation/training.py) để tái lập mốc; tham khảo fit/lock/reload/collect của runner benchmark. Không sửa helper gốc để phù hợp model mới. Khóa architecture/depth/width/heads/graph trước outer-test; dùng cấu hình tác giả làm điểm xuất phát, chỉ thay phần cần cho shape. Nếu thu nhỏ kiến trúc phải đặt tên **adapted small**, nêu rõ, không giả là bản gốc.

Một số phương pháp phụ thuộc loss/pretraining riêng. Nếu bỏ thành phần đó làm thay đổi bản chất, phải ghi rõ "backbone adapted, CE-only"; nếu forward không khả thi thì ghi blocker, không tạo model khác gắn cùng tên. Thử recipe đầy đủ của paper là **track phụ chưa được đưa vào bảng chung**: cần đăng ký ngân sách công bằng và được điều phối chốt trước chạy; không so với A3 untuned rồi kết luận về kiến trúc. Không trộn kết quả tuning A3 sáu recipe với mười model một recipe.

OOM: không tự giảm batch/frame/joint hay đổi precision cho một model. Thử môi trường có đủ RAM/VRAM được phép hoặc CPU; nếu cần recipe chung khác, báo điều phối để chốt track riêng và chạy lại A3 cùng điều kiện. Không mua GPU/dịch vụ hoặc bỏ qua guard.

## 5. Split, seeds và thang đánh giá

1. Bốn outer subject; mỗi outer giữ nguyên toàn bộ người test. Ba người còn lại tạo ba inner folds: hai train, một validation. CSV hiện có không thay cho ba inner này.
2. Seeds **42, 142, 242**. Seed inner = `seed + outer*1000 + inner_subject*10`; refit = `seed + outer*1000 + 999`, giống runner chuẩn.
3. Chọn epoch từ inner, refit toàn bộ ba người outer-train từ khởi tạo mới, rồi mới đánh giá outer-test. Mọi model/prototype/teacher/state phải reset giữa các fit; không dùng nhãn hoặc embedding outer-test để học.
4. Primary policy **`fixed`, threshold `0.5`, prediction = score >= threshold**. `balanced` và `sensitivity` chỉ là bảng phụ, dùng đúng `select_thresholds` gốc trên inner OOF, cùng score với fixed. Grid 37 ngưỡng từ 0.05 đến 0.95; target sensitivity 0.95, minimum specificity 0.75, tie-break theo helper.
5. Dùng đúng `metrics` gốc: Accuracy, Balanced Accuracy, Precision, Sensitivity/Recall, Specificity, **F1 lớp Fall**, F2, TP/TN/FP/FN và clip false-positive rate. **F1 hiện hành không phải Macro-F1**. Nếu thêm Macro-F1 phải là cột phụ, không thay cột F1 cũ.
6. Mỗi seed pooled OOF đủ 160 clip; sau đó mean và **sample SD, ddof=1** qua ba seed. Không lấy trung bình bốn fold thay pooled, không báo seed tốt nhất. CSV tỷ lệ giữ `[0,1]`; bảng trình bày `%`; chênh lệch dùng điểm phần trăm. Giữ metric không xác định là null theo helper, không âm thầm đổi quy ước.
7. Báo từng subject, confusion matrix, danh sách FN/FP và paired delta so A3 của cùng seed. Ba seed vẫn là cùng 160 video, không phải 480 mẫu độc lập. Bốn subject và outer-test đã quan sát là giới hạn bắt buộc.

### A3 đối chứng và ngân sách

Mỗi người chạy **một bộ A3 full tham chiếu** cùng code loader/recipe/environment/device với năm ứng viên; không cần train A3 riêng cho từng ứng viên nếu lock môi trường giống hệt. Nếu dependency model buộc tạo môi trường khác, phải chạy A3 đối chứng tương ứng môi trường đó và ghi tách run. Không chỉ chép số 89,38% hoặc 89,58% lịch sử làm đối chứng mới.

- Một model: 3 seeds x 4 outer x (3 inner + 1 refit) = **48 fits**, **12 checkpoint outer**.
- Năm đối thủ + một A3 mỗi người: **288 fits / 72 checkpoint outer**.
- Toàn đợt tối thiểu: **576 fits / 144 checkpoint outer**, gồm hai bộ A3 kiểm soát. Smoke và môi trường bổ sung chưa tính.
- Một model full: 480 dòng OOF cho fixed; **1.440 dòng** khi có đủ ba policy. Mỗi `(variant,seed,policy)` có đúng 160 ID, mỗi khóa mẫu duy nhất.
- Báo A3-Tùng và A3-Đức riêng nếu môi trường khác; không lấy trung bình chúng thành mốc thứ tư hoặc chọn bộ có lợi. Bảng chung giữ provenance/run ID; latency chỉ xếp cạnh nhau sau đo trên cùng máy.

## 6. Nguồn để AI đọc, pin trước triển khai

Các URL/commit dưới được kiểm qua GitHub API ngày 23/11/2026; có source Python, **chưa kiểm chạy, độ đầy đủ hay tính đúng của toàn bộ implementation**. SHA là snapshot khởi đầu, không phải checkpoint được phép dùng để pretrain. Trước M0, đọc paper, cấu hình và giấy phép; ghi source SHA thực dùng, các patch và license/NOTICE. Nếu metadata license trống thì kiểm LICENSE và nguồn từng file; chưa rõ quyền thì không chép/phân phối code, báo blocker và tiếp tục model khác.

| Mô hình | Repo nguồn | Commit snapshot | Lưu ý triển khai |
|---|---|---|---|
| CascadeFormer | [Yusen-Peng/CascadeFormer](https://github.com/Yusen-Peng/CascadeFormer) | `847efac9bec015d2d1d8f2104d255a0258ad76d7` | Không tải checkpoint/pretrain; bản CE từ đầu phải công bố khác pipeline pretrain của paper. Metadata license chưa rõ. |
| DHA-eGCN | [oskar-ika-adi-nugroho/DHA-eGCN](https://github.com/oskar-ika-adi-nugroho/DHA-eGCN) | `307b5086175964d7781c7e070483c5f4c0e0c7ce` | Chốt một biến thể graph; không dùng RICH4/RICH5 ensemble. Apache-2.0 theo metadata. |
| Hyper-GCN | [6UOOON9/Hyper-GCN](https://github.com/6UOOON9/Hyper-GCN) | `cefe70b9c62ab825722cc8ece3fa26ecf5c334a1` | Không nhầm HyperGCN 2019; ánh xạ part/hyperedge 33 khớp. Metadata license chưa rõ. |
| ProtoGCN | [firework8/ProtoGCN](https://github.com/firework8/ProtoGCN) | `ddf7f274f9f5d9e45a2fcfeb299bfb3fd7c2303d` | Prototype chỉ học từ train; loss ngoài CE không thuộc bảng chính. MIT theo metadata. |
| InfoGCN++ | [stnoah1/infogcn2](https://github.com/stnoah1/infogcn2) | `8c44085ac134f35dc85f683dcc775fdb43172ad1` | Là `infogcn2`, không InfoGCN gốc. Đánh giá toàn clip 32 mốc, không so prefix accuracy với clip accuracy; không gọi online benchmark. Metadata license chưa rõ. |
| SkateFormer | [KAIST-VICLab/SkateFormer](https://github.com/KAIST-VICLab/SkateFormer) | `0249e19d96b33e9150760129bbede56464126ae7` | Pin SHA, không dựa tên default branch; kiểm partition/embedding 33 khớp và padding. MIT theo metadata. |
| BlockGCN | [ZhouYuxuanYX/BlockGCN](https://github.com/ZhouYuxuanYX/BlockGCN) | `7b45971a931c5b0ba60f3a7ba11a91e544de54fb` | Kiểm topology/block index MediaPipe, không giữ NTU graph. Apache-2.0 theo metadata. |
| DeGCN | [WoominM/DeGCN_pytorch](https://github.com/WoominM/DeGCN_pytorch) | `a0a261fbf4369095083a7b22fa371fb9448f69a8` | Kiểm toán tử deformable và tương thích môi trường trước full. MIT theo metadata. |
| JT-GraphFormer | [ZhengYaolin/JT-GraphFormer](https://github.com/ZhengYaolin/JT-GraphFormer) | `71fb81d177c151f029403cff9b7f2b0d93f8e01c` | Kiểm joint-trajectory graph 32x33, không dùng NTU preprocessing. MIT theo metadata; xác minh metadata bài báo khi viết related work. |
| DS-GCN | [davelailai/DS-GCN](https://github.com/davelailai/DS-GCN) | `ffc42f0f5ddd5fd7e06cb2d7bd7e6c5a34fec6ed` | Nguồn dựa pyskl/MMCV; không chạy nguyên pipeline NTU. Metadata license chưa rõ. |

Không chép mù quáng `pip install -r` của cả mười repo vào cùng env. Nguồn có hệ dependency khác nhau, ví dụ Hyper-GCN ghi Torch 1.13, InfoGCN++ có Apex/torchdiffeq, DS-GCN dựa pyskl. Ưu tiên port module cần thiết vào env chuẩn với test; nếu không thể, env riêng và A3 cùng env. Tên hội nghị/năm trong danh sách chat không thay metadata paper đã kiểm; bảng này giao model theo repo/SHA, không tuyên bố top-10 SOTA.

## 7. Thiết lập local từ đầu

Từ bản clone của mình, kiểm sạch worktree trước merge. Chưa clone thì:

```bash
git clone https://github.com/aBaoN9/me-stgcn-ta-fall-detection.git
cd me-stgcn-ta-fall-detection
git fetch origin
git status --short
git branch -a
git show origin/main:docs/discussions/answers.md
```

Tùng dùng nhánh `dinhtung`, Đức dùng `vinhduc`. Có local thì `git switch <ten-nhanh>`; chưa có local nhưng có remote thì `git switch --track origin/<ten-nhanh>`. Không chạy nguyên placeholder. Sau khi lưu bài hiện tại và worktree sạch, `git merge origin/main`; không reset/stash/force-push để né xung đột.

Ví dụ PowerShell, Python 3.11, env nghiên cứu riêng:

```powershell
py -3.11 -m venv .venv-modern
.venv-modern/Scripts/python -m pip install --upgrade pip
.venv-modern/Scripts/python -m pip install -r training/component_ablation/requirements.txt
.venv-modern/Scripts/python -c "import sys, torch, numpy, pandas, sklearn; print(sys.version); print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(numpy.__version__, pandas.__version__, sklearn.__version__)"
.venv-modern/Scripts/python scripts/training_inputs.py --input-root data/gmdcsa24
.venv-modern/Scripts/python -m training.component_ablation.validate_pipeline --input-root data/gmdcsa24
.venv-modern/Scripts/python -B -X utf8 scripts/check_repository.py
.venv-modern/Scripts/python -B -m training.backbone_benchmark.test_benchmark
```

Linux: `python3.11 -m venv .venv-modern`, thay executable bằng `.venv-modern/bin/python`. Lệnh pip trên là điểm bắt đầu theo requirements repo, **không bảo đảm wheel GPU phù hợp driver**. Muốn CUDA: kiểm `nvidia-smi`, chọn wheel Torch cùng phiên bản từ [trang cài chính thức](https://pytorch.org/get-started/locally/) phù hợp driver trong env riêng; kiểm `torch.cuda.is_available()` và tên GPU trước smoke. Không cài lại driver/CUDA hệ thống. CPU được phép nhưng runner riêng phải hỗ trợ và khóa rõ, không xóa guard runner cũ.

Thiếu input thì dùng `scripts/training_inputs.py --input-root data/gmdcsa24 --download`; sai hash thì dừng, không sửa hash chuẩn. Sau thêm dependencies model, chạy lại kiểm tra. Lưu `pip freeze`, Python/Torch/CUDA/driver, platform, GPU/CPU, thread count và code SHA vào artifact run. Không cần MediaPipe hoặc video để train benchmark này.

## 8. Kaggle

1. Notebook mới của cá nhân, GPU miễn phí có quyền dùng; không sửa notebook gốc đã khóa. Giữ môi trường Kaggle nếu tương thích, ghi phiên bản thực. Nếu khác env chuẩn thì A3 phải chạy trong cùng env đó.
2. Đưa code đã commit trên nhánh mình vào notebook, checkout **SHA bất biến** có adapter/runner/tests mới. Không dùng HEAD di động cho run đã khóa. Có mạng thì clone repo; không có mạng thì chuẩn bị code/dependencies trước.
3. Input: dùng `data/gmdcsa24` trong checkout hoặc gắn đúng bốn file đã publish vào `/kaggle/input`; truyền đường dẫn thật. Chạy input guard và validate_pipeline trong chính runtime.
4. Đặt output mới dưới `/kaggle/working/<owner>-<model>-<run-id>`, một writer mỗi output. Chạy test, smoke thật rồi full bằng entrypoint do mình triển khai. Không dùng CLI TCN–A3 để giả làm runner mười model.
5. Sao lưu từng fold đã hoàn tất, lock/config, lịch sử và checkpoint; dữ liệu `/kaggle/working` có thể mất khi phiên kết thúc. Resume chỉ với cùng signature, kiểm hash/reload các fold đã xong; đổi code/input/env/config thì run mới.
6. Export notebook mới vào vùng training của mình, pin CODE_SHA/INPUT_ROOT/OUTPUT và ghi lệnh exact trong README. Không đưa token hoặc thông tin đăng nhập vào notebook/log/repo.

Xem thêm [hướng dẫn input và Kaggle](../training-start.md). Không có GPU/quota hoặc dependency chưa chạy là blocker thực tế; AI không tự cấp được tài nguyên.

## 9. Hợp đồng runner để AI triển khai

Mỗi người triển khai package của mình, giữ cùng logic chuẩn. README mới phải có **lệnh thực đã kiểm** cho test, smoke, full, resume, audit/reload và latency. Các interface dưới là yêu cầu phải xây dựng, chưa phải lệnh có sẵn:

```text
python -m training.modern_benchmark_tung.runner --help
python -m training.modern_benchmark_duc.runner --help
required CLI: --model --input-root --output --mode smoke|full --device cpu|cuda
full guard: --smoke-evidence <real-smoke-output>
resume: --resume (identical signature only)
```

Không chạy cả hai package; mỗi người làm package của mình. `--model` phải hỗ trợ đúng năm ID được giao và `A3_motion_attention`; model không tồn tại phải FAIL, không fallback về A3.

Trước full từng model: test shape/finite/backward với batch 1 và 16; train/eval mode; missing pose/visibility/padding; no split leakage; tensor parity; class weights; deterministic augmentation; metric edge cases; checkpoint reload; resume/signature; OOF coverage. Test âm phải bắt input sai hash, thiếu fold/seed/policy, trùng ID, sai nhãn/subject, NaN, score khác giữa các policy hoặc metric sửa tay. Không tạo completion thành công từ một thư mục trống.

Smoke thật: một seed 42, một outer subject 1, đủ ba inner + một refit, tối đa hai epoch inner; một model **4 fits / 1 checkpoint outer**. Chạy cả A3 và từng ứng viên, cùng input/code/env/config liên quan với full. Smoke ghi `synthetic=false`, `scientific_result=false`, mode smoke; không dùng outer smoke score để chọn kiến trúc. Smoke FAIL phải sửa/test lại trước full; thay code cần lock và smoke mới, không sửa lock cũ.

Full phải khóa cấu hình **trước fit**, kiểm input/hash/test/smoke, dùng đủ ba seed/bốn outer, kiểm reload mỗi checkpoint, lưu dự đoán và audit trước `complete=true`. Model A3 dùng nguyên `SkeletonModel`/`fit_model` chuẩn; đối chiếu parity thay vì viết một A3 khác.

## 10. Nộp sản phẩm

Mỗi model có README, upstream SHA/license/patch notes, `locked_protocol.json`, hash input/code/config/environment, history của đủ 36 inner + 12 refit, 12 checkpoint outer load được, model manifest SHA-256, `predictions.csv`, `fold_metrics.csv`, `pooled_by_seed.csv`, `seed_mean_std.csv`, `errors.csv`, `completion.json`, log test/smoke/audit. Lưu raw inner OOF để tái kiểm epoch/ngưỡng; inner checkpoints có thể bổ sung nhưng không thay 12 checkpoint outer.

Schema predictions giữ nguyên:

```text
sample_index,video_id,subject,label,score,prediction,threshold,variant,seed,policy
```

Completion lưu expected/actual fits, outer, seeds, policies, model IDs; thiếu phần nào phải PARTIAL/NOT_RUN, không điền số kỳ vọng vào số đã chạy. Kết quả không thắng A3 vẫn nộp đầy đủ. Không lấy bảng paper/ảnh chụp thay dự đoán thật.

Latency: tất cả model và A3 đo lại trên **cùng máy**, FP32, batch 1, eval/no-grad, 30 warmup + 200 lần đo, CUDA synchronize đúng thiết bị; median/p95 milliseconds, parameters và peak memory khi có. Tách model-only khỏi preprocessing/end-to-end. Hai máy khác nhau thì hai bảng, không xếp hạng tốc độ chung.

Commit/push đích danh file trong vùng được giao, cập nhật báo cáo cá nhân và mở PR. Checkpoint lớn dùng Release asset theo [quyết định nộp sản phẩm](../discussions/answers.md), manifest có URL/dung lượng/hash và lệnh tải/load đã kiểm; không upload video, secrets, env hoặc source bên thứ ba chưa rõ quyền. Chỉ stage checkpoint nhỏ đích danh nếu phù hợp Git, không `git add -f .`. Không tự merge main hoặc cập nhật demo/deployment.

## 11. Prompt bàn giao cho AI của hai bạn

```text
Tôi là Tùng (dinhtung) / Đức (vinhduc); chọn đúng người đang làm việc.
Đọc AGENTS.md, origin/main:docs/discussions/answers.md, nhiệm vụ cá nhân và
docs/research/modern-model-benchmark.md, quyết định GB-20261124-001.
Triển khai và TRAIN năm model được giao trong vùng riêng, không dừng ở kế hoạch.
Trước hết kiểm nhánh/worktree, input SHA-256 và nguồn paper/code/license.
Giữ tensor v2 (N,8,32,33), nhãn/split/seeds/recipe/metric y hệt hợp đồng.
Lập M0 và ghi rõ CE-adapted khác paper; không dùng pretrained hoặc tự đổi mô hình.
Dùng helper chuẩn, thêm test, smoke thật và lock trước full; được tự chạy khi PASS,
không chờ Gia Bảo duyệt từng bước. Chạy A3 đối chứng cùng môi trường.
Nộp checkpoint, raw OOF, metric đúng F1-Fall/fixed 0.5, hash/log/audit và PR.
Không sửa nguồn/kết quả cũ, không bịa CLI/số liệu, không coi quyền train là đã train.
Nếu có blocker, ghi bằng chứng và tiếp tục model độc lập, không âm thầm đổi hợp đồng.
Xác nhận đã nhận GB-20261124-001 trong file trao đổi của mình và push đúng nhánh.
```

M0, test/smoke/full và xác nhận của từng người hiện **chưa được quan sát** ở lần giao việc này. Publish tài liệu không có nghĩa họ đã đọc hoặc bắt đầu chạy.
