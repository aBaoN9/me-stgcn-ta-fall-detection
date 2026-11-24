# Trả lời và quyết định của người điều phối

File do người điều phối cập nhật trên `main`. Thành viên fetch và đọc `origin/main:docs/discussions/answers.md`, không sửa file này trên nhánh cá nhân để tự cấp quyền cho mình.

## Snapshot

- **Hiện hành 24/11/2026:** `GB-20261124-001` giao Tùng năm model đầu, Đức năm model sau; triển khai và train với đầu vào/recipe/đánh giá chung A3. Đây là mở rộng nhiệm vụ mới, thay hạn chế Đức không đổi backbone cho track này. Xem [hướng dẫn đầy đủ](../research/modern-model-benchmark.md); các snapshot trước là lịch sử, không dùng để chặn nhiệm vụ mới.

- Cập nhật 23/11/2026: nghiệm thu/merge SkateFormer adapted small qua PR #4; tích hợp demo FallScope theo yêu cầu mới Gia Bảo, thay trạng thái hoãn demo nhưng không khôi phục PoseLab/giao lại WEB-01/WEB-02. Tuning local vẫn tách khỏi lần publish này; deployment/bài/hạn Đức giữ nguyên. Chi tiết `GB-20261123-001` dưới đây; các snapshot có ngày cũ là lịch sử.

- Phạm vi mới 21/11/2026: gỡ mã PoseLab, demo mới để sau; kế hoạch tuning A3 so TCN/GRU chỉ `PLANNED / NOT_RUN` theo `GB-20261121-002`. Quyền tự chủ/bài/hạn Đức và bằng chứng đã nhận của Tùng không đổi.
- Cập nhật 21/11/2026: nghiệm thu bổ sung GRU/CTR-GCN adapted của Tùng qua PR #3, tích hợp bản sửa monitor và cập nhật README theo `GB-20261121-001`. Quyền tự chủ, hạn bài Đức, model triển khai, trạng thái tắt PoseLab và Word giữ nguyên.
- Cập nhật: 20/11/2026, đã nghiệm thu benchmark TCN–A3 của Tùng và merge PR #2 theo `GB-20261120-001`. Quyền tự chủ/CPU và hỗ trợ input vẫn giữ; Đức tiếp tục nhiệm vụ/hạn 25/11, không suy ra đã nghiệm thu bài Đức. PoseLab vẫn dừng, model triển khai không đổi.
- Tùng tự đề xuất mô hình/phương pháp khác trên cùng dataset; Đức thay đổi có kiểm soát ME-STGCN-TA hiện tại. Cùng giao thức đánh giá, không áp sẵn CTR-GCN hoặc quality-aware.
- Cả hai không chờ duyệt M0/chọn phương án/kiến trúc/review lại trước full trong nhiệm vụ; tự ghi và khóa phương án, sửa lỗi và kiểm tra đạt. Quyền chạy không phải nghiệm thu code/kết quả.
- Chưa cấp quyền đổi dataset/giao thức chung, tiêu tiền GPU hoặc thay model/backend triển khai. Thiếu input/quyền/tài nguyên thì báo blocker thật, không chờ duyệt kỹ thuật đã được cấp quyền.
- Quy trình chung: [kênh trao đổi](README.md).

## GB-20261124-001 - Giao 10 model, cùng đầu vào và thang đánh giá A3

- Nguồn: yêu cầu trực tiếp của Gia Bảo giao Tùng năm model đầu, Đức năm model sau, hướng dẫn thiết lập và train, bắt buộc cùng đầu vào và đánh giá với mô hình đề xuất. Publish tài liệu điều phối trên main theo yêu cầu; không push hộ nhánh cá nhân.
- Quyết định: **APPROVED_TO_IMPLEMENT_AND_TRAIN / AUTONOMOUS**. Đây là giao sản phẩm gồm code + model đã train + kết quả thật, không dừng ở khảo sát. Tự đọc nguồn/license, khóa M0, test/smoke thật rồi full; không chờ duyệt từng bước. Đợt mới hiện `NOT_RUN`, không tuyên bố đã chạy thay thành viên.
- **Tùng / dinhtung:** CascadeFormer, DHA-eGCN, Hyper-GCN, ProtoGCN, InfoGCN++. Vùng mới `training/modern_benchmark_tung/`, `results/modern_benchmark_tung/`, `docs/research/modern-benchmark-tung.md` và báo cáo/trao đổi cá nhân.
- **Đức / vinhduc:** SkateFormer, BlockGCN, DeGCN, JT-GraphFormer, DS-GCN. Vùng mới `training/modern_benchmark_duc/`, `results/modern_benchmark_duc/`, `docs/research/modern-benchmark-duc.md` và báo cáo/trao đổi cá nhân.
- **Chuẩn bắt buộc:** [hợp đồng benchmark](../research/modern-model-benchmark.md), bốn input đúng hash, feature v2 `(N,8,32,33)`, cùng loader/augmentation, A3_motion_attention của 07, recipe CE cố định từ đầu, nested LOSO 4 outer/3 inner, seeds 42/142/242. Bảng chính fixed 0.5, F1 lớp Fall, pooled 160 clip/seed rồi mean/SD `ddof=1`; giữ đầy đủ FN/FP và raw OOF. Không thay đầu vào/pretrained/ensemble để lấy điểm cao.
- Mỗi người chạy A3 tham chiếu cùng môi trường với năm model. Dự kiến tối thiểu 288 fits và 72 checkpoint outer/người, tổng 576 fits/144 checkpoint, chưa tính smoke/môi trường bổ sung. Đây là ngân sách thiết kế, không là số đã chạy. Bảng chính là **CE-adapted**, không tự gọi tái lập đầy đủ paper khi đã bỏ loss/pretraining riêng.
- **Thay thế phạm vi:** quy tắc cũ không giao sẵn model và Đức không được thay backbone không áp dụng đợt này. Giữ bài contribution và nghiệm thu Tùng trước đây nguyên trạng; không yêu cầu sửa kết quả cũ. SkateFormer mới của Đức không được thay bằng checkpoint adapted small của Tùng. Không đổi demo/deployment, notebook/Word hoặc tuning track của Gia Bảo.
- Hạn bài cũ **25/11/2026 UTC+7** không tự đổi; chưa có hạn riêng cho toàn bộ 10 model mới. Cả hai cập nhật tiến độ/ETA/blocker tại mốc này, không giảm folds/seeds hoặc bịa số để nộp đủ.
- Ngay phiên tiếp theo: fetch/đọc quyết định, xác nhận ID trong `docs/discussions/dinhtung.md` hoặc `vinhduc.md`, push nhánh mình, nhận main an toàn, thiết lập và bắt đầu triển khai. Chưa có xác nhận thì không ghi là đã nhận/đã train. Dùng prompt AI, lệnh setup và checklist trong hợp đồng; runner mười model cần được triển khai, không có sẵn từ lần publish hướng dẫn.
- Quyền commit/push code/artifact đúng vùng, upload Release asset và mở PR giữ nguyên; không tự merge main, tiêu tiền, đổi hợp đồng hoặc sửa helper/kết quả cũ. Thiếu code/license/dependency/GPU thì báo blocker cụ thể và làm model độc lập tiếp theo.

## GB-20261123-001 - Nhận SkateFormer và tích hợp demo local

- Nguồn yêu cầu: Gia Bảo yêu cầu thêm model Tùng vào web, sau đó cập nhật repo ngày 23/11/2026 (UTC+7). Review sản phẩm đã thực hiện trước khi tích hợp; không phải cấp lại quyền train từng bước.
- **ACCEPTED_BENCHMARK:** PR #4, head `20c67f57c87d1a2cfc0489bf23da45522c4c8d2c`, đã merge lúc 12:55 ngày 23/11/2026 tại `3c6a39eaf4ee4d37e0848b94bad800aa2dcd4e3a`. Nhận mã, thiết kế, 48 fits, 12 checkpoint full và dự đoán SkateFormer adapted small nguyên trạng. Kết quả âm hợp lệ: Accuracy 84,17%, Sensitivity 84,81%, F1 84,05%; không thay A3/TCN/GRU hoặc model/threshold triển khai, không bắt train lại để có số đẹp.
- Review độc lập khớp 55 artifact, 11 source khóa, bốn input chuẩn và 7.200 dòng dự đoán; 20 test nghiên cứu PASS. Không chạy lại train/full GPU latency và không gọi local checks là CI. Giới hạn bốn subject/kiến trúc thu gọn giữ nguyên.
- **FallScope local:** năm model chạy được, gồm SkateFormer; hai baseline hiển thị nhưng khóa vì thiếu checkpoint. Phân tích trước, phát đồng bộ timestamp; benchmark tách khỏi Fall score video. Có báo cáo, sự kiện/tua, tiến độ/hủy và lịch sử opt-in; không hosting công khai, không giả Accuracy live. Demo qua 31 test và replay 60 checkpoint thật trên checkout main không có tuning local.
- Quyết định mới thay phần hoãn demo trong `GB-20261121-002` chỉ cho prototype do Gia Bảo yêu cầu. PoseLab vẫn gỡ, WEB-01/WEB-02 vẫn hủy; không thay quyền tự chủ/nhiệm vụ/hạn 25/11 của Đức, không nhận thay bài Đức. Không nhập ba commit runner/Kaggle tuning của `giabao` hoặc bộ lưu PARTIAL local trong đợt publish này. Word, notebook, dataset và deployment giữ nguyên.

## GB-20261121-002 - Gỡ PoseLab, hoãn demo và lập kế hoạch tuning A3

- Yêu cầu Gia Bảo ngày 21/11/2026: xóa PoseLab, cập nhật repo; app demo triển khai sau; lên kế hoạch tuning A3 để cạnh tranh với hai model của Tùng.
- **PoseLab `REMOVED`:** gỡ 31 file đã theo dõi trong `app/`, gồm backend/frontend/launcher/requirements và bản sao biểu đồ trong app. Giữ model trong `models/deployment/`, dữ liệu, kết quả/biểu đồ nghiên cứu gốc, notebook, Word và lịch sử Git. Thay chỉ dẫn cũ giữ nguồn app; không phải lệnh xóa bằng chứng hoặc lịch sử.
- **Demo mới `DEFERRED`:** không làm UI/QA, tải model app hoặc dựng launcher khác trong phiên này. WEB-01/WEB-02 vẫn `CANCELLED`, không giao lại cho Tùng/Đức.
- **A3 `PLANNED / NOT_RUN`:** [kế hoạch](../research/a3-tuning-plan.md) do Gia Bảo giữ; hai đối chứng chính hiểu là TCN và GRU, CTR-GCN adapted tham chiếu phụ. Sáu recipe/model, chọn bằng inner validation; ma trận công bằng dự kiến 684 fits, chưa viết runner hoặc train. Không hứa thắng, không chọn bằng outer-test đã xem, không thay deployment.
- Không sửa nguồn/lock/kết quả đã nghiệm thu, không yêu cầu Tùng train lại; không đổi đề xuất/mốc/hạn **25/11/2026** hoặc quyền tự chủ của Đức. Kế hoạch mới không là điều kiện chặn nghiệm thu hai bạn. Xem báo cáo Gia Bảo cho kiểm tra và tra SHA remote sau publish; không suy ra hai bạn đã đọc ở máy riêng.

## GB-20261121-001 - Nghiệm thu bổ sung GRU/CTR-GCN và sửa monitor

- **Nguồn yêu cầu:** Gia Bảo yêu cầu sửa lỗi đã review, merge PR #3 và cập nhật README đúng tiến độ ngày 21/11/2026 (UTC+7). Nhận head `5965c3e005e28928fd6137234095d64ecf41339d`, giữ lịch sử và không sửa/push nhánh cá nhân Tùng/Đức.
- **Quyết định:** `ACCEPTED_BENCHMARK` cho phần GRU/CTR-GCN adapted bổ sung; không thay nghiệm thu TCN–A3 theo `GB-20261120-001`. Có 96 fits mới, 24 checkpoint full và 2.880 dòng dự đoán; smoke 8 fits/hai checkpoint riêng. [Bằng chứng](../../results/backbone_benchmark/tung-gru-ctr-rtx3050-20261120-233932/README.md) và [thiết kế/giới hạn](../research/gru-ctrgcn-addendum.md).
- **Đối chiếu:** 80 hash artifact, 11 file code lock và bốn input khớp; lịch sử/split/chọn epoch-ngưỡng và tái tính metric/năm bảng so sánh PASS. Replay CPU 24 checkpoint full cho cùng 2.880 quyết định, score lệch tối đa khoảng `0.000397` so với CUDA; smoke cũng khớp. Không coi replay là train lại hoặc đo lại latency GPU.
- **Bản sửa điều phối:** `backup.mkdir(parents=True)` trong monitor để hỗ trợ checkout chưa có `.runtime`; thêm ba test riêng cho runtime mới/có sẵn/guard nhiệt. Không đổi nguồn train đã khóa, config, checkpoint hoặc kết quả. Không yêu cầu Tùng train lại. Sau tích hợp, 16 tests local và repository checker PASS; không có CI được cấu hình cho PR.
- **Kết luận được phép:** fixed Accuracy trung bình TCN 91,04%, GRU 89,58%, A3 chạy lại 89,38%, CTR-GCN adapted 88,54%. GRU nhanh nhất theo phép đo model-only, CTR-GCN adapted nhỏ nhất nhưng FN nhiều nhất; A3 mạnh hơn trên S4. Đây là kết quả khám phá của 160 clip/bốn người đã xem, không chứng minh ưu thế phổ quát hoặc tính năng chưa đo. Latency đo lại cả bốn, không trộn với phép đo cũ.
- **Phạm vi giữ nguyên:** chưa thay model/threshold triển khai, bật PoseLab hoặc cập nhật Word; bài Đức tiếp tục độc lập theo hạn 25/11. Không đặt thêm thí nghiệm thành điều kiện chặn nghiệm thu Tùng. Sau publish, hai bạn fetch/merge main an toàn; không suy ra đã đọc ở máy riêng.

## GB-20261120-001 - Nghiệm thu và tích hợp benchmark TCN–A3 của Tùng

- Gia Bảo yêu cầu ngày 20/11/2026: bổ sung bài Tùng vào main và cập nhật README đầy đủ, sau kết luận đủ điều kiện nhận PR. Nghiệm thu phạm vi **M0–M3 benchmark TCN–A3**, không phải tuyên bố TCN tốt hơn trên mọi dữ liệu hoặc cấp phép đổi deployment.
- Nguồn đã review: `dinhtung` tại `b978ff0fdd2e4a9f38d9c2cd5bcc72950d98bcf4`, bộ kết quả được nộp ở `582b653c56d2d871fa86e8ae827205483aa3a746`, code train `12eaf80e3f2dcbdc9df33749fa61aad5b6f6ce0c`. [PR #2](https://github.com/aBaoN9/me-stgcn-ta-fall-detection/pull/2) đã **MERGED** bằng merge commit `bbe3c64638ce2cfae31348cb64a455a97c3548bf` ngày 20/11/2026, giữ lịch sử commit và nhánh thành viên.
- Phần nhận vào main: `training/backbone_benchmark/`, bộ run `results/backbone_benchmark/tung-m2-rtx3050-20261120-165523/`, thiết kế/phân tích và báo cáo/trao đổi Tùng. Đủ **96 fits, 24 outer runs, 24 checkpoint, 2.880 dòng dự đoán** (gồm ba chính sách), đúng bốn input chuẩn. Không sửa helper chung, notebook/model/kết quả cũ hoặc ứng dụng.
- Kiểm độc lập trước tích hợp: 68 SHA-256 đúng; code/input lock khớp; đủ lịch sử inner/refit và chia subject không giao nhau; raw OOF/metric/nhãn/video/subject khớp. Nạp 24/24 checkpoint và suy luận CPU cho cùng 2.880 quyết định; score sai lệch tối đa khoảng `0.000508` do khác môi trường. 10/10 kiểm thử benchmark PASS. Review dùng Torch 2.11.0+cpu, không gọi là đã tái lập CUDA 2.10.0+cu126 hoặc train lại. Repo checker PASS; chính sách CRLF của artifact được giữ để không đổi hash, không normalize CSV/JSON cho sạch diff.
- Kết quả chính `fixed`/0,5: Accuracy TCN **91,04% ± 0,72**, A3 chạy lại **89,38% ± 2,50**, tăng **1,67 điểm phần trăm**; TCN ít tham số hơn, nhưng thua A3 về Accuracy ở seed 242. Chỉ bốn người, outer fold đã được xem trong phát triển, không phải xác nhận độc lập. [README](../../README.md) tách bảng mới khỏi 07/08, ghi chi phí/latency, lỗi lặp, bằng chứng và cách kiểm/chạy lại.
- Chấp nhận bàn giao Tùng **`ACCEPTED_BENCHMARK`**. Các snapshot cá nhân `READY_FOR_REVIEW` hoặc `NOT_RUN` cũ phản ánh thời điểm ghi, không đảo quyết định nghiệm thu này. Không yêu cầu tìm chỉ số/seed để A3 thắng hoặc chạy thêm mới chấp nhận sản phẩm. So sánh cùng mức FP, kiểm robustness pose, lỗi cặp hoặc A3 gọn hơn chỉ là đề xuất nghiên cứu tiếp, phải công bố là phân tích bổ sung và thiết kế công bằng; chưa phải kết quả đã đo.
- **Không thay model/threshold triển khai, không bật PoseLab, không sửa Word trong phiên này.** Đức vẫn có quyền tự chủ và tiếp tục phần riêng; không lấy nghiệm thu Tùng làm nghiệm thu Đức hoặc tự thay mốc Đức bằng TCN. Kết quả mới đã public trên main không đồng nghĩa hai bạn đã đọc ở máy riêng.

## GB-20261119-002 - Gỡ vướng vận hành, được tự full CPU và nộp artifact

- Người quyết định: Gia Bảo, chủ repo/người điều phối; yêu cầu ngày 19/11/2026 rà lại và đưa hết phần cần thiết để Tùng/Đức làm bài, không bắt duyệt từng bước. Đây là bổ sung vận hành cho quyền tự chủ hiện hành, không nghiệm thu bài làm hay cho phép đổi giao thức nghiên cứu.
- Trả lời **`DUC-20261117-001`** trên `vinhduc` commit `eac041170f4d85a059386c87df2f1ad332a91a38`: (1) đủ cả bốn input đúng hash đã trên main tại `data/gmdcsa24/`, không phải bộ dựng lại; (2) **được full trên CPU nếu không có GPU**, áp dụng cả Đức và Tùng, hai vế so cùng thiết bị/recipe/ngân sách và báo cáo phần cứng/thời gian; (3) được tự commit checkpoint nhỏ đích danh cùng manifest/hash trên nhánh mình hoặc chọn GitHub Release cho artifact không phù hợp Git, không phải xin lại cách lưu. Không cấp tiền dịch vụ hay tài khoản/quota.
- CPU là thay đổi vận hành được cho phép, không đổi nhãn/split/folds/seeds/metric/selection. Tự ghi M0 addendum và khóa code/config/thiết bị trước outer-test; runner/notebook riêng phải thực thi lựa chọn đúng lock và giữ nguyên guard. Đức đang `allow_cpu_full=false` và notebook assert CUDA, CLI chưa có cờ override: được tự bổ sung trong vùng riêng và cập nhật lock, không sửa helper chung hoặc giả sử quyền mới tự sửa code. Không chờ duyệt addendum này; chữ ký đổi thì thư mục run mới, không ghi đè lock hoặc ép resume.
- Cấp [tool kiểm/tải input](../../scripts/training_inputs.py) chỉ dùng Python chuẩn, dữ liệu pin commit `afc7370c324aabec1a6d1076b19d8247c818d96c`, kiểm đủ size/hash và không ghi đè file hiện có. [Hướng dẫn bắt đầu](../training-start.md) có local/Kaggle/offline, tách code/data, CPU, kiểm tra và bàn giao. Giữ notebook code commit đã khóa; chỉ merge main ngoài notebook không làm cell pin commit cũ và quét `/kaggle/input` tự tìm được dữ liệu mới. Thành viên tự ghi thay đổi cell vận hành.
- Tùng được bắt đầu phần TCN/M0 đã nộp hoặc tự điều chỉnh có ghi lý do trước outer-test theo `GB-20261118-001`; Đức tiếp tục thay đổi có kiểm soát ME-STGCN-TA theo `GB-20261116-002`. Không merge code hai bạn chưa review lên main hoặc làm thay nghiên cứu để gọi là sẵn sàng.
- Tự kiểm dữ liệu/test/smoke thật PASS rồi full; đây không phải quyền bỏ kiểm tra. Hạn model đã train + kết quả thật vẫn **25/11/2026 (UTC+7)**. Tài khoản, quota, dependency và khả năng chạy trên máy thành viên chưa được xác minh ở phiên này; không nhận là hai bạn đã đọc, smoke/full hoặc hoàn tất vì đã publish hỗ trợ.

## GB-20261119-001 - Cấp bốn input chuẩn trên main cho Tùng và Đức

- Người quyết định: Gia Bảo, chủ repo/người điều phối; yêu cầu trực tiếp trong chat ngày 19/11/2026 (UTC+7), sau tin Tùng cần bốn file để làm bài: kiểm main và đưa file lên. Đây là yêu cầu publish input cụ thể, không phải quyền public toàn bộ dữ liệu riêng.
- Đã kiểm: trước cập nhật, main chỉ có `data/README.md`, không có bốn input và không có Release chứa chúng. Đã tìm đủ bản export notebook 01 hiện có; hai tensor mang đuôi `.zip`, hai CSV mang tên `pose_mnf.txt`/`cross.txt`. SHA-256 cả bốn khớp tuyệt đối protocol 07.
- Phạm vi được publish: `data/gmdcsa24/gmdcsa24_pose_32.npz`, `gmdcsa24_features_32.npz`, `gmdcsa24_pose_manifest.csv`, `gmdcsa24_cross_subject_folds.csv`, kèm `input_manifest.json` ghi nguồn/hash/dung lượng. Chỉ chép byte-for-byte sang tên chuẩn, không regenerate/recompress/sửa CSV/nhãn/split. Bốn input tổng 15.100.639 byte; giữ nguyên hash chuẩn trong `results/component_ablation/locked_protocol.json`.
- Nguồn GMDCSA-24: Ekram Alam, bản `v2.1`, DOI `10.5281/zenodo.13354453`; metadata nguồn ghi CC BY 4.0. Đây là tensor/CSV dẫn xuất từ notebook 01 của nhóm, không đóng gói video gốc. Xem [nguồn, hash và hướng dẫn](../../data/README.md).
- Tùng/Đức fetch và nhận main an toàn trên nhánh mình rồi dùng **`--input-root data/gmdcsa24`**; trên Kaggle gắn đủ bốn file và truyền đường dẫn tương ứng. Không cần chờ gửi riêng/cấp lại cùng bộ input hoặc chạy lại 01. Loader đọc NPZ với `allow_pickle=False`; CSV giữ path Kaggle gốc chỉ để truy vết, không đòi video trên máy mới.
- Kiểm tra: bốn hash, cấu trúc/ID/nhãn/subject/mask/duration, 160 mẫu/79 Fall/81 ADL và bốn fold không rò rỉ PASS; `python -m training.component_ablation.validate_pipeline --input-root data/gmdcsa24` PASS, gồm parity feature, 100 ca augmentation và shape các biến thể gốc. Đây là validation trên máy điều phối, không phải smoke/full train TCN và không xác nhận GPU/môi trường máy thành viên.
- Chỉ mở ignore đúng bốn input và metadata này. Video, `data/local/`, tensor riêng khác, token/môi trường vẫn không được publish. Không thay dữ liệu/giao thức/deployment hoặc nguồn/kết quả cũ.
- Quyền tự triển khai/smoke/train/nộp sản phẩm đã cấp vẫn giữ, không có bước xin duyệt mới. Hai bạn xác nhận đường dẫn/hash đã nhận trong báo cáo/câu hỏi cá nhân; nếu còn blocker thì nêu lỗi/quyền/GPU cụ thể. Publish main không chứng minh thành viên đã tải hoặc đã train. Hạn 25/11/2026 giữ nguyên.

## GB-20261118-001 - Tùng tự chủ triển khai và train, không chờ duyệt

- Người quyết định: Gia Bảo, chủ repo/người điều phối; yêu cầu trực tiếp trong chat ngày 18/11/2026 (UTC+7): chốt để Tùng bắt đầu làm/train và tự do thực hiện, không bắt Gia Bảo duyệt từng bước.
- Trả lời: `TUNG-20261117-001`, M0 trên `origin/dinhtung` tại `d65e59c` ngày 17/11/2026. Quyết định: **`APPROVED_TO_PROCEED / AUTONOMOUS`**; được bắt đầu M1 ngay, tự chạy full sau kiểm tra đạt. Không phải xác nhận đã có code/model/kết quả hoặc nghiệm thu M0 thành sản phẩm cuối.
- Thay toàn bộ cổng chờ duyệt M0, chọn/đổi mô hình, cấu hình, thiết bị, smoke/full hoặc review lại trước chạy dành cho Tùng trong tài liệu/quyết định cũ và M0 đã gửi. `GB-20261116-002` của Đức giữ nguyên; từ nay cả hai tự thực hiện trong nhiệm vụ, review sau để nghiệm thu/merge.

### Quyền tự quyết và hành động tiếp theo

1. Bắt đầu từ **TCN_v2_small so với A3 chuẩn/07 chạy lại**, cùng tám kênh feature v2, ba seed và thiết kế 96 fits đã đề xuất. Đây là phương án xuất phát được phép, không ép phải giữ TCN/hidden32/96 fits nếu cần thiết kế hợp lý hơn trước outer-test. Tùng tự chọn mô hình/phương pháp khác, kiến trúc, cấu hình, ngân sách công bằng và dự phòng GRU trong nhiệm vụ; ghi lý do, phiên bản M0 và khóa trước outer-test, không xin duyệt lại. Giữ dataset/nhãn/split/seed và giao thức đánh giá chung; không giảm coverage rồi gọi full.
2. Tự triển khai code/runner/notebook vùng riêng, test shape/mask/split/metric/parity đối chứng, smoke dữ liệu thật, sửa lỗi và full khi đủ điều kiện. Không cần chờ Gia Bảo xem code, chọn siêu tham số, xác nhận từng lệnh hay trả lời Markdown để bắt đầu. M0 là đăng ký phương án (`M0_DOCUMENTED`), không là đơn xin phép.
3. Tự chọn Kaggle hoặc GPU local có quyền sử dụng, trong hạn mức miễn phí/đã có, kiểm runtime/VRAM và ghi thiết bị; cặp so chạy/đo cùng điều kiện hoặc công bố rõ khác biệt. Không xin lại chỉ để đổi thiết bị. Không tự mua GPU/dịch vụ hoặc vượt giới hạn tài khoản/công cụ.
4. Tùng và AI của Tùng được commit/push code, test, M0, báo cáo, model và kết quả đúng vùng nhiệm vụ lên `dinhtung`, upload GitHub Release khi cần và mở PR, không xin lại từng lần. Quyền nộp artifact đã có theo `GB-20261116-001`; tự kiểm auth/quyền thực tế và báo lỗi nếu bị chặn. Không force-push, sửa/push nhánh khác/main hoặc tự merge.
5. Fetch/đọc quyết định này, cập nhật cùng câu `TUNG-20261117-001` và báo cáo trên nhánh mình: đã nhận quyền, việc đang làm và blocker thật. Bỏ `PENDING_APPROVAL`/`WAITING_FOR_MAIN` cho phần đã được cấp quyền. Không cần chờ xác nhận lại việc đã đọc để triển khai; không xóa câu hỏi khác chưa giải quyết.

### Điều kiện vẫn giữ, không phải cổng duyệt thủ công

- Bốn input có quyền, đúng hash/căn chỉnh và nhãn/subject; test và smoke thật PASS trước full. Chưa có input/runtime/GPU thì ghi `BLOCKED`/`NOT_RUN`, báo nguồn/quyền cần hỗ trợ cụ thể và tiếp tục code/test độc lập. Quyết định này cấp quyền, **không xác nhận dữ liệu đã được chuyển hoặc GPU đã sẵn sàng**.
- Không tuning bằng outer-test hoặc lỗi test đã biết; không chọn seed đẹp. Đổi phương án sau khi đã xem outer phải giữ run cũ, công bố test đã được xem và phiên bản mới, không biến nó thành blind test hoặc hồi tố phương án khóa.
- Lưu checkpoint tạo dự đoán, raw OOF, metric tái tính, config/code/input hash, log và lệnh tái lập; nộp model + kết quả qua GitHub theo checklist hiện hành. Hạn **25/11/2026 (UTC+7)** giữ nguyên; kết quả âm hợp lệ, chưa chạy thì không bịa hoặc lấy 06/07/08 làm kết quả mới.
- Tự do nghiên cứu trong nhiệm vụ không cho phép đổi dataset/nhãn/giao thức chung, thêm dữ liệu/weights ngoài chưa được phép, public dữ liệu riêng, sửa helper/checker/kết quả gốc, phần Đức, deployment hoặc bật PoseLab. Chỉ hỏi khi thực sự cần quyền/tài nguyên hoặc làm ngoài phạm vi; không gửi yêu cầu duyệt kỹ thuật đã được tự quyết.
- Gia Bảo review sản phẩm để nghiệm thu/tổng hợp/quyết định merge sau bàn giao; không biến review thành điều kiện chờ trước train. Đăng quyết định trên main chưa chứng minh Tùng đã đọc; chỉ ghi nhận khi có phản hồi trên nhánh cá nhân.

## GB-20261116-002 - Đức tự chủ triển khai và chạy thí nghiệm

**Hiệu lực cập nhật 18/11/2026:** quyền của Đức giữ nguyên; các câu dưới đây nói quy trình Tùng không đổi chỉ là lịch sử, đã được thay bởi `GB-20261118-001`.

- Người quyết định: Gia Bảo, chủ repo/người điều phối; yêu cầu trực tiếp trong chat ngày 16/11/2026 (UTC+7): sửa quy định để không chờ duyệt làm kéo dài quá trình.
- Quyết định: `APPROVED` cho quyền tự thực hiện của Đức, không phải xác nhận code/kết quả đã nghiệm thu. Thay các điều kiện chờ duyệt M0, thành phần kiến trúc và review lại trước smoke/full dành cho Đức trong `GB-20261030-001`, `GB-20261116-001` và review ngày 29/10/2026. Quy trình của Tùng không đổi.

### Phạm vi được tự quyết

1. Giữ đúng mốc `A3_motion_attention` / `fixed` của notebook 07, mã `training/component_ablation/` tại `c8f654bf0f1943ffcfe375911c8b3e0015beadc5`: tám kênh, attention bật, dropout 0.3, Fall multiplier 1.0, feature `v2-mask-visibility-raw-pose-augmentation`, ngưỡng 0.5. Chạy lại bản gốc cùng bản thay đổi; không đổi mốc thành cost225/06/T0/T1/Q1 hoặc lấy số cũ làm đối chứng mới.
2. Tự chọn, giải thích và triển khai thay đổi có kiểm soát về xử lý, huấn luyện, chính sách quyết định hoặc thành phần kiến trúc như pooling/attention. Không thay hẳn mô hình/backbone sang hướng của Tùng; không bắt buộc quality-aware hoặc công thức cụ thể.
3. Ghi M0 với giả thuyết, phần giữ nguyên/thay đổi, cặp so sánh, tiêu chí, cấu hình/grid và ngân sách; khóa trước outer-test, ghi `M0_DOCUMENTED`. Đây là đăng ký phương án, không phải đơn xin duyệt; không chờ trả lời hoặc `APPROVED` từng bước.
4. Tự triển khai, sửa lỗi, chạy test, validation, smoke thật rồi full khi đủ điều kiện bên dưới. Không cần Gia Bảo review lại bản sửa trước chạy; test thất bại phải sửa/chạy lại, không bỏ guard để vượt điều kiện.
5. Được commit/push mã, test, phương án, báo cáo, model và kết quả thuộc vùng nhiệm vụ lên `vinhduc`, upload artifact theo checklist `GB-20261116-001` và mở PR, không xin lại từng lần. Kiểm outgoing commits, chỉ stage file đúng phạm vi; không force-push, sửa nhánh khác hoặc tự merge main.

### Điều kiện kỹ thuật và nghiệm thu

- Bốn input có quyền sử dụng, đúng hash/căn chỉnh; giữ dataset/nhãn/subject split, seed 42/142/242 và nguyên tắc nested LOSO chung. Không tuning bằng outer-test, chọn seed đẹp hoặc đổi dữ liệu để có số tốt hơn.
- Test hồi quy/parity, dữ liệu/split/metric và smoke trên dữ liệu thật phải PASS, có log/bằng chứng. Các lỗi DUC-REV còn liên quan vẫn phải xử lý/test; fixture không thay chạy thật. Có môi trường/tài nguyên được phép mới chạy; thiếu input/GPU thì báo sớm, ghi `NOT_RUN` và làm tiếp việc độc lập.
- Báo cáo partial/full trung thực, lưu config/hash/code/run/seed/fold, raw OOF và model dùng tạo kết quả. Hạn **thứ Tư 25/11/2026 (UTC+7)** vẫn yêu cầu **MODEL ĐÃ TRAIN + KẾT QUẢ ĐÁNH GIÁ trên GitHub**, không chỉ M0/tiến độ; không yêu cầu kết quả phải tốt hơn.
- Người điều phối review/kiểm chứng để nghiệm thu, tổng hợp và quyết định merge main sau bàn giao. Quyền tự chạy không nghĩa là bản sửa đã kiểm độc lập hoặc được gắn `DONE`/`complete=true` khi thiếu bằng chứng.
- Vẫn hỏi khi muốn đổi dataset/mốc/giao thức chung hoặc làm ngoài phạm vi; không tự mua GPU/dịch vụ, public dữ liệu riêng tư/token, sửa helper/checker chung, model triển khai, bật PoseLab hoặc sửa phần Tùng.

### Trả lời các câu hỏi Đức đang chờ

- `DUC-20261103-001`: phần xin thay đổi thành phần kiến trúc đã được cho phép, gồm thử quality-aware pooling nếu Đức chọn. Không cần chờ duyệt phạm vi/M0 riêng; tự khóa phương án và kiểm tra theo quy định mới.
- `DUC-20261028-001`: bỏ điều kiện phải được người điều phối review lại/duyệt riêng trước full. Danh sách lỗi DUC-REV và nghĩa vụ sửa/test vẫn giữ khi dùng mã liên quan; chưa độc lập xác nhận bản sửa ngày 03/11 đã PASS. Yêu cầu input/quyền/GPU chưa được quyết định này giải quyết.
- Đức cập nhật cùng file trao đổi/báo cáo: xác nhận đã nhận quyền, tách blocker dữ liệu/GPU còn mở. Không đánh dấu toàn bộ câu hỏi `RESOLVED` hoặc tự sửa answers để nhận là đã nghiệm thu.

## GB-20261116-001 - Nhắc Tùng và Đức thực hiện nhiệm vụ

**Hiệu lực mới:** các điều kiện chờ phê duyệt riêng cho Đức/Tùng trong thông báo này đã được thay bởi `GB-20261116-002`/`GB-20261118-001`; hạn, sản phẩm và checklist nộp vẫn giữ. Cả hai tự ghi/khóa phương án và tự chạy khi kiểm tra đạt, không chờ duyệt từng bước.

- Người ra thông báo: Gia Bảo, chủ repo/người điều phối; yêu cầu trực tiếp trong chat ngày 16/11/2026, UTC+7.
- Cập nhật hạn theo yêu cầu tiếp theo của Gia Bảo ngày 16/11/2026: **chậm nhất thứ Tư, 25/11/2026 (UTC+7)**. "Thứ Tư tuần sau" tính từ thứ Hai 16/11 là 25/11, không phải 18/11.
- Gia Bảo bổ sung trong chat ngày 16/11/2026: hạn này áp dụng cho **sản phẩm MODEL ĐÃ TRAIN + KẾT QUẢ ĐÁNH GIÁ của cả hai, phải upload lên GitHub**, kèm mã/lệnh tái lập và báo cáo; không chỉ đề xuất hoặc báo cáo tiến độ. Tùng nộp mô hình/phương pháp khác; Đức nộp cặp mốc gốc và thay đổi có kiểm soát. Thiếu phê duyệt/input/GPU phải báo sớm để điều phối giải quyết, nộp phần đã làm đúng hạn và ghi phần chưa chạy `NOT_RUN`; chưa đủ sản phẩm là chưa hoàn tất. Tùng giữ điều kiện duyệt M0/full; Đức tự ghi/khóa M0, tự kiểm test/smoke rồi full theo `GB-20261116-002`, review nghiệm thu sau bàn giao.
- Yêu cầu: cả hai chủ động bắt đầu hoặc tiếp tục phần nghiên cứu được giao và nộp tiến độ, không chỉ đọc phân công rồi chờ. Không áp sẵn cách giải hay yêu cầu kết quả phải tốt hơn.
- Trạng thái: `ACTION_REQUIRED` cho xác nhận/cập nhật; quyền tự chạy của Đức theo `GB-20261116-002`, không mở rộng quyền của Tùng.

### Hiện trạng đã kiểm tra

- Fetch ngày 16/11: `origin/dinhtung` ở `1ab8164` (28/10), chưa thấy M0/code/kết quả mới được nộp cho hướng hiện hành. Chỉ kết luận về phần đã gửi lên repo, không suy đoán việc làm riêng.
- `origin/vinhduc` ở `da5b552` (03/11): ghi nhận đề xuất quality-aware pooling và bản sửa DUC-REV-01..05; không yêu cầu Đức làm lại từ đầu. Các kiểm tra PASS là do Đức báo cáo, chưa được người điều phối độc lập kiểm lại; smoke thật/full vẫn `NOT_RUN`, chưa có kết quả train mới.
- `DUC-20261103-001` và `DUC-20261028-001`: quyền tự thực hiện đã trả lời ở `GB-20261116-002`; không còn chờ duyệt phạm vi/M0/review lại để chạy. Nguồn bốn input đúng hash và GPU vẫn là phụ thuộc chưa giải quyết, không quy thành lỗi chậm thực hiện của Đức.

### Tùng - mô hình hoặc phương pháp khác

1. Fetch và đọc hướng dẫn mới từ `origin/main`; thực hiện trên `dinhtung`, không thực nghiệm trên main.
2. Ngay phiên tiếp theo, xác nhận đã nhận và bắt đầu/nộp M0: tự chọn mô hình hoặc phương pháp khác trên cùng GMDCSA24, giải thích lý do, phần giữ nguyên/thay đổi, cách so với ME-STGCN-TA, ngân sách và hỗ trợ cần thiết. Không bắt buộc CTR-GCN hay chỉ đổi backbone.
3. Cập nhật `docs/reports/dinhtung.md`: việc đã làm, bằng chứng, bước tiếp theo, blocker. Chờ duyệt thì tiếp tục đọc baseline, thiết kế, chuẩn bị mã/kiểm tra trong phạm vi; không tự chạy full.

### Đức - thay đổi có kiểm soát mô hình hiện tại

1. Tiếp tục trên `vinhduc`, lấy đúng `A3_motion_attention` / policy `fixed` của notebook 07 và `training/component_ablation/` làm mốc: tám kênh, attention bật, Fall multiplier 1.0, ngưỡng 0.5. Không dùng checkpoint final 06, A3_cost_225, T0/T1 hoặc Q1 thay mốc.
2. Cập nhật `docs/reports/vinhduc.md` cho khớp commit đã push, dẫn bản sửa, lệnh/bằng chứng kiểm tra và trạng thái thật; giữ các câu hỏi đang mở theo đúng ID. Nêu rõ phần làm được ngay và phần cần Gia Bảo cung cấp/quyết định.
3. Được tự chọn thay đổi thành phần kiến trúc, kể cả quality-aware pooling, theo `GB-20261116-002`; không chờ duyệt. Tự review/test, chuẩn bị môi trường/notebook và kiểm input; không bỏ hash guard, dùng dữ liệu giả làm smoke thật hoặc tự mua GPU.
4. Khi đã ghi/khóa M0, đủ input/môi trường và test/smoke thật PASS, tự chạy full, không chờ review lại hoặc duyệt riêng. Chạy lại bản gốc cùng bản thay đổi trong điều kiện so sánh như nhau; không dùng số lịch sử làm đối chứng mới.

### Sản phẩm bắt buộc và cách upload GitHub

1. **Model thật:** trọng số/checkpoint hoặc artifact mô hình đã fit phù hợp phương pháp, có mã dựng/load và lệnh inference/evaluate chạy lại được. Ghi rõ run, seed, fold, cấu hình, phiên bản feature và dữ liệu đã dùng để train; nộp checkpoint dùng tạo kết quả hoặc liên kết artifact tương ứng, không lấy model cũ đổi tên thành model mới. Không bắt buộc ONNX, giao diện hoặc một lần train all-subject mới ngoài phương án đã duyệt.
2. **Kết quả thật:** bảng Accuracy, Precision, Recall, Specificity, F1, FN/FP và confusion matrix; kết quả từng fold/seed và tổng hợp mean/std theo protocol, kèm raw OOF/dự đoán, config, hash input/code/model, log/history và trạng thái hoàn tất. Tùng đối chiếu ME-STGCN-TA theo cùng đánh giá; Đức có kết quả cho cả mốc chạy lại và bản thay đổi. Không chọn checkpoint/seed đẹp bằng outer-test, không lấy smoke hoặc số lịch sử làm kết quả mới.
3. **Vị trí:** Tùng để model theo run trong `results/backbone_benchmark/<run>/models/` và kết quả trong cùng run; Đức dùng `results/contribution_study/<run>/models/` và kết quả tương ứng. Mã/lệnh chạy ở vùng training cá nhân; báo cáo cá nhân dẫn commit, artifact và cách tái lập. Không ghi đè `models/deployment/` hoặc bằng chứng kết quả gốc.
4. **Đưa lên GitHub:** Gia Bảo yêu cầu và cho phép commit/push sản phẩm trong phạm vi nhiệm vụ lên đúng nhánh `dinhtung`/`vinhduc`, mở PR để review; không tự merge main hoặc gửi kèm thay đổi ngoài nhiệm vụ. Artifact nhỏ phù hợp Git thì commit trực tiếp. `.pt`/`.onnx`/`.npz` đang bị ignore: chỉ stage đích danh artifact đã kiểm trong thư mục model riêng bằng `git add -f -- <file>` nếu cần; không bỏ ignore toàn cục, không force-add cả thư mục dự án.
5. **Artifact lớn:** upload dưới dạng GitHub Release asset trên repo này và ghi đường dẫn tải, SHA-256, dung lượng, run/seed/fold trong manifest được commit trên nhánh cá nhân. Không nhét checkpoint lớn vào lịch sử Git, không tự bật dịch vụ trả phí hoặc chỉ để model ở Kaggle/máy riêng mà không có bản GitHub truy cập được. Không upload video/tensor riêng, token hoặc môi trường. Thiếu quyền upload thì báo ngay, không nói đã gửi.
6. **Nghiệm thu:** kiểm lại remote/PR và khả năng tải, đối chiếu hash, thử load model và tái tính metric từ raw predictions. Ghi PASS/FAIL/NOT_RUN. Chỉ báo đủ sản phẩm khi người review lấy được model và kết quả tương ứng; chỉ có notebook, ảnh chụp bảng điểm hoặc báo cáo chữ là chưa đủ. Model học toàn dữ liệu nếu có phải ghi rõ không có test độc lập, không gán metric outer-fold cho chính checkpoint đó.

### Xác nhận và bàn giao

- Cả hai xác nhận ngay phiên làm việc tiếp theo trong `docs/discussions/dinhtung.md` hoặc `docs/discussions/vinhduc.md`, dẫn `GB-20261116-001` và hạn **25/11/2026 (UTC+7)**; bàn giao model và kết quả chậm nhất ngày này theo checklist trên, kèm báo cáo, bằng chứng, bước tiếp theo và hỗ trợ cần thiết. Không chờ hết hạn mới xác nhận; gặp blocker phải báo sớm, không im lặng chờ hoặc bịa kết quả.
- Commit/push xác nhận, câu hỏi và sản phẩm theo quyền/phạm vi nêu trên; không xóa câu hỏi chưa giải quyết, không push hộ nhau hoặc tự merge main.
- Giữ cùng dataset/nhãn/split theo subject, quy tắc đánh giá/chọn cấu hình/ngưỡng, seed 42/142/242 và ngân sách so sánh đã thống nhất. Không chọn theo outer-test hoặc seed đẹp; kết quả âm vẫn hợp lệ.
- Thiếu input/GPU/phê duyệt: ghi `NOT_RUN` và blocker cụ thể cho phần chưa chạy, tiếp tục việc độc lập. Đức không cần chờ Tùng để chuẩn bị cặp đối chứng.
- PoseLab vẫn dừng, WEB-01/WEB-02 vẫn `CANCELLED`; không làm web/UI/QA thay nghiên cứu.
- Đăng trên main không đồng nghĩa hai bạn đã đọc; chỉ ghi nhận đã nhận khi có xác nhận trên nhánh cá nhân.

Đọc thông báo mà không cần merge code:

```bash
git fetch origin
git show origin/main:docs/discussions/answers.md
git show origin/main:docs/tasks/dinhtung.md
git show origin/main:docs/tasks/vinhduc.md
```

**Lưu ý lịch sử:** các cổng chờ phê duyệt/M0/review lại trước chạy dành cho Tùng/Đức trong các quyết định cũ dưới đây không còn hiệu lực theo `GB-20261118-001`/`GB-20261116-002`. Phát hiện lỗi, yêu cầu bằng chứng và giới hạn dữ liệu/an toàn vẫn giữ.

## GB-20261030-001 - Phân công hai hướng và dừng PoseLab

- Người quyết định: Gia Bảo, chủ repo/người điều phối; xác nhận trực tiếp trong chat ngày 30/10/2026, múi giờ UTC+7.
- Quyết định: `APPROVED` cho phạm vi phân công mới và dừng app; **không phải phê duyệt cấu hình hoặc full train**.
- **Tùng / dinhtung:** giữ dataset, dùng mô hình hay phương pháp khác để xử lý và so sánh với ME-STGCN-TA. Tùng tự lựa chọn và giải thích thay đổi, không bắt buộc CTR-GCN hoặc ma trận B0-B3.
- **Đức / vinhduc:** thay đổi có kiểm soát mô hình hiện tại, giữ ME-STGCN-TA làm mốc; so bản gốc với thay đổi được giải thích. Không giao sẵn quality-aware/công thức Q0-Q1 và không đổi hẳn mô hình như nhiệm vụ Tùng. Giữ kiến trúc trong hướng chính, sửa kiến trúc cần xin mở rộng phạm vi.
- **Chốt nhận diện mốc cho Đức:** `A3_motion_attention` / `fixed` của notebook 07 và mã `training/component_ablation/` tại `c8f654bf0f1943ffcfe375911c8b3e0015beadc5`; `channels=8`, `attention=True`, `fall_multiplier=1.0`, feature v2. Nguồn/cấu hình/kết quả lịch sử và cách chạy lại bản gốc cùng bản thay đổi ghi trong [nhiệm vụ Đức](../tasks/vinhduc.md). Không dùng checkpoint final 06, A3_cost_225, T0/T1 hoặc Q1 thay mốc. Đây là chốt đối chứng, không áp sẵn cách cải tiến hoặc phê duyệt full train.
- Cùng dữ liệu/nhãn/split theo subject, quy tắc chọn cấu hình/ngưỡng, seed, ngân sách tương đương và cách tính kết quả theo [protocol](../research-protocol.md). Nộp M0 ngắn về phần giữ nguyên/thay đổi, lý do, cặp so và ngân sách trước full; không chọn bằng outer-test.
- Hủy `WEB-01` và `WEB-02`, không còn việc PoseLab khi rảnh hoặc chờ GPU. Chặn launcher và khởi động ASGI; giữ mã/model/bằng chứng để truy vết, không xóa lịch sử cục bộ hoặc kết quả.
- Thay phạm vi giao ngày 28/10/2026 và phần ép hướng quality-aware/công thức trong `DUC-20261028-001` ngày 29/10. Giữ phát hiện lỗi phần mềm/kiểm dữ liệu và nghĩa vụ sửa nếu mã còn dùng; không tự nâng trạng thái bài làm đã review.
- Không yêu cầu bỏ bài làm cũ: thành viên chỉ rõ phần tái sử dụng, phần cần sửa và phương án tiếp theo. Không bắt hai bạn tạo số tốt hơn; báo cả kết quả âm và đánh đổi.
- Gia Bảo đã yêu cầu bổ sung mốc rõ ràng cho Đức rồi publish lên main trong chat ngày 30/10/2026. Tích hợp sau kiểm tra và xác nhận remote/PR trước khi nói đã gửi; hai bạn fetch/đọc main và nhận cập nhật an toàn, không reset bài làm hoặc ghi đè nhánh cá nhân. Quyết định này không cho phép AI đẩy thay nhánh Tùng/Đức.

## DUC-20261028-001 - Review cũ, phạm vi đã cập nhật

**Đọc GB-20261030-001 trước.** Nội dung sau giữ làm lịch sử review; phần chỉ định quality-aware/Q0-Q1 không còn bắt buộc. Các lỗi kỹ thuật và yêu cầu trung thực bằng chứng vẫn áp dụng nếu tiếp tục dùng mã đó.

- ID câu hỏi / người hỏi: `DUC-20261028-001` / Đức, nhánh `vinhduc`.
- Thời gian / vai trò: 2026-10-29 19:18 UTC+7 / AI điều phối theo yêu cầu chủ repo.
- Snapshot đã review: `origin/vinhduc` tại `1cbb778` (28/10/2026); không suy ra commit sau đã được duyệt.
- Quyết định: `CHANGES_REQUESTED`, không bác bỏ hướng nghiên cứu, không yêu cầu làm lại từ đầu. Chưa có quyết định cũ cho ID này.

### Đánh giá: giữ phần đúng, chưa nghiệm thu nghiên cứu

- Đúng phân công, không sửa model triển khai/nguồn cũ, không lấy web thay nghiên cứu. DUC-01 có script và số liệu tái lập được, không yêu cầu train lại 07/08 hay viết lại bảng đúng. Failure mode chưa xác minh bằng video vẫn phải ghi giới hạn.
- Người điều phối chạy lại audit PASS 22/22, checker repo PASS; bộ kiểm pooling PASS 13/13, smoke tổng hợp đủ 8 outer / 32 fits. Môi trường review: Python 3.11, torch 2.11.0+cpu, numpy 1.26.4, pandas 2.2.3, scikit-learn 1.8.0; khác môi trường Đức đã báo. Chưa chạy smoke thật/full.
- Giữ đề xuất `w = softmax(a + log(0.1 + q'))`, q từ visibility đã mask/augment, epsilon 0.1 cố định, avg-pool kernel 3 / stride 2 / padding 1 bỏ padding; Q0 multiplier 1.0, recipe 07. Không yêu cầu đổi kiến trúc hoặc search epsilon. Đây chưa là APPROVED cho full.
- Related work vẫn là bản thảo; review này không xác nhận đã thẩm định toàn văn từng nguồn hoặc tính mới. Smoke giả không chứng minh cải thiện khoa học.

### DUC-REV-01 - Khóa đúng dữ liệu trước train (bắt buộc)

- Sửa guard trong `training/contribution_study/runner.py`. Signature hiện chỉ ghi hash input và kiểm lần chạy trước trong cùng output, chưa đối chiếu bộ được duyệt.
- Kiểm đủ bốn SHA-256 với `results/component_ablation/locked_protocol.json` trước fit; sai hash/thiếu input phải dừng. Không tự nhận hash mới là bộ chuẩn. Smoke thật cũng qua guard.
- Test: bộ chuẩn qua; đổi một hash hoặc thiếu một input bị chặn trước train. Test tổng hợp dùng đường riêng có nhãn synthetic, không bypass guard dữ liệu thật và không được xuất scientific_result=true.

### DUC-REV-02 - Không báo hoàn tất khi thiếu artifact (bắt buộc)

- Sửa `run_neural`/`summarize_run` trong runner. Đã tái hiện: 8 báo cáo complete=false, fits=0, chỉ có fixed, metrics rỗng vẫn xuất complete=true, fits_expected=32, fits=0.
- Kiểm đúng tập variant/seed/outer, đủ fixed/balanced/sensitivity, complete từng fold, số inner/refit/fits kỳ vọng. Không dùng số file hoặc coverage nhóm đang có làm bằng chứng duy nhất.
- Kiểm ID/index/subject/label với dữ liệu nguồn, score/threshold hữu hạn, prediction khớp threshold; tái tính và đối chiếu fold/pooled metric. Giữ kiểm uniqueness/coverage hiện có.
- Resume chỉ bỏ qua fold có metadata/checkpoint/attention/history hợp lệ. Artifact thiếu/hỏng phải báo rõ hoặc chạy lại an toàn, không ghi complete=true, không trộn signature.
- Test âm: complete=false, sai fits, thiếu policy/artifact, sai nhãn/căn chỉnh, metric sai đều bị chặn. Test dương: run hợp lệ vẫn PASS. Fixture không được coi là bằng chứng train thật.

### DUC-REV-03 - Đồng bộ đúng GPU khi đo latency

- Sửa `benchmark_latency.py`: chuẩn hóa `torch.device`, kiểm device.type và đồng bộ đúng thiết bị CUDA. Hiện so sánh chuỗi đúng bằng cuda khiến cuda:0 không synchronize.
- Test CPU không gọi CUDA sync; cuda và cuda:0 đều đồng bộ đúng. Có thể mock trên máy CPU; chưa có GPU thì ghi benchmark GPU NOT_RUN, không giả kết quả.
- Số CPU đã nộp không chịu lỗi này; không bắt đo lại CPU chỉ vì sửa nhánh CUDA. Giữ warmup/eval/no-grad/batch/median/p95.

### DUC-REV-04 - Audit không được bỏ qua metric NaN

- Sửa `audit_existing.py`, hàm compare và đối chiếu seed summary: kiểm đủ khóa, uniqueness, giá trị hữu hạn trước tính sai lệch. Trường hợp metric không xác định hợp lệ cần quy ước rõ, không âm thầm bỏ qua.
- Kiểm bổ sung đã tái hiện: accuracy tái tính 0.9/0.8 so với bảng lưu NaN/0.8 vẫn PASS, max_abs_diff=0 do np.nanmax bỏ ô NaN. Không có bằng chứng bảng hiện tại chứa lỗi này; đây là yêu cầu làm checker đáng tin cậy.
- Test ô thiếu/NaN và khóa trùng phải FAIL; audit 07/08 nguyên trạng vẫn PASS, không thay nguồn kết quả cũ.

### DUC-REV-05 - Cập nhật báo cáo và cách tái lập

- Snapshot còn ghi code chưa commit dù đã có `1cbb778`: cập nhật commit, từng DUC-REV đã xử lý, lệnh/check/PASS/FAIL/NOT_RUN, hash và artifact kiểm chứng. Giữ lịch sử, không tự ghi DONE.
- Làm rõ lệnh kiểm/audit Q0-Q1 sau full. Audit hiện hard-code cặp A/T, chưa có lệnh dùng trực tiếp Q0/Q1; bổ sung entrypoint hoặc sửa mô tả đang hứa tái sử dụng để nêu đúng phần còn thiếu.
- Giữ parity Q0, shape/mask/augmentation, all-missing, gradient/leakage; chạy lại toàn bộ cùng checker sau sửa. Không yêu cầu Q1 phải thắng để nghiệm thu.

### Phạm vi, blocker và thứ tự tiếp tục

1. Sửa trên vinhduc trong [phạm vi Đức](../tasks/vinhduc.md), không sửa helper 07/checker/protocol chung/model triển khai hoặc answers.md. Các việc sửa/test và chuẩn bị notebook Kaggle làm được ngay, không chờ Tùng; web vẫn phụ.
2. Nguồn bốn input đúng hash còn chờ chủ repo/nhóm cung cấp nơi có quyền truy cập. Đây là phụ thuộc phía điều phối, không phải lý do quy lỗi Đức; không tạo tensor giả thay dữ liệu hoặc tự tải nguồn không rõ quyền.
3. Có input thì chạy validation, guard hash và smoke thật; báo phân bố q, mức clip/frame thiếu pose, min/mean/phân bố. Không đổi công thức theo outer score. Máy không GPU không phải lỗi thực hiện; không tự mua dịch vụ hoặc public dữ liệu.
4. Nộp lại code/báo cáo/log/check trên vinhduc, cập nhật cùng câu hỏi `DUC-20261028-001` dẫn commit và DUC-REV, mở PR theo quy tắc commit/push hiện hành. Chưa có input vẫn nộp review phần mềm, ghi smoke thật/full NOT_RUN.
5. Chỉ sau review bản sửa và smoke thật, main mới trả lời APPROVED riêng cho full Q0/Q1 24 outer / 96 fits. Hiện CHƯA được chạy full, thêm cost/loss/backbone hoặc thay deployment.

Điều kiện nhận lại: test hồi quy dương/âm cho DUC-REV-01/02/03/04, các lỗi không còn tái hiện, checker và bộ kiểm hiện có PASS, báo cáo cập nhật đúng. Không bắt làm lại phần đã đạt.

Đức và AI của Đức đọc phản hồi từ main, không cần merge để nhận chỉ dẫn:

```bash
git fetch origin
git show origin/main:docs/discussions/answers.md
```

Sau khi đọc, làm tiếp trên vinhduc, cập nhật báo cáo và cùng câu hỏi theo ID này. Không tự đánh dấu APPROVED hoặc push lên main.

## Mẫu trả lời - chưa phải quyết định

- ID câu hỏi / người hỏi: điền đúng ID từ file trao đổi cá nhân.
- Thời gian trả lời UTC+7 / người quyết định: điền rõ.
- Quyết định: `APPROVED`, `CHANGES_REQUESTED`, `REJECTED` hoặc `NEED_INFO`.
- Phạm vi được phép / điều kiện / lý do: điền cụ thể.
- Việc cần làm tiếp / bằng chứng cần nộp: điền rõ.
- Thay thế quyết định cũ (nếu có): dẫn ID/commit, không để hai quyết định mâu thuẫn không giải thích.

Giữ các câu trả lời còn liên quan trong snapshot để thành viên đối chiếu; lịch sử đầy đủ ở Git. Không hứa đã kiểm câu mới nếu chưa fetch/đọc nhánh của thành viên.
