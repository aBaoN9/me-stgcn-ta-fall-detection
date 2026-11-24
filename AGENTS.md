# Quy tắc làm việc cho AI và thành viên

**Phân công mới 24/11/2026 (`GB-20261124-001`):** Tùng triển khai/train CascadeFormer, DHA-eGCN, Hyper-GCN, ProtoGCN, InfoGCN++; Đức triển khai/train SkateFormer, BlockGCN, DeGCN, JT-GraphFormer, DS-GCN. Đọc [hợp đồng và thiết lập](docs/research/modern-model-benchmark.md) trước khi làm. Cả hai tự test/smoke/full sau khi khóa M0, không chờ duyệt từng bước. Cùng đầu vào v2 `(N,8,32,33)`, recipe CE chung, nested LOSO/seed/metric với A3; đây là adaptation, không tái lập nguyên recipe paper. Quyết định này thay giới hạn cũ "Đức chỉ sửa A3" trong track mới; giữ nguyên bài/kết quả cũ, không coi các model mới đã chạy.

**Cập nhật 23/11/2026 (`GB-20261123-001`):** đã nghiệm thu/merge PR #4 SkateFormer adapted small của Tùng, head `20c67f5`, merge `3c6a39e`; nhận kết quả nghiên cứu bổ sung âm, không thay deployment hoặc yêu cầu train lại. Theo yêu cầu mới Gia Bảo, [FallScope](demo_streamlit/README.md) được tích hợp làm prototype Streamlit local với năm model; điều này thay trạng thái hoãn demo mới, không khôi phục PoseLab hoặc giao lại WEB-01/WEB-02. Tuning A3 trên `giabao` là track riêng, không nhập runner/bộ lưu PARTIAL vào lần cập nhật main này. Các thông báo có ngày cũ bên dưới là lịch sử; quyền tự chủ/nhiệm vụ/hạn Đức giữ nguyên.

**Quyết định mới 21/11/2026 (`GB-20261121-002`):** Gia Bảo yêu cầu gỡ PoseLab: xóa mã `app/` khỏi repo hiện hành, demo mới để sau (`DEFERRED`). Quyết định này thay chỉ dẫn giữ nguồn app ở các thông báo cũ, không cho phép xóa model, dữ liệu, kết quả, notebook, Word hoặc lịch sử. [Kế hoạch tuning A3](docs/research/a3-tuning-plan.md) là track Gia Bảo, `PLANNED / NOT_RUN`; chưa giao chạy train. Không thay nhiệm vụ/quyền tự chủ/hạn 25/11 của Đức, không bắt Tùng train lại, không hứa vượt TCN/GRU.

**Nghiệm thu bổ sung 21/11/2026 (`GB-20261121-001`):** đã nhận GRU/CTR-GCN adapted trong PR #3 của Tùng, head `5965c3e`, tích hợp cùng bản sửa monitor tự tạo thư mục cha `.runtime` và ba test riêng. Run `results/backbone_benchmark/tung-gru-ctr-rtx3050-20261120-233932/` có 96 fits, 24 checkpoint full và 2.880 dòng dự đoán mới; mốc TCN–A3 giữ nguyên. Không yêu cầu train lại vì sửa tiện ích ngoài code lock; không đổi deployment, PoseLab, Word hoặc nghiệm thu thay bài Đức. Xem README/answers để không giao lại việc đã hoàn tất.

**Nghiệm thu 20/11/2026 (`GB-20261120-001`):** benchmark TCN–A3 của Tùng đã được nhận M0–M3 và merge PR #2 vào main, trạng thái `ACCEPTED_BENCHMARK`. Code/checkpoint/kết quả xem README và `results/backbone_benchmark/tung-m2-rtx3050-20261120-165523/`; không giao lại như chưa làm. Không thay model/threshold triển khai hoặc mốc của Đức, không bật PoseLab. Các hướng bổ sung là nghiên cứu tiếp, không là điều kiện chặn nghiệm thu.

**Bổ sung 19/11/2026 (`GB-20261119-002`):** cả Tùng và Đức được tự full CPU nếu không có GPU, hai vế cùng thiết bị/recipe/ngân sách; tự ghi M0 addendum và khóa config/code mới trước outer-test, không xin lại. Giữ guard/test/smoke thật và dùng output mới khi đổi chữ ký. Tool kiểm/tải input, cách Kaggle pin code cũ và checklist nộp ở [bắt đầu train](docs/training-start.md).

Repo này tập trung nghiên cứu Fall/ADL. Đọc trước khi sửa; áp dụng toàn repo.

**Quy định mới 18/11/2026:** theo `GB-20261118-001` (Tùng) và `GB-20261116-002` (Đức) trong [answers](docs/discussions/answers.md), **cả hai được tự chọn phương án trong nhiệm vụ, triển khai, smoke và full train, không chờ Gia Bảo duyệt M0/kiến trúc/từng lần chạy**. Ghi và khóa phương án trước outer-test; tự kiểm tra dữ liệu/test/smoke thật đạt trước full. Review là nghiệm thu sản phẩm sau, không phải cổng chờ trước thực hiện. Tùng được bắt đầu TCN so với A3 từ M0 đã nộp, tự điều chỉnh hoặc dùng phương án dự phòng trước outer-test có ghi lý do; không xin lại từng bước.

**Hạn nộp đã chốt:** thứ Tư, **25/11/2026 (UTC+7)**. Sản phẩm của cả hai là **model đã train và kết quả đánh giá** trên GitHub, kèm mã/lệnh tái lập và báo cáo theo `GB-20261116-001`. Cả hai không chờ phê duyệt trong phạm vi đã giao; thiếu input/GPU hoặc kiểm tra lỗi phải báo sớm, nộp phần đã làm và blocker đúng hạn. Chưa có sản phẩm thật không đánh dấu hoàn tất; không bịa số liệu hoặc thay model gốc.

## Phân công hiện hành - cập nhật ngày 24/11/2026

- Tùng / `dinhtung`: năm model 1-5 trong [phân công](docs/research/modern-model-benchmark.md); ghi `training/modern_benchmark_tung/`, `results/modern_benchmark_tung/`, thiết kế `docs/research/modern-benchmark-tung.md` và báo cáo/trao đổi cá nhân.
- Đức / `vinhduc`: năm model 6-10, được thay backbone trong track mới; ghi `training/modern_benchmark_duc/`, `results/modern_benchmark_duc/`, thiết kế `docs/research/modern-benchmark-duc.md` và báo cáo/trao đổi cá nhân. Bài contribution cũ giữ riêng.
- Cả hai giữ mốc `A3_motion_attention` / `fixed` của notebook 07, chạy A3 đối chứng cùng môi trường. Không dùng checkpoint final 06, A3_cost_225, Q1 hoặc tuning chưa nghiệm thu thay mốc.
- Hợp đồng mới khóa cùng tensor tám kênh/32 mốc/33 khớp, dữ liệu/hash/nhãn, nested LOSO, seed 42/142/242, threshold 0.5, F1 lớp Fall và mean/sample SD. Không đổi sang YOLO, pretrained, ensemble hay tự lấy Macro-F1 thay F1.
- M0/source/license/adaptation phải ghi trước outer-test; test và smoke thật PASS rồi tự full, commit/push nhánh mình và mở PR. Không chờ duyệt từng bước, không tự thay danh sách hoặc bỏ guard.
- Nghiệm thu cũ không bị thu hồi; SkateFormer adapted small cũ không tính là bản mới. Hạn bài cũ 25/11/2026 giữ nguyên, chưa chốt hạn riêng cho toàn bộ 10 model; báo tiến độ/blocker tại mốc đó, không bịa kết quả cho đủ.
- Chỉ import helper chung, không sửa nguồn/kết quả cũ hoặc phần bạn kia. FallScope đã được phép theo `GB-20261123-001`, nhưng đợt giao việc này không giao web/demo/deployment; PoseLab và WEB-01/02 vẫn hủy.
- Các quyết định cũ bên dưới và trong tài liệu lịch sử chỉ còn hiệu lực ở phần không xung đột `GB-20261124-001`; quyền thực thi không đồng nghĩa đã nghiệm thu.

## Xác định người và đọc đúng nhiệm vụ

- Chưa xác định người trong cuộc trò chuyện: **hỏi "Bạn là Gia Bảo (người điều phối), Tùng hay Đức?" trước khi sửa/chạy train**. Không đoán từ tài khoản Git/máy/nhánh.
- Tùng: nhánh `dinhtung`, [nhiệm vụ](docs/tasks/dinhtung.md), [báo cáo](docs/reports/dinhtung.md).
- Đức: nhánh `vinhduc`, [nhiệm vụ](docs/tasks/vinhduc.md), [báo cáo](docs/reports/vinhduc.md).
- Gia Bảo là chủ repo/người điều phối: phát triển kỹ thuật trên `giabao`, [nhiệm vụ](docs/tasks/giabao.md), [báo cáo](docs/reports/giabao.md). `main` giữ bản chuẩn đã duyệt và quyết định chung trong `docs/discussions/answers.md`; không phải nhánh thử nghiệm cá nhân. Người khác cần người điều phối giao việc, không tự nhận phần của hai bạn.
- Câu hỏi danh tính có thể hỏi ngay trong chat. Sau khi biết người, các câu hỏi cho nhóm/người điều phối phải ghi vào [kênh Markdown](docs/discussions/README.md), không chỉ nhắn trong chat rồi coi nhóm đã nhận.
- Vai trò đã rõ thì không hỏi lại mỗi lượt. Sau khi đọc, tóm tắt hiện trạng, nhiệm vụ, phạm vi sửa, tiêu chí xong và bước đầu tiên.
- Thứ tự đọc: [README](README.md), [bàn giao](docs/handoff.md), [bảng việc](docs/tasks/README.md), nhiệm vụ cá nhân, [protocol](docs/research-protocol.md), [phương pháp](docs/methodology.md), [ablation](docs/component-ablation.md), [timing](docs/timing-ablation.md), rồi code/kết quả liên quan. Không bắt người dùng đọc hết thay AI.

<!-- CODEGRAPH_START -->
## CodeGraph

Nếu repo có `.codegraph/`, dùng `codegraph_explore` hoặc `codegraph explore` trước grep/find/đọc code để định vị/hiểu mã. Source trả về coi như đã đọc. Repo clone không có index thì bỏ qua, không tự index.
<!-- CODEGRAPH_END -->

## Git và ranh giới

- Trước khi làm: `git status --short`, `git branch --show-current`, `git remote -v`. Thay đổi không rõ nguồn gốc: hỏi trước; không reset/clean/stash/ghi đè hộ người khác.
- Thành viên phải ở đúng nhánh trước khi sửa; chưa chuyển an toàn được thì dừng sửa. Gia Bảo sửa code/thử nghiệm trên `giabao`, chỉ tích hợp vào main sau review/kiểm tra và yêu cầu của chủ repo. Tài liệu bàn giao/quy tắc/quyết định điều phối được cập nhật trực tiếp trên main; publish vẫn cần yêu cầu của chủ repo.
- AI làm cùng Gia Bảo được đọc/fetch nhánh Tùng/Đức để review, không cần merge vào giabao chỉ để xem. Không tự sửa hoặc push hai nhánh cá nhân đó; giabao không tự đồng nghĩa code đã được duyệt vào main.
- Chỉ ghi vùng trong nhiệm vụ; nguồn cũ được đọc/import, không sửa loop train chung để tiện riêng mình.
- Thành viên không push trực tiếp main, force-push, tự merge hoặc thay model/threshold triển khai. Người điều phối duyệt tích hợp; publish main chỉ khi chủ repo yêu cầu.
- Ngoài quyền nộp sản phẩm đã cấp ở mục kế tiếp, không tự commit/push code khi người dùng chưa yêu cầu. **Ngoại lệ trao đổi được chủ repo cho phép sẵn:** mỗi lần hỏi nhóm, cập nhật/ghi đè file trao đổi cá nhân hiện tại, tạo commit thường và push nhánh được giao, không phải hỏi lại quyền cho lần gửi Markdown đó. Chỉ stage file trao đổi; kiểm cả commit chưa push để không gửi kèm code ngoài yêu cầu. Không force-push, không commit hộ thay đổi khác. Không đưa token, video/tensor riêng, môi trường hoặc checkpoint lớn vào Git.
- **Quyền nộp sản phẩm đã cấp:** Tùng/Đức và AI tương ứng được commit/push code, kiểm tra, M0, báo cáo và model/kết quả trong vùng nhiệm vụ lên `dinhtung`/`vinhduc`, upload artifact theo `GB-20261116-001` và mở PR, không hỏi lại Gia Bảo mỗi lần. Chỉ stage/outgoing đúng phạm vi; không vượt cơ chế phê duyệt công cụ, không push nhánh khác/main, tự merge, sửa nguồn/kết quả gốc hoặc model triển khai.
- Mở PR về main; báo cáo trên nhánh được đọc qua PR/fetch, không tự xuất hiện trên main.

**Ngoại lệ dữ liệu được chủ repo yêu cầu publish ngày 19/11/2026 (`GB-20261119-001`):** bốn input chuẩn của notebook 01 và manifest metadata trong `data/gmdcsa24/` được chia sẻ trên main, đúng hash 07; dùng `--input-root data/gmdcsa24`, xem [hướng dẫn dữ liệu](data/README.md). Đây không phải quyền upload video/tensor riêng khác. Không tái tạo hoặc sửa nhãn/split/hash; giữ nguồn và điều kiện sử dụng.

## Trao đổi nhóm qua Markdown

- Tùng dùng `docs/discussions/dinhtung.md` trên `dinhtung`; Đức dùng `docs/discussions/vinhduc.md` trên `vinhduc`. Đọc [quy trình](docs/discussions/README.md) và giữ toàn bộ câu chưa giải quyết trong bản hiện tại, không xóa mất câu khác khi ghi đè.
- Câu hỏi phải có ID, thời gian UTC+7, ngữ cảnh/file/log, đã thử gì, lựa chọn đề xuất và quyết định cần nhóm trả lời. Không ghi mật khẩu, token, dữ liệu riêng hoặc log có bí mật.
- Mỗi lần gửi/cập nhật câu hỏi: commit/push bình thường và kiểm push thành công. Không có quyền/mạng/auth thì báo lỗi, không nói đã gửi; không vượt cơ chế phê duyệt của công cụ.
- Người điều phối trả lời trong `docs/discussions/answers.md` trên main, dẫn đúng ID. Đầu mỗi phiên và trước việc đang chờ duyệt, fetch và đọc file trả lời; không tự xem im lặng là đồng ý.
- Cả Tùng và Đức không đặt `WAITING_FOR_MAIN` cho M0/chọn phương án/sửa kiến trúc/smoke/full/commit-push trong phạm vi đã cấp. Ghi `IN_PROGRESS`, `M0_DOCUMENTED` hoặc `BLOCKED` theo tình trạng thật; chỉ chờ khi ngoài phạm vi hoặc cần input/quyền truy cập/tài nguyên. Các câu hỏi còn lại vẫn theo quy trình, không suy im lặng là đồng ý.

## PoseLab đã gỡ, demo mới để sau

- [WEB-01/WEB-02](docs/tasks/web-app.md) có trạng thái `CANCELLED`, không phải việc phụ khi chờ GPU. Ngoại lệ cho Đức sửa frontend và Tùng QA web đã bị thu hồi.
- Mã `app/` đã gỡ theo `GB-20261121-002`; truy vết mã cũ trong lịch sử Git. Giữ `models/deployment/`, dữ liệu, kết quả, notebook và Word; không tự cài thư viện/tải model để chạy demo, thay model/threshold hoặc xóa dữ liệu lịch sử.
- Demo mới `DEFERRED`, chỉ triển khai khi Gia Bảo yêu cầu mới; tài liệu hoặc nhánh cũ còn hướng dẫn chạy không phải lệnh khôi phục.

## Liêm chính nghiên cứu

- 160 video, 4 subject; không chia ngẫu nhiên frame/clip để test, không gọi ba seed là 480 mẫu độc lập.
- Epoch/hyperparameter/cost/ngưỡng chỉ chọn bằng inner validation theo subject, không dùng outer-test hoặc clip lỗi đã biết.
- Kết quả 06/07/08 không phải kết quả code mới; không ghi đè bằng chứng cũ trong results.
- Không mặc định backbone mới hơn là tốt hơn; không lấy số paper trên dataset khác làm đối chứng trực tiếp. Ghi adaptation/khác biệt paper.
- Không hứa accuracy 95%, chọn seed đẹp, giấu FN/FP, đổi nhãn vì model sai hoặc dùng smoke như full.
- Chưa chạy: `NOT_RUN`; thiếu input/GPU: báo lý do/hỗ trợ cần thiết. Không tạo số liệu giả.
- Khóa phương pháp trước held-out. Outer folds cũ đã được xem trong phát triển; phải công bố giới hạn.

## README và nhật ký

- `README.md` ở gốc mô tả hiện trạng mới nhất trên main: model/kết quả đã kiểm chứng, cách chạy và giới hạn. Khi thay đổi, sửa trực tiếp mục/bảng tương ứng; không thêm các khối cập nhật theo ngày chồng lên đầu README.
- Diễn biến cũ ghi trong `docs/nhat-ky/YYYY-MM.md`, có commit/PR/run, bằng chứng kiểm tra và điều không thay đổi; mục lục ở [nhật ký](docs/nhat-ky/README.md). Khi tái cấu trúc lớn, lưu bản README theo commit nguồn, sửa liên kết tương đối cho đúng vị trí.
- Không xóa hoặc sửa source, notebook, Word, checkpoint/lock/predictions đã công bố chỉ để dọn README. Không trộn kết quả/latency các lượt khác nhau hoặc dùng archive PARTIAL như model đã nghiệm thu; lịch sử phải được ghi rõ là lịch sử.

## Kiểm tra và bàn giao

- Kiểm tra nhẹ trước; cả Tùng và Đức tự chạy full khi phương án đã khóa, input/hash/split đúng, test và smoke thật PASS, đủ tài nguyên có quyền sử dụng; không cần duyệt thủ công. Tự chọn Kaggle, GPU local hoặc CPU theo `GB-20261119-002`, có quyền sử dụng trong hạn mức miễn phí/đã có, không cần xin chọn thiết bị; ghi thiết bị và đo cặp so cùng điều kiện. Không bỏ kiểm tra, tự cài lại CUDA/PyTorch hệ thống, tiêu tiền dịch vụ hoặc tải dataset không rõ quyền.
- Chạy `python scripts/check_repository.py` trước bàn giao. Checker khóa 9 notebook cũ; notebook mới để trong training riêng, không sửa checker để bỏ lỗi.
- Code mới cần check shape/mask/split leakage/OOF coverage/tái tính metric. Ghi lệnh, phiên bản, hash input-code-config, seed, thiết bị, PASS/FAIL/NOT_RUN.
- Cập nhật báo cáo cá nhân mỗi phiên: đã làm, file/commit/run/PR, bằng chứng, kiểm tra, lỗi, còn thiếu, việc main cần quyết định.
- Chỉ `READY_FOR_REVIEW` khi đủ tiêu chí. Kết quả âm hợp lệ nếu đủ thí nghiệm, không phải lý do bỏ thực nghiệm.
