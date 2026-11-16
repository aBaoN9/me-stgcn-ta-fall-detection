# Đức - Thay đổi có kiểm soát mô hình hiện tại

Cập nhật theo phân công của Gia Bảo ngày 30/10/2026. Nhánh `vinhduc`. [Báo cáo](../reports/vinhduc.md). Đọc [AGENTS](../../AGENTS.md), [bàn giao](../handoff.md), [protocol](../research-protocol.md) và quyết định mới trong [answers](../discussions/answers.md) trước khi sửa.

**Cập nhật quyền tự chủ 16/11/2026 - GB-20261116-002:** không cần chờ Gia Bảo duyệt M0, sửa thành phần kiến trúc hoặc từng lần smoke/full. Tự ghi/khóa phương án, kiểm tra rồi thực hiện; người điều phối review lúc bàn giao. Quy định này thay cổng chờ cũ, không thay mốc A3, dataset, đánh giá hoặc yêu cầu chất lượng. **Hạn 25/11/2026 (UTC+7): model + kết quả trên GitHub**, không chỉ đề xuất.

## Câu hỏi được giao

**Giữ bộ dữ liệu GMDCSA-24, lấy ME-STGCN-TA hiện tại làm mốc, đề xuất thay đổi có kiểm soát và so sánh với bản gốc theo cùng giao thức đánh giá.**

## Mốc chính xác: A3_motion_attention của notebook 07

Trong nhiệm vụ Đức, **"mô hình hiện tại làm mốc" chỉ đúng biến thể `A3_motion_attention`, policy `fixed`, thuộc notebook 07 và `training/component_ablation/`**. Không tự chọn một phiên bản ME-STGCN-TA khác vì cùng tên.

| Thành phần nhận diện | Nguồn/cấu hình gốc |
|---|---|
| Notebook thực nghiệm | [07-component-ablation.ipynb](../../notebooks/07-component-ablation.ipynb) |
| Khai báo biến thể | `VARIANTS["A3_motion_attention"]` trong [training.py](../../training/component_ablation/training.py): `channels=8`, `attention=True`, `fall_multiplier=1.0` |
| Kiến trúc | `SkeletonModel(channels=8, attention=True, dropout=0.3)` trong [models.py](../../training/component_ablation/models.py): bốn block ST-GCN 64/64/128/128, temporal attention và đầu phân loại hai lớp |
| Đầu vào | `(N,8,32,33)`, feature version `v2-mask-visibility-raw-pose-augmentation`; thứ tự kênh và xử lý mask/augmentation theo [features.py](../../training/component_ablation/features.py) |
| Recipe và input gốc | [locked_protocol.json](../../results/component_ablation/locked_protocol.json); AdamW, class-balanced cross-entropy, không tăng thêm chi phí Fall (`1.0` không có nghĩa bỏ cân bằng lớp) |
| Đánh giá chính | Nested LOSO 4 outer/3 inner, seed 42/142/242, policy `fixed` ngưỡng 0.5; chọn epoch từ inner, refit outer-train, không chọn theo outer-test |
| Nguồn mã để truy vết | Bộ `training/component_ablation/` tại commit `c8f654bf0f1943ffcfe375911c8b3e0015beadc5`; xác nhận parity nếu dùng runner riêng |

Kết quả lịch sử nhận diện đúng dòng `A3_motion_attention` / `fixed` trong [seed_mean_std.csv](../../results/component_ablation/seed_mean_std.csv): accuracy **89,58 ± 3,44%**, recall **91,98%**, specificity **87,24%**, F1 **89,72%**. Đây là bằng chứng cũ để đối chiếu, không phải kết quả thí nghiệm mới của Đức.

**Không lấy làm mốc thay thế:** model all-subject/ONNX của notebook 06 trong `models/deployment/`, kết quả 04/04b/05, `A3_cost_225`, nhánh timing T0/T1 hoặc bản quality-aware Q1. Không dùng checkpoint đã học cả 160 video để đánh giá như ngoài mẫu.

Đức chạy lại **bản gốc A3 này và bản thay đổi** trong cùng runner/môi trường/dữ liệu/split/seed/chính sách đánh giá phù hợp; không ghép số lịch sử làm đối chứng mới. M0 tự ghi và khóa trước outer-test để giải thích thay đổi, mốc/cặp so và ngân sách, không phải hồ sơ chờ duyệt. Recipe phía đối chứng phải tái lập mốc; thay đổi recipe phía ứng viên phải nằm trong yếu tố nghiên cứu đã ghi, không âm thầm đổi đối chứng. Đổi mốc hoặc dataset/giao thức chung vẫn ngoài phạm vi và cần hỏi.

## Quyền đề xuất thay đổi

Đức tự xác định vấn đề và đề xuất thay đổi có lý do, nói rõ giữ nguyên gì và thay đổi gì. Không giao sẵn công thức, siêu tham số hoặc cách triển khai; quality-aware pooling và ma trận Q0/Q1 cũ không còn là lời giải bắt buộc. Không thay bằng mô hình hoàn toàn khác như nhiệm vụ Tùng.

Được tự chọn thay đổi xử lý/huấn luyện/quyết định hoặc thành phần kiến trúc của A3, gồm pooling/attention; không cần xin mở rộng phạm vi riêng cho các thay đổi này. Phải nói rõ khi đã đổi kiến trúc, không gọi là giữ nguyên; không thay hẳn mô hình/backbone như nhiệm vụ Tùng. Mỗi cặp chỉ thay một yếu tố hoặc nhóm yếu tố có lý do và được khóa trước, không gộp rồi quy kết tác dụng riêng. Quality-aware pooling đã nộp được phép tự triển khai/kiểm chứng trong quyền này, không bị ép tiếp tục nếu thấy không phù hợp.

## Tự kiểm tra trước full và quyền thực hiện

1. Ghi M0/locked protocol và kế hoạch đối chứng trong vùng riêng trước outer-test; giữ cùng dataset/nhãn/split, seed 42/142/242, primary fixed và quy tắc chọn bằng inner. Không tinh chỉnh theo outer-test hoặc chọn seed đẹp.
2. Xác minh bốn input đúng hash/căn chỉnh và mốc A3; xử lý lỗi DUC-REV còn liên quan. Chạy test hồi quy/parity, dữ liệu/split/metric và smoke thật, lưu log PASS/FAIL. Fixture hoặc smoke giả không thay dữ liệu thật; thiếu input/test FAIL thì dừng phần phụ thuộc.
3. Khi các kiểm tra trên PASS và có môi trường/tài nguyên được phép, tự chạy full theo phương án đã khóa, không đợi người điều phối review lại code hoặc gửi APPROVED. Ghi partial/full/NOT_RUN đúng thực tế, không dùng quyền tự chủ để bỏ guard hoặc sửa completion cho đẹp.
4. Tự commit/push code, kiểm tra, phương án, báo cáo và model/kết quả trong phạm vi lên `vinhduc`, mở PR theo quyền Gia Bảo cấp; không xin lại từng lần. Model lớn upload GitHub Release asset có link/hash theo checklist `GB-20261116-001`. Review cuối vẫn quyết định nghiệm thu và merge main.
5. Báo sớm nếu thiếu input/quyền/GPU hoặc gặp lỗi. Không chờ phê duyệt cho việc đã được cấp; vẫn hỏi trước khi đổi dataset/mốc/giao thức chung, thay hẳn mô hình, tiêu tiền hoặc sửa ngoài vùng nhiệm vụ. Không sửa deployment, bật PoseLab hay push nhánh khác.

## Phạm vi được ghi

- `training/contribution_study/**`: mã thực nghiệm/kiểm tra/audit, hướng dẫn và notebook riêng nếu cần.
- `results/contribution_study/**`: bằng chứng và run mới tách khỏi kết quả gốc.
- `docs/research/contribution-study.md`: vấn đề, phương án, phần giữ nguyên/thay đổi và so sánh.
- `docs/reports/vinhduc.md`, `docs/discussions/vinhduc.md`: tiến độ và trao đổi cá nhân theo [quy trình](../discussions/README.md).

Không sửa phần Tùng, nguồn/kết quả cũ, helper chung, checker, protocol/Word hoặc model. PoseLab và WEB-01 đã hủy; quyền sửa `app/static/**` và `app/README.md` trước đây không còn. Không tự làm UI khi chờ dữ liệu/GPU.

## Mốc thực hiện

1. **DUC-01 - Hiểu mốc gốc:** đối chiếu mã/cấu hình A3_motion_attention của 07 đã chỉ định, dùng 06/08 làm bối cảnh, không tự thay mốc. Giữ các kiểm tra metric/dữ liệu có ích đã làm; không chép kết quả cũ thành kết quả mới.
2. **DUC-02 / M0 - Tự ghi phương án:** nêu vấn đề, giả thuyết, phần giữ nguyên/thay đổi, cặp so, tiêu chí và ngân sách; khóa trước outer-test và ghi `M0_DOCUMENTED`, không chờ duyệt. Không cần chọn lại quality-aware nếu không phù hợp.
3. **DUC-03 / M1-M2 - Tự kiểm chứng:** triển khai, tự kiểm test/smoke thật; đạt điều kiện thì tự chạy full cặp gốc/thay đổi theo protocol đã khóa. Kiểm tra tắt thay đổi có khôi phục đối chứng khi áp dụng được; lưu raw OOF/config/seed/hash/log/FN/FP. Chỉ đổi ngưỡng thì giữ cùng score và báo thay chính sách quyết định, không phải kiến trúc mới.
4. **DUC-04 / M3 - Phân tích và bàn giao:** kết luận thay đổi giúp gì, không giúp gì và giới hạn. Có thể đối chiếu kết quả Tùng khi đã review, nhưng phần kiểm chứng của Đức không phải chờ Tùng mới làm được.

## Bài làm và phản hồi đã có

Review `DUC-20261028-001` vẫn là checklist lỗi cần tự xử lý/test nếu dùng tiếp mã; không còn là cổng chờ người điều phối duyệt lại trước chạy. `GB-20261116-002` trả lời quyền thực hiện cho cả câu này và `DUC-20261103-001`: được tự tiếp tục khi kiểm tra đạt, không đồng nghĩa bản sửa/metric đã nghiệm thu. Trong file câu hỏi cá nhân, ghi đã nhận quyền, tách các yêu cầu dữ liệu/GPU còn mở; không xóa lịch sử hoặc gắn `RESOLVED` cho blocker chưa hết.

## Tiêu chí hoàn tất

- [ ] Mốc gốc đúng, M0/phương án có giới hạn rõ và đã ghi/khóa trước outer-test; không cần phê duyệt thủ công.
- [ ] Kiểm tra dữ liệu/split/metric và smoke có bằng chứng; lỗi đã review được xử lý nếu mã vẫn dùng.
- [ ] Có cặp gốc/thay đổi đánh giá như nhau, đủ dự đoán ngoài mẫu và tổng hợp theo protocol.
- [ ] Báo cả cải thiện, suy giảm và FN/FP; không dùng outer-test để chọn thay đổi.
- [ ] Checker repo PASS; báo cáo, lệnh tái lập và phần chưa chạy rõ ràng. Kết quả không cải thiện vẫn hợp lệ.

Đầu phiên fetch và đọc `origin/main:docs/discussions/answers.md`. Thiếu input/GPU thì báo lý do, tiếp tục việc độc lập và ghi full `NOT_RUN`; không tự mua dịch vụ, đổi dữ liệu, sửa deployment hoặc push nhánh Tùng/main.
