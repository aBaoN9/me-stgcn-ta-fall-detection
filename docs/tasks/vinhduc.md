# Đức - Thay đổi có kiểm soát mô hình hiện tại

Cập nhật theo phân công của Gia Bảo ngày 30/10/2026. Nhánh `vinhduc`. [Báo cáo](../reports/vinhduc.md). Đọc [AGENTS](../../AGENTS.md), [bàn giao](../handoff.md), [protocol](../research-protocol.md) và quyết định mới trong [answers](../discussions/answers.md) trước khi sửa.

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

Đức chạy lại **bản gốc A3 này và bản thay đổi** trong cùng runner/môi trường/dữ liệu/split/seed/chính sách đánh giá phù hợp; không chỉ ghép số lịch sử vào bảng mới rồi gọi là đối chứng cùng điều kiện. M0 chỉ cần giải thích thay đổi, xác nhận tái lập mốc và cặp so; nếu thật sự cần đổi mốc/recipe, xin Gia Bảo duyệt và nêu rõ đây là track khác. Chốt mốc không có nghĩa giao sẵn cách cải tiến.

## Quyền đề xuất thay đổi

Đức tự xác định vấn đề và đề xuất thay đổi có lý do, nói rõ giữ nguyên gì và thay đổi gì. Không giao sẵn công thức, siêu tham số hoặc cách triển khai; quality-aware pooling và ma trận Q0/Q1 cũ không còn là lời giải bắt buộc. Không thay bằng mô hình hoàn toàn khác như nhiệm vụ Tùng.

Giữ kiến trúc hiện tại trong hướng chính để kiểm chứng tác động của thay đổi xử lý/huấn luyện/quyết định. Nếu muốn sửa thành phần kiến trúc, phải nêu rõ là mở rộng phạm vi và xin Gia Bảo duyệt trước; không gọi là giữ nguyên kiến trúc. Mỗi cặp so sánh chỉ thay một yếu tố hoặc nhóm thay đổi đã được giải thích và duyệt, không gộp nhiều thay đổi rồi quy kết tác dụng riêng.

## Phạm vi được ghi

- `training/contribution_study/**`: mã thực nghiệm/kiểm tra/audit, hướng dẫn và notebook riêng nếu cần.
- `results/contribution_study/**`: bằng chứng và run mới tách khỏi kết quả gốc.
- `docs/research/contribution-study.md`: vấn đề, phương án, phần giữ nguyên/thay đổi và so sánh.
- `docs/reports/vinhduc.md`, `docs/discussions/vinhduc.md`: tiến độ và trao đổi cá nhân theo [quy trình](../discussions/README.md).

Không sửa phần Tùng, nguồn/kết quả cũ, helper chung, checker, protocol/Word hoặc model. PoseLab và WEB-01 đã hủy; quyền sửa `app/static/**` và `app/README.md` trước đây không còn. Không tự làm UI khi chờ dữ liệu/GPU.

## Mốc thực hiện

1. **DUC-01 - Hiểu mốc gốc:** đối chiếu mã/cấu hình A3_motion_attention của 07 đã chỉ định, dùng 06/08 làm bối cảnh, không tự thay mốc. Giữ các kiểm tra metric/dữ liệu có ích đã làm; không chép kết quả cũ thành kết quả mới.
2. **DUC-02 / M0 - Đề xuất:** nêu vấn đề, giả thuyết, thay đổi định kiểm chứng, phần giữ nguyên, cặp đối chứng, tiêu chí và ngân sách. Gửi Gia Bảo thống nhất trước full train; không cần tự chọn lại quality-aware nếu không phù hợp.
3. **DUC-03 / M1-M2 - Kiểm chứng:** triển khai và smoke; chạy mốc gốc cùng bản thay đổi theo protocol đã duyệt. Kiểm tra việc tắt thay đổi có khôi phục hành vi đối chứng khi áp dụng được; lưu raw OOF, cấu hình/seed/hash/log, FN/FP và đánh đổi. Nếu chỉ đổi ngưỡng, giữ cùng score và báo đây là thay chính sách quyết định, không phải kiến trúc mới.
4. **DUC-04 / M3 - Phân tích và bàn giao:** kết luận thay đổi giúp gì, không giúp gì và giới hạn. Có thể đối chiếu kết quả Tùng khi đã review, nhưng phần kiểm chứng của Đức không phải chờ Tùng mới làm được.

## Bài làm và phản hồi đã có

Review `DUC-20261028-001` còn giá trị về lỗi kiểm dữ liệu, nghiệm thu full, NaN và đo latency nếu tiếp tục dùng các mã đó. Quyết định `GB-20261030-001` thay phần ép hướng/công thức Q0/Q1, không xóa phát hiện lỗi hoặc tự biến `CHANGES_REQUESTED` thành `APPROVED`. Giữ lịch sử, nêu phần tái sử dụng và xin duyệt phương án tiếp theo; không tự bỏ toàn bộ bài làm đã có.

## Tiêu chí hoàn tất

- [ ] Mốc gốc và phương án M0 được duyệt, thay đổi có giới hạn rõ.
- [ ] Kiểm tra dữ liệu/split/metric và smoke có bằng chứng; lỗi đã review được xử lý nếu mã vẫn dùng.
- [ ] Có cặp gốc/thay đổi đánh giá như nhau, đủ dự đoán ngoài mẫu và tổng hợp theo protocol.
- [ ] Báo cả cải thiện, suy giảm và FN/FP; không dùng outer-test để chọn thay đổi.
- [ ] Checker repo PASS; báo cáo, lệnh tái lập và phần chưa chạy rõ ràng. Kết quả không cải thiện vẫn hợp lệ.

Đầu phiên fetch và đọc `origin/main:docs/discussions/answers.md`. Thiếu input/GPU thì báo lý do, tiếp tục việc độc lập và ghi full `NOT_RUN`; không tự mua dịch vụ, đổi dữ liệu, sửa deployment hoặc push nhánh Tùng/main.
