# Tùng - Đối chứng backbone trên dữ liệu nhóm

Nhánh `dinhtung`. [Báo cáo](../reports/dinhtung.md). Đọc [AGENTS](../../AGENTS.md), [bàn giao](../handoff.md), [protocol](../research-protocol.md) trước khi sửa.

## Mục tiêu và phạm vi

Trả lời: cùng GMDCSA-24 và điều kiện đánh giá, CTR-GCN tham chiếu hơn/kém local ST-GCN-style ở sensitivity/specificity, độ ổn định theo người và chi phí nào? Không có nhiệm vụ "chạy cho bằng được số tốt hơn".

Chỉ được ghi:

- `training/backbone_benchmark/**`: adapter, model registry, runner, validation, dependencies riêng, notebook Kaggle nếu cần.
- `results/backbone_benchmark/**`: mỗi run một thư mục, chỉ bằng chứng gọn đưa Git.
- `docs/research/backbone-benchmark.md`: thiết kế, related work, bảng kết quả/kết luận.
- `docs/reports/dinhtung.md`: tiến độ/bàn giao cá nhân.
- `docs/discussions/dinhtung.md`: câu hỏi nhóm; cập nhật cùng file và commit/push mỗi lần hỏi theo [quy trình](../discussions/README.md).
- Nếu còn thời gian: `docs/research/web-app-qa.md` cho [WEB-02](web-app.md), chỉ review/QA, không cùng sửa UI với Đức.

Không sửa phần Đức, app/deployment, notebook/kết quả cũ, checker, README gốc/protocol chung. Đọc/import code cũ được; cần thay ngoài phạm vi thì hỏi qua Markdown, push và chờ duyệt. Đọc `origin/main:docs/discussions/answers.md` mỗi đầu phiên và trước phần đang chờ.

## TUNG-01 - Thiết kế, nộp M0

1. Đọc `training/component_ablation/models.py`, `features.py`, `training.py`, ablation 07. Mô tả backbone/head local, không gắn nhãn nguyên bản paper.
2. Đọc paper/repo chính chủ CTR-GCN trong protocol; ghi commit/giấy phép và adaptation: graph 33 khớp, 32 mốc, 2 lớp, 3/8 kênh, single-person, single-stream, scratch.
3. Không load weights graph 25 khớp/3D vào 33 khớp/2D mà không mapping. Chính: train scratch; pretrain là track riêng cần duyệt.
4. Nộp config/shape/pooling/loss/ngân sách/điều kiện dừng; người điều phối duyệt trước full run.

## TUNG-02 - Runner và smoke

Runner riêng, reuse/import feature/split/metric cũ khi phù hợp, không sửa loop cũ tại chỗ. Registry nhận channels/backbone/pooling rõ, trả logits Fall/ADL. Model không có attention thì không bịa attention để vẽ.

Kiểm synthetic forward/backward 3/8 kênh, logits shape, finite loss/gradient, graph index, train-val-test subject disjoint, class weight chỉ fit train, augmentation chỉ train, seed, preprocessing parity và metric từ confusion matrix. Smoke riêng, ghi lệnh thực chạy.

## TUNG-03 - Benchmark đầy đủ

Ma trận chính tối thiểu, khóa tên tương ứng trong config:

| Mã | Backbone | Input | Pooling | Fall multiplier |
|---|---|---|---|---|
| B0 | Local ST-GCN-style, đối chứng A0 | 3 kênh | Average | 1,0 |
| B1 | Local ST-GCN-style, đối chứng A1 | 8 kênh | Average | 1,0 |
| B2 | CTR-GCN adapted | 3 kênh | Average | 1,0 |
| B3 | CTR-GCN adapted | 8 kênh | Average | 1,0 |

Giữ normalization/head/dropout/augmentation tương đương ở mức kiểm soát được; khai báo phần khác nội tại backbone. Nested LOSO, ba seed, fixed recipe, primary fixed 0,5 theo protocol. Bốn biến thể: 48 outer, 192 fits nếu không search thêm; ghi số thực, không hứa runtime theo notebook cũ.

So B0/B2 và B1/B3 để xét backbone; B0/B1 và B2/B3 để xét motion. A3/cost 07 chỉ là tham khảo hệ thống đầy đủ, không phải cặp thuần đổi backbone. Track paper recipe nếu có báo riêng/công bằng ngân sách.

Xuất artifact/schema chung, parameters/latency median-p95 cùng máy/backend, lỗi FN/FP theo subject/mẫu. Tất cả bảng phải tái tạo từ CSV.

## TUNG-04 - Bàn giao

Viết `docs/research/backbone-benchmark.md`: nguồn/adaptation, khác recipe, đủ seed/subject, chi phí, hạn chế và kết luận trung thực. CTR-GCN kém hơn thì phân tích, không giấu.

Báo cáo chứa code/result/config/hash/log link/lệnh tái lập; mở PR về main khi người dùng cho phép push. Đức dùng kết quả từ PR/commit duyệt, không cần bạn sửa hộ phần Đức.

## Tiêu chí hoàn tất

- [ ] M0 duyệt, nguồn/giấy phép/adaptation rõ, đúng branch/phạm vi.
- [ ] Code chạy được, validation/smoke PASS, hướng dẫn local/Kaggle.
- [ ] Đủ B0-B3, 4 subject x 3 seed, OOF/schema/metric duy nhất/đủ coverage.
- [ ] Chi phí đo cùng điều kiện, bảng lỗi/kết luận không cherry-pick.
- [ ] Checker repo PASS; báo cáo/PR có bằng chứng, phần thiếu ghi rõ.

Thiếu input/GPU/giấy phép -> `BLOCKED` cụ thể sau khi làm phần thiết kế/kiểm tra khả thi. Notebook chưa chạy không là hoàn tất.
