# Đức - Kiểm chứng cải tiến và bằng chứng học thuật

Nhánh `vinhduc`. [Báo cáo](../reports/vinhduc.md). Đọc [AGENTS](../../AGENTS.md), [bàn giao](../handoff.md), [protocol](../research-protocol.md) trước khi sửa.

## Mục tiêu và phạm vi

Chứng minh motion/attention/cost đã giúp gì, thử **một** thay đổi có kiểm soát xử lý pose không tin cậy, kết luận cải thiện/trade-off bằng số. Không mặc định đề xuất phải hơn mọi backbone.

Chỉ được ghi:

- `training/contribution_study/**`: audit OOF, quality-aware model/runner, validation, notebook Kaggle nếu cần.
- `results/contribution_study/**`: audit/bảng/ảnh và run mới tách thư mục.
- `docs/research/contribution-study.md`: thiết kế/bản thảo Markdown để main đưa Word.
- `docs/reports/vinhduc.md`: tiến độ/bàn giao cá nhân.
- `docs/discussions/vinhduc.md`: câu hỏi nhóm; cập nhật cùng file và commit/push mỗi lần hỏi theo [quy trình](../discussions/README.md).
- Nếu còn thời gian: `app/static/**`, `app/README.md`, `docs/research/web-app-redesign.md` cho [WEB-01](web-app.md), không tự đổi backend/model/API.

Ngoài ngoại lệ frontend WEB-01, không sửa phần Tùng, backend app/deployment, nguồn/kết quả cũ, checker, README gốc/protocol/Word. Phần chung hỏi qua Markdown, push và chờ người điều phối duyệt. Đọc `origin/main:docs/discussions/answers.md` mỗi đầu phiên và trước phần đang chờ.

## DUC-01 - Audit, làm ngay không chờ Tùng

1. Tái tính từ `results/component_ablation/predictions.csv` và `results/timing_ablation/predictions.csv`; kiểm uniqueness/160 ID mỗi nhóm/confusion matrix/seed summary, đối chiếu bảng cũ. Nộp script, không chỉ chép README.
2. Tách motion A0/A1, A2/A3; attention A0/A2, A1/A3; cost A3/A3_cost_225; timing T0/T1. Delta từng seed/subject, mean/std và FN/FP; không suy nhân quả từ attention heatmap.
3. Viết điều có thể/không thể kết luận: motion có bằng chứng giúp, attention trên motion chưa chắc, cost có trade-off. 06/07 không là cặp chỉ khác một yếu tố.
4. Xem lỗi lặp `S4_FALL_05`, `S3_FALL_06`, các ADL FP để mô tả failure mode; chưa có video thì ghi chưa xác minh. Không chỉnh thiết kế/ngưỡng theo score outer-test của clip đó.

## DUC-02 - Quality-aware temporal pooling, nộp M0

Giả thuyết: pooling xét chất lượng pose giảm ảnh hưởng mốc thiếu/visibility thấp. **Chưa có kết quả, chưa xác nhận tính mới**.

- Giữ local A3/tám kênh v2/loss/multiplier 1,0/augmentation/recipe. Chỉ đổi pooling với prior chất lượng từ mask gốc/visibility, không thêm depth/acceleration/pretrain/threshold heuristic cùng lúc.
- Nộp công thức `q_t`, epsilon, xử lý mask/ánh xạ 32 mốc sang temporal sau stride. Dùng metadata thực có trong NPZ, không suy timestamp/pose liên tục đã mất.
- Hướng cần kiểm tra, không là kết luận: chuẩn hóa lại attention bằng prior dương `epsilon + q_t`. q hữu hạn/có giới hạn, không dựa label/subject/outer prediction; khóa công thức trước train.
- Augmentation mất pose phải cập nhật q từ mask/visibility đã augment, không dùng q sạch để cung cấp thông tin không tương ứng input.
- All-missing cần fallback xác định tránh NaN; test riêng, không quảng bá probability trên input không đủ pose. Không tự thay abstain của app hoặc tính abstain là TN.
- Đọc related work chính chủ về confidence/occlusion-aware skeleton; trích nguồn, không nhận ý tưởng chung là phát minh nhóm. M0 ghi đóng góp đồ án: adaptation + kiểm chứng.

Người điều phối duyệt công thức/config trước full run. Không đủ dữ liệu xác định q tin cậy: báo hạn chế/xin đổi kế hoạch, không bịa confidence.

## DUC-03 - Thực nghiệm và ablation

| Mã | Cấu hình | Mục đích |
|---|---|---|
| Q0 | A3, attention gốc, multiplier 1,0 | Đối chứng chạy lại cùng runner/môi trường |
| Q1 | Như Q0 + quality prior khóa trước | Chỉ xét pooling |

Theo protocol: 24 outer, 96 fits cho hai biến thể nếu không search thêm. Kiểm tắt prior thì Q1 tương đương Q0 (cùng weights/input, tolerance công bố), shape sau stride/all-missing/gradient/metric từ raw predictions. Không cần Q1 thắng để báo đúng.

Đo parameters/latency cùng máy/backend. Robustness mất pose tổng hợp là phụ nếu khóa mức nhiễu/mask/seed, cùng nhiễu Q0/Q1, vẫn báo tập sạch; không gọi test ngoài. Cost 2,25 cho Q1 chỉ thêm sau duyệt, phải có Q0 cùng cost; không trộn loss với pooling.

## DUC-04 - Tổng hợp khi có kết quả Tùng

Người điều phối duyệt PR/commit Tùng rồi đọc benchmark/nối schema chung, ghi commit/run; không lấy số paper dataset khác hay tự merge thay main.

Viết vấn đề/backbone/lý do chọn, đóng góp kiểm chứng, đối chứng nâng cao trên dữ liệu nhóm, ablation, sensitivity/specificity/lỗi từng người, chi phí/giới hạn/kết quả âm. So Q1/A3 với B3 khác pooling, không gọi chỉ đổi backbone. Khác recipe/backend thì nêu rõ.

Main nhận Markdown/bảng/ảnh để cập nhật `docs/bao-cao.docx`; không sửa Word trực tiếp. Chưa có Tùng: DUC-04 `WAITING_FOR_TUNG`, làm phần độc lập trước.

## Tiêu chí hoàn tất

- [ ] Audit tái tạo bảng cũ, giải thích đúng motion/attention/cost/timing.
- [ ] M0 duyệt, công thức q/giới hạn/related work rõ.
- [ ] Code có parity/shape/mask/split/metric checks PASS, hướng dẫn tái lập.
- [ ] Q0/Q1 đủ ba seed x bốn subject, raw OOF/schema/latency, không giấu kết quả âm.
- [ ] Bản thảo có kết quả Tùng duyệt; còn chờ thì không báo hoàn tất toàn nhiệm vụ.
- [ ] Checker PASS, báo cáo/PR có bằng chứng/giới hạn, không thay deployment.
