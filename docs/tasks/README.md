# Bảng giao việc nghiên cứu

**Sẵn sàng vận hành 19/11/2026:** [hướng dẫn bắt đầu](../training-start.md), đủ input chuẩn và tool kiểm/tải; `GB-20261119-002` cho cả hai tự full CPU nếu không có GPU với addendum/lock trước outer-test, không chờ duyệt lại.

**Thông báo cập nhật 18/11/2026:** Gia Bảo yêu cầu Tùng và Đức đọc `GB-20261116-001` trong [answers](../discussions/answers.md), xác nhận đã nhận và triển khai phần việc được phép ngay phiên tiếp theo. Cả hai được tự thực hiện theo `GB-20261118-001` (Tùng) và `GB-20261116-002` (Đức), không chờ duyệt M0/chọn phương án/kiến trúc/review lại trước full.

**Hạn nộp sản phẩm: thứ Tư, 25/11/2026 (UTC+7), cho cả hai bạn. Sản phẩm gồm MODEL ĐÃ TRAIN + KẾT QUẢ ĐÁNH GIÁ, đưa lên GitHub**, kèm mã/lệnh tái lập và báo cáo. Không chỉ nộp M0 hoặc tiến độ. Tùng nộp mô hình/phương pháp khác; Đức nộp cặp mốc gốc/thay đổi có kiểm soát. Checklist artifact trong `GB-20261116-001` ở answers. Thiếu input/quyền/GPU phải báo sớm và nộp phần đã làm đúng hạn; chưa có đủ model/kết quả thật là chưa hoàn tất, cả hai tự chạy full sau khi tự kiểm đủ điều kiện, không chờ duyệt theo hai quyết định tự chủ; không ai được bịa kết quả.

Phân công hiện hành do Gia Bảo xác nhận ngày 30/10/2026, thay phạm vi giao ngày 28/10. Đây là hướng công việc, không áp sẵn lời giải. Tùng và Đức có quyền `AUTONOMOUS`, phải tự ghi/khóa M0 (`M0_DOCUMENTED` khi thực hiện), không phải đã hoàn tất; tiến độ/code đã có giữ trong báo cáo cá nhân, không xóa hoặc nhận là đã nghiệm thu. Người điều phối publish bản chung trên main sau kiểm tra; file local chưa đến máy thành viên.

| Người | Nhánh | Phần chính | Trạng thái | Nhiệm vụ | Báo cáo |
|---|---|---|---|---|---|
| Tùng | `dinhtung` | Giữ dataset, dùng mô hình hoặc phương pháp khác; so với ME-STGCN-TA | AUTONOMOUS (quyền thực hiện, chưa DONE) | [TUNG-01 đến TUNG-04](dinhtung.md) | [Tùng](../reports/dinhtung.md) |
| Đức | `vinhduc` | Giữ ME-STGCN-TA làm mốc, thay đổi có kiểm soát; so với bản gốc | AUTONOMOUS (quyền thực hiện, chưa DONE) | [DUC-01 đến DUC-04](vinhduc.md) | [Đức](../reports/vinhduc.md) |
| Gia Bảo (người điều phối) | `giabao` | Phát triển kỹ thuật, review, chuẩn bị tích hợp và cập nhật Word | Theo yêu cầu | [Vai trò Gia Bảo](giabao.md) | [Gia Bảo](../reports/giabao.md) |

`main` giữ bản chuẩn đã duyệt và tài liệu điều phối chung. Quyết định theo ID vẫn ở [answers.md](../discussions/answers.md) trên main, không chuyển sang giabao. Nhánh giabao không tự chứa code mới của Tùng/Đức và không thay thế main.

## Mốc bàn giao

1. **M0 - Chốt trước full train:** xác nhận tên/nhánh/input; nộp thiết kế/protocol, khác biệt so với code/paper. Cả hai tự ghi/khóa M0, kiểm test/smoke thật rồi full, không chờ duyệt; M0 là đăng ký phương án, không là đơn xin phép.
2. **M1 - Có thể chạy lại:** code vùng riêng, lệnh chạy, check shape/split/mask/metric, smoke. Smoke không là kết quả học thuật.
3. **M2 - Bằng chứng:** model/checkpoint đã train và kết quả full nested LOSO ba seed, raw OOF, config/hash/log, tổng hợp/lỗi; upload GitHub theo checklist trong answers. Không có GPU thì được tự dùng CPU theo `GB-20261119-002`; chỉ ghi `BLOCKED` khi thiếu input/quyền hoặc không có tài nguyên phù hợp, không bịa bảng hoặc đánh dấu sản phẩm hoàn tất.
4. **M3 - Review:** PR kèm báo cáo/tiêu chí xong; người điều phối quyết định merge. Không tự nâng model web theo bảng test.

Hai bạn tự đề xuất cách làm và giải thích thay đổi; không bắt buộc CTR-GCN/B0-B3 hoặc quality-aware/Q0-Q1. Giữ dataset, nhãn, split, seed và nguyên tắc đánh giá chung; cấu hình cụ thể chốt ở M0. Hạn nộp sản phẩm model và kết quả là 25/11/2026 theo cập nhật của Gia Bảo ngày 16/11/2026; cả hai tự ghi/khóa M0 và tự kiểm đủ điều kiện, không chờ duyệt trước full, không yêu cầu kết quả phải cải thiện.

**Mốc Đức không còn mơ hồ:** `A3_motion_attention` / `fixed` của notebook 07, tám kênh, temporal attention, Fall multiplier 1.0; nguồn/recipe và yêu cầu chạy lại cặp đối chứng trong [nhiệm vụ Đức](vinhduc.md). Không dùng checkpoint demo 06, A3_cost_225 hoặc T0/T1 thay mốc.

## PoseLab đã dừng và trao đổi nhóm

- [WEB-01/WEB-02](web-app.md) đã `CANCELLED` ngày 30/10/2026. Không còn nhiệm vụ UI/QA web, kể cả khi chờ GPU hoặc còn thời gian; không tự bật lại PoseLab.
- Mọi câu hỏi/đề nghị với nhóm gửi qua [file Markdown cá nhân](../discussions/README.md), cập nhật cùng file và commit/push mỗi lần hỏi. Người điều phối trả lời trong [answers.md](../discussions/answers.md) trên main; không xem im lặng là duyệt.

## Ranh giới tránh đụng nhau

- Gia Bảo: sửa code/thử nghiệm và chuẩn bị tích hợp trên giabao theo yêu cầu cụ thể; cập nhật báo cáo riêng. Không tự lấy việc đang giao Tùng/Đức hoặc push nhánh của hai bạn. Tài liệu quy tắc/bàn giao/quyết định chung do Gia Bảo giữ trên main.
- Tùng: `training/backbone_benchmark/`, `results/backbone_benchmark/`, `docs/research/backbone-benchmark.md`, báo cáo Tùng.
- Đức: `training/contribution_study/`, `results/contribution_study/`, `docs/research/contribution-study.md`, báo cáo Đức.
- Đọc/import `training/component_ablation/`, không sửa nguồn/kết quả cũ. Notebook mới nếu cần nằm trong training riêng, không vào `notebooks/` đang khóa inventory.
- Ngoại lệ cho Đức sửa frontend và Tùng QA web đã thu hồi. Giữ nguồn/model/bằng chứng cũ, không tự xóa bài làm hoặc dữ liệu.
- Mỗi người được ghi file `docs/discussions/` đúng tên mình. `answers.md` chỉ người điều phối sửa trên main.
- AGENTS, README gốc, protocol chung, backend app, model/Word do người điều phối giữ. Sửa ngoài phạm vi: gửi câu hỏi Markdown có file/lý do, push và chờ duyệt.

## Phối hợp

Hai hướng làm độc lập theo [protocol/schema chung](../research-protocol.md). Đức không cần đợi Tùng để kiểm chứng cặp gốc/thay đổi của mình. Gia Bảo đã chốt mốc/giao thức; cả hai tự ghi cấu hình/ngân sách trong phạm vi được cấp, không chờ duyệt trước full; tổng hợp liên nhánh chỉ dùng kết quả/commit đã review, không tự merge hoặc điền số còn thiếu. Phản hồi cũ về lỗi phần mềm còn áp dụng nếu mã đó được tái sử dụng; xem quyết định GB-20261030-001 trong answers.
