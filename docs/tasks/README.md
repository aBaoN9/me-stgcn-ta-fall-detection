# Bảng giao việc nghiên cứu

Phân công hiện hành do Gia Bảo xác nhận ngày 30/10/2026, thay phạm vi giao ngày 28/10. Đây là hướng công việc, không áp sẵn lời giải. Trạng thái phương án mới `M0_REQUIRED`; tiến độ/code đã có giữ trong báo cáo cá nhân, không xóa hoặc nhận là đã nghiệm thu. Người điều phối publish bản chung trên main sau kiểm tra; file local chưa đến máy thành viên.

| Người | Nhánh | Phần chính | Trạng thái | Nhiệm vụ | Báo cáo |
|---|---|---|---|---|---|
| Tùng | `dinhtung` | Giữ dataset, dùng mô hình hoặc phương pháp khác; so với ME-STGCN-TA | M0_REQUIRED | [TUNG-01 đến TUNG-04](dinhtung.md) | [Tùng](../reports/dinhtung.md) |
| Đức | `vinhduc` | Giữ ME-STGCN-TA làm mốc, thay đổi có kiểm soát; so với bản gốc | M0_REQUIRED | [DUC-01 đến DUC-04](vinhduc.md) | [Đức](../reports/vinhduc.md) |
| Gia Bảo (người điều phối) | `giabao` | Phát triển kỹ thuật, review, chuẩn bị tích hợp và cập nhật Word | Theo yêu cầu | [Vai trò Gia Bảo](giabao.md) | [Gia Bảo](../reports/giabao.md) |

`main` giữ bản chuẩn đã duyệt và tài liệu điều phối chung. Quyết định theo ID vẫn ở [answers.md](../discussions/answers.md) trên main, không chuyển sang giabao. Nhánh giabao không tự chứa code mới của Tùng/Đức và không thay thế main.

## Mốc bàn giao

1. **M0 - Chốt trước full train:** xác nhận tên/nhánh/input; nộp thiết kế/protocol, khác biệt so với code/paper. Người điều phối duyệt trước full run.
2. **M1 - Có thể chạy lại:** code vùng riêng, lệnh chạy, check shape/split/mask/metric, smoke. Smoke không là kết quả học thuật.
3. **M2 - Bằng chứng:** full nested LOSO ba seed, raw OOF, config/hash/log, tổng hợp/lỗi. Thiếu GPU/input thì `BLOCKED`, không bịa bảng.
4. **M3 - Review:** PR kèm báo cáo/tiêu chí xong; người điều phối quyết định merge. Không tự nâng model web theo bảng test.

Hai bạn tự đề xuất cách làm và giải thích thay đổi; không bắt buộc CTR-GCN/B0-B3 hoặc quality-aware/Q0-Q1. Giữ dataset, nhãn, split, seed và nguyên tắc đánh giá chung; cấu hình cụ thể chốt ở M0. Không đặt deadline hoặc yêu cầu kết quả phải cải thiện.

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

Hai hướng làm độc lập theo [protocol/schema chung](../research-protocol.md). Đức không cần đợi Tùng để kiểm chứng cặp gốc/thay đổi của mình. Gia Bảo thống nhất mốc ME-STGCN-TA, đánh giá và ngân sách trước full; tổng hợp liên nhánh chỉ dùng kết quả/commit đã review, không tự merge hoặc điền số còn thiếu. Phản hồi cũ về lỗi phần mềm còn áp dụng nếu mã đó được tái sử dụng; xem quyết định GB-20261030-001 trong answers.
