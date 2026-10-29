# Bảng giao việc nghiên cứu

Ngày giao: 28/10/2026. Trạng thái ban đầu `TODO`, **chưa có thí nghiệm mới nào chạy**. Người điều phối cập nhật bảng này trên main sau review; thành viên cập nhật báo cáo riêng.

| Người | Nhánh | Phần chính | Trạng thái | Nhiệm vụ | Báo cáo |
|---|---|---|---|---|---|
| Tùng | `dinhtung` | Benchmark backbone skeleton trên GMDCSA-24 | TODO | [TUNG-01 đến TUNG-04](dinhtung.md) | [Tùng](../reports/dinhtung.md) |
| Đức | `vinhduc` | Kiểm chứng đóng góp, quality-aware pooling, trade-off | TODO | [DUC-01 đến DUC-04](vinhduc.md) | [Đức](../reports/vinhduc.md) |
| Gia Bảo (người điều phối) | `giabao` | Phát triển kỹ thuật, review, chuẩn bị tích hợp và cập nhật Word | Theo yêu cầu | [Vai trò Gia Bảo](giabao.md) | [Gia Bảo](../reports/giabao.md) |

`main` giữ bản chuẩn đã duyệt và tài liệu điều phối chung. Quyết định theo ID vẫn ở [answers.md](../discussions/answers.md) trên main, không chuyển sang giabao. Nhánh giabao không tự chứa code mới của Tùng/Đức và không thay thế main.

## Mốc bàn giao

1. **M0 - Chốt trước full train:** xác nhận tên/nhánh/input; nộp thiết kế/protocol, khác biệt so với code/paper. Người điều phối duyệt trước full run.
2. **M1 - Có thể chạy lại:** code vùng riêng, lệnh chạy, check shape/split/mask/metric, smoke. Smoke không là kết quả học thuật.
3. **M2 - Bằng chứng:** full nested LOSO ba seed, raw OOF, config/hash/log, tổng hợp/lỗi. Thiếu GPU/input thì `BLOCKED`, không bịa bảng.
4. **M3 - Review:** PR kèm báo cáo/tiêu chí xong; người điều phối quyết định merge. Không tự nâng model web theo bảng test.

Không đặt deadline giả; nhóm thống nhất theo lịch/tài nguyên. Không mở hướng tùy tiện để thay tiêu chí bắt buộc.

## Web nếu còn thời gian và trao đổi nhóm

- [WEB-01/WEB-02](web-app.md) là phần phụ `OPTIONAL_NOT_STARTED`: Đức làm lại UI trên backend hiện tại; Tùng review/QA độc lập. Không thay việc nghiên cứu hoặc tự đổi model.
- Mọi câu hỏi/đề nghị với nhóm gửi qua [file Markdown cá nhân](../discussions/README.md), cập nhật cùng file và commit/push mỗi lần hỏi. Người điều phối trả lời trong [answers.md](../discussions/answers.md) trên main; không xem im lặng là duyệt.

## Ranh giới tránh đụng nhau

- Gia Bảo: sửa code/thử nghiệm và chuẩn bị tích hợp trên giabao theo yêu cầu cụ thể; cập nhật báo cáo riêng. Không tự lấy việc đang giao Tùng/Đức hoặc push nhánh của hai bạn. Tài liệu quy tắc/bàn giao/quyết định chung do Gia Bảo giữ trên main.
- Tùng: `training/backbone_benchmark/`, `results/backbone_benchmark/`, `docs/research/backbone-benchmark.md`, báo cáo Tùng.
- Đức: `training/contribution_study/`, `results/contribution_study/`, `docs/research/contribution-study.md`, báo cáo Đức.
- Đọc/import `training/component_ablation/`, không sửa nguồn/kết quả cũ. Notebook mới nếu cần nằm trong training riêng, không vào `notebooks/` đang khóa inventory.
- Ngoại lệ web có thời gian: Đức được ghi `app/static/**`, `app/README.md`, `docs/research/web-app-redesign.md`; Tùng ghi `docs/research/web-app-qa.md`, không cùng sửa UI.
- Mỗi người được ghi file `docs/discussions/` đúng tên mình. `answers.md` chỉ người điều phối sửa trên main.
- AGENTS, README gốc, protocol chung, backend app, model/Word do người điều phối giữ. Sửa ngoài phạm vi: gửi câu hỏi Markdown có file/lý do, push và chờ duyệt.

## Phối hợp

Đức làm DUC-01 đến DUC-03 độc lập với Tùng. Tùng xuất [schema chung](../research-protocol.md). Khi PR Tùng có kết quả, Đức dùng bản/commit được duyệt để làm DUC-04, ghi SHA và không tự merge thay main. Chưa có benchmark: DUC-04 ghi `WAITING_FOR_TUNG`, không tự điền số.
