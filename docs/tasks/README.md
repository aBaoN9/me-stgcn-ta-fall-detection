# Bảng giao việc nghiên cứu

Ngày giao: 28/10/2026. Trạng thái ban đầu `TODO`, **chưa có thí nghiệm mới nào chạy**. Người điều phối cập nhật bảng này trên main sau review; thành viên cập nhật báo cáo riêng.

| Người | Nhánh | Phần chính | Trạng thái | Nhiệm vụ | Báo cáo |
|---|---|---|---|---|---|
| Tùng | `dinhtung` | Benchmark backbone skeleton trên GMDCSA-24 | TODO | [TUNG-01 đến TUNG-04](dinhtung.md) | [Tùng](../reports/dinhtung.md) |
| Đức | `vinhduc` | Kiểm chứng đóng góp, quality-aware pooling, trade-off | TODO | [DUC-01 đến DUC-04](vinhduc.md) | [Đức](../reports/vinhduc.md) |
| Người điều phối | `main` | Duyệt protocol, cấp input, review PR, tích hợp và cập nhật Word | Theo mốc | [Checklist](../handoff.md) | Đọc hai PR/báo cáo |

## Mốc bàn giao

1. **M0 - Chốt trước full train:** xác nhận tên/nhánh/input; nộp thiết kế/protocol, khác biệt so với code/paper. Người điều phối duyệt trước full run.
2. **M1 - Có thể chạy lại:** code vùng riêng, lệnh chạy, check shape/split/mask/metric, smoke. Smoke không là kết quả học thuật.
3. **M2 - Bằng chứng:** full nested LOSO ba seed, raw OOF, config/hash/log, tổng hợp/lỗi. Thiếu GPU/input thì `BLOCKED`, không bịa bảng.
4. **M3 - Review:** PR kèm báo cáo/tiêu chí xong; người điều phối quyết định merge. Không tự nâng model web theo bảng test.

Không đặt deadline giả; nhóm thống nhất theo lịch/tài nguyên. Không mở hướng tùy tiện để thay tiêu chí bắt buộc.

## Ranh giới tránh đụng nhau

- Tùng: `training/backbone_benchmark/`, `results/backbone_benchmark/`, `docs/research/backbone-benchmark.md`, báo cáo Tùng.
- Đức: `training/contribution_study/`, `results/contribution_study/`, `docs/research/contribution-study.md`, báo cáo Đức.
- Đọc/import `training/component_ablation/`, không sửa nguồn/kết quả cũ. Notebook mới nếu cần nằm trong training riêng, không vào `notebooks/` đang khóa inventory.
- AGENTS, README gốc, protocol chung, app, model/Word do người điều phối giữ. Sửa ngoài phạm vi: ghi file/lý do trong báo cáo, chờ duyệt.

## Phối hợp

Đức làm DUC-01 đến DUC-03 độc lập với Tùng. Tùng xuất [schema chung](../research-protocol.md). Khi PR Tùng có kết quả, Đức dùng bản/commit được duyệt để làm DUC-04, ghi SHA và không tự merge thay main. Chưa có benchmark: DUC-04 ghi `WAITING_FOR_TUNG`, không tự điền số.
