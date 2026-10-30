# Bàn giao dự án cho nhóm

Ngày lập: 28/10/2026. Cập nhật phân công: 30/10/2026 theo Gia Bảo. Gia Bảo là người điều phối, phát triển kỹ thuật trên `giabao`; Tùng làm `dinhtung`, Đức làm `vinhduc`. `main` giữ bản chuẩn đã duyệt và quyết định chung. Đây là điểm bắt đầu cho cả thành viên và AI.

Gia Bảo và AI đọc [vai trò riêng](tasks/giabao.md): không dùng main để thử code, không chuyển kênh trả lời nhóm khỏi main. Việc review nhánh khác không đòi hỏi merge; chỉ tích hợp code sau kiểm tra và khi chủ repo yêu cầu.

## Đã có gì, không cần làm lại?

| Phần | Hiện trạng |
|---|---|
| Dữ liệu | 160 video, 79 Fall/81 ADL, 4 người; MediaPipe 33 khớp, 32 mốc. Video/tensor không trong repo; xem [data](../data/README.md). |
| Demo cũ | PoseLab đã dừng ngày 30/10/2026; launcher/ASGI bị chặn, WEB-01/WEB-02 hủy. Nguồn `app/` và ONNX/config giữ để truy vết, không phải nhiệm vụ phát triển web. |
| Model triển khai | Notebook 06: corrected angle, nested LOSO, final all-subject train. Không tự thay bằng ứng viên mới. |
| Baseline/ablation | Notebook 07 và `training/component_ablation/`: SVM v2, A0/A1/A2/A3, cost 2,25; seed 42/142/242. [Phân tích](component-ablation.md) đã có cả ưu điểm và lỗi. |
| Timing | Notebook 08: T0/T1 đã chạy; T1 chưa cải thiện ổn định. Không giao lại như chưa làm. |
| Báo cáo | `docs/bao-cao.docx` là bản Word hiện tại, người điều phối cập nhật sau review; không tạo nhiều bản Word song song. |

## Khoảng trống học thuật

Tập trung nghiên cứu, không làm PoseLab. Cần trả lời: **trên cùng dataset và cách đánh giá, mô hình/phương pháp khác đạt gì; thay đổi có kiểm soát mô hình hiện tại giúp gì và đánh đổi gì?**

- Tùng: giữ dataset, tự đề xuất mô hình hoặc phương pháp khác để xử lý bài toán, so với ME-STGCN-TA theo giao thức chung. Không bắt buộc CTR-GCN hoặc chỉ được đổi backbone.
- Đức: giữ mô hình ME-STGCN-TA làm mốc, tự đề xuất thay đổi có kiểm soát và so với bản gốc. Không bắt buộc quality-aware hoặc công thức cũ; không đổi hẳn mô hình như phần Tùng.
- Mốc chính xác của Đức là `A3_motion_attention` / `fixed` trong notebook 07 và `training/component_ablation/`, không phải checkpoint final 06 hoặc A3_cost_225; xem [nguồn/cấu hình mốc](tasks/vinhduc.md). Chạy lại bản gốc cùng bản thay đổi, không lấy số cũ coi là run đối chứng mới.
- Hai bạn giải thích phần giữ nguyên/thay đổi và lý do, nộp M0 trước full; không được giao sẵn lời giải. Giữ bài làm và lỗi đã review nếu tiếp tục dùng mã, xem GB-20261030-001 trong answers.
- Không tốt hơn vẫn phải báo trung thực. Đóng góp đồ án cần bằng chứng, không mặc định là phát minh mới của ngành.

Đọc [bảng việc](tasks/README.md), nhiệm vụ cá nhân và [giao thức](research-protocol.md).

## Khởi động sau khi clone

```bash
git clone https://github.com/aBaoN9/me-stgcn-ta-fall-detection.git
cd me-stgcn-ta-fall-detection
git status --short
git fetch origin
git branch -a
```

Yêu cầu AI đọc `AGENTS.md`; chưa biết tên thì phải hỏi trước khi sửa. Sau khi xác định Tùng, dùng **một** trường hợp:

```bash
# Da co nhanh local:
git switch dinhtung
# Chua co local nhung da co origin/dinhtung:
git switch --track origin/dinhtung
```

Đức thay `dinhtung` bằng `vinhduc`; Gia Bảo thay bằng `giabao` và đọc nhiệm vụ điều phối tương ứng. Nếu chưa có cả local lẫn remote, chỉ tạo nhánh được giao sau khi xác nhận `origin/main` đã chứa bộ bàn giao:

```bash
git switch -c dinhtung origin/main
```

Không chạy cả ba lệnh switch máy móc. Không mang thay đổi chưa commit từ nhánh khác sang. Nhánh tạo trước bộ hướng dẫn: trong worktree sạch dùng `git merge origin/main` để nhận tài liệu sau khi người điều phối publish; không reset nhánh có bài làm.

## Dữ liệu, môi trường và kiểm tra

- Train cần dữ liệu được cấp quyền ở [data/README](../data/README.md); hỏi nguồn input phù hợp phương án M0, không giả định máy clone đã có. Không cần chạy PoseLab để nghiên cứu.
- Tạo môi trường Python riêng; xem [train](../training/component_ablation/README.md) và [Kaggle](notebook-guide.md). Không chạy lại 01 nếu đã có bốn input đúng hash.
- `python scripts/check_repository.py`: kiểm toàn vẹn repo/bằng chứng cũ, không chứng minh code mới đúng.
- `python -m training.component_ablation.validate_pipeline --input-root data/local`: cần input/dependencies; không phải full train.
- Lệnh code mới do tác giả bổ sung vào README vùng nhiệm vụ; đối chiếu commit/nhánh để biết đã có gì, không đoán lệnh hoặc coi tài liệu là code đã chạy.

## Main đọc báo cáo bằng cách nào?

Thành viên cập nhật báo cáo riêng, commit/push code/báo cáo khi người dùng yêu cầu, mở PR về main. Ngoại lệ đã được chủ repo cho phép sẵn: **mỗi lần hỏi nhóm phải cập nhật file trao đổi cá nhân và commit/push**, xem [quy trình Markdown](discussions/README.md). Người điều phối đọc qua PR hoặc fetch, không cần merge:

```bash
git fetch origin
git show origin/dinhtung:docs/reports/dinhtung.md
git show origin/vinhduc:docs/reports/vinhduc.md
git show origin/dinhtung:docs/discussions/dinhtung.md
git show origin/vinhduc:docs/discussions/vinhduc.md
git diff origin/main...origin/dinhtung
git diff origin/main...origin/vinhduc
```

Chỉ dùng khi remote đã push. Không chép báo cáo về main rồi coi code đã tích hợp. Bảng việc main là snapshot do người điều phối cập nhật sau review, không đồng bộ tự động.

Người điều phối trả lời theo ID tại `docs/discussions/answers.md` trên main. Thành viên fetch/đọc `origin/main:docs/discussions/answers.md` đầu mỗi phiên và trước phần chờ duyệt; chưa trả lời thì tiếp tục việc độc lập, không tự xem là chấp thuận. Ghi đè là commit mới cập nhật cùng file, không xóa lịch sử hoặc force-push. Chủ repo kiểm thủ công, không có monitor tự động.

## Checklist trước khi mời clone

- Người điều phối review, commit và push bộ bàn giao lên main khi đồng ý publish.
- Tạo/publish các nhánh giabao/dinhtung/vinhduc từ main đã có hướng dẫn, hoặc merge main vào nhánh tạo sớm khi an toàn; không ghi đè nhánh đã có bài làm.
- Cấp nguồn bốn input/kiểm quyền; xác nhận GPU/Kaggle và duyệt cấu hình đầu tiên.
- Bật bảo vệ main/yêu cầu PR review nếu có quyền. `AGENTS.md` không thay branch protection; lượt viết tài liệu không cấu hình bảo vệ remote.
- Bộ bàn giao mới trong working tree local thì chưa đến máy thành viên.
