# Bàn giao dự án cho nhóm

**Phạm vi 21/11/2026 (`GB-20261121-002`):** mã PoseLab đã gỡ khỏi repo; demo mới để sau, không có việc UI/QA hiện hành. Gia Bảo đã lập [kế hoạch tuning A3 so TCN/GRU](research/a3-tuning-plan.md), `PLANNED / NOT_RUN`, chưa train. Không đổi bằng chứng đã nhận, nhiệm vụ/hạn Đức hoặc model triển khai.

**Nghiệm thu bổ sung 21/11/2026:** GRU/CTR-GCN adapted trong PR #3 đã được nhận theo `GB-20261121-001`, cùng bản sửa monitor khi chưa có `.runtime`. [Bộ bổ sung](../results/backbone_benchmark/tung-gru-ctr-rtx3050-20261120-233932/README.md) có 96 fits, 24 checkpoint full mới; [README](../README.md) so sánh đủ bốn model và phân biệt hai lần đo latency. 16 tests local/repo checker PASS; không train lại, đổi deployment, bật PoseLab, sửa Word hoặc nghiệm thu thay bài Đức.

**Cập nhật nghiệm thu 20/11/2026:** benchmark TCN–A3 của Tùng đã `ACCEPTED_BENCHMARK`, PR #2 merge vào main tại `bbe3c64`. [README](../README.md) và [bộ bàn giao](../results/backbone_benchmark/tung-m2-rtx3050-20261120-165523/README.md) có code, 24 checkpoint, dự đoán và hướng dẫn. Quyết định `GB-20261120-001` trong [answers](discussions/answers.md) là trạng thái điều phối mới hơn snapshot cá nhân; chưa thay model triển khai hoặc nghiệm thu bài Đức.

**Hỗ trợ chạy 19/11/2026:** [bắt đầu train](training-start.md) có tool input, local/Kaggle và xử lý notebook pin code cũ. Theo `GB-20261119-002`, cả hai được tự full CPU nếu thiếu GPU, khóa addendum/config/code trước outer-test và giữ điều kiện so sánh; không chờ duyệt.

**Quy định hiện hành 18/11/2026 — `GB-20261118-001` (Tùng), `GB-20261116-002` (Đức):** Cả hai tự chọn phương án trong nhiệm vụ, triển khai, kiểm tra, smoke và full train khi đủ điều kiện; không chờ Gia Bảo duyệt M0, mô hình/kiến trúc, cấu hình, thiết bị hoặc review lại bản sửa trước chạy. Tùng được bắt đầu TCN so với A3 từ M0 đã nộp, tự điều chỉnh hoặc dùng dự phòng trước outer-test có ghi lý do. M0 phải được ghi/khóa trước outer-test; dữ liệu, split, seed và đánh giá chung không đổi; test và smoke thật phải PASS. Được commit/push sản phẩm đúng phạm vi lên nhánh cá nhân, upload artifact và mở PR, không xin lại từng lần. Review cuối vẫn quyết định nghiệm thu/merge main. Xem [quyết định chính thức](discussions/answers.md).

Ngày lập: 28/10/2026. Cập nhật phân công: 30/10/2026 theo Gia Bảo. Gia Bảo là người điều phối, phát triển kỹ thuật trên `giabao`; Tùng làm `dinhtung`, Đức làm `vinhduc`. `main` giữ bản chuẩn đã duyệt và quyết định chung. Đây là điểm bắt đầu cho cả thành viên và AI.

Gia Bảo và AI đọc [vai trò riêng](tasks/giabao.md): không dùng main để thử code, không chuyển kênh trả lời nhóm khỏi main. Việc review nhánh khác không đòi hỏi merge; chỉ tích hợp code sau kiểm tra và khi chủ repo yêu cầu.

## Đã có gì, không cần làm lại?

| Phần | Hiện trạng |
|---|---|
| Dữ liệu | 160 video, 79 Fall/81 ADL, 4 người; MediaPipe 33 khớp, 32 mốc. Bốn tensor/CSV chuẩn đã có trong `data/gmdcsa24/` từ 19/11/2026, đúng hash 07; video gốc không trong repo. Xem [data](../data/README.md). |
| Demo cũ | PoseLab dừng 30/10, mã `app/` gỡ 21/11/2026; WEB-01/WEB-02 hủy. ONNX/config/model và lịch sử Git giữ nguyên; demo mới `DEFERRED`. |
| Model triển khai | Notebook 06: corrected angle, nested LOSO, final all-subject train. Không tự thay bằng ứng viên mới. |
| Baseline/ablation | Notebook 07 và `training/component_ablation/`: SVM v2, A0/A1/A2/A3, cost 2,25; seed 42/142/242. [Phân tích](component-ablation.md) đã có cả ưu điểm và lỗi. |
| Timing | Notebook 08: T0/T1 đã chạy; T1 chưa cải thiện ổn định. Không giao lại như chưa làm. |
| Benchmark Tùng | TCN–A3 đã nghiệm thu M0–M3 qua PR #2 ngày 20/11: 96 fits, 24 checkpoint, TCN Accuracy 91,04% so với A3 chạy lại 89,38%; xem [bộ bàn giao](../results/backbone_benchmark/tung-m2-rtx3050-20261120-165523/README.md). Không thay deployment. |
| Báo cáo | `docs/bao-cao.docx` là bản Word hiện tại, người điều phối cập nhật sau review; không tạo nhiều bản Word song song. |

## Khoảng trống học thuật

Tập trung nghiên cứu, không làm PoseLab. Cần trả lời: **trên cùng dataset và cách đánh giá, mô hình/phương pháp khác đạt gì; thay đổi có kiểm soát mô hình hiện tại giúp gì và đánh đổi gì?**

- Tùng: giữ dataset, tự đề xuất mô hình hoặc phương pháp khác để xử lý bài toán, so với ME-STGCN-TA theo giao thức chung. Không bắt buộc CTR-GCN hoặc chỉ được đổi backbone.
- Đức: giữ mô hình ME-STGCN-TA làm mốc, tự đề xuất thay đổi có kiểm soát và so với bản gốc. Không bắt buộc quality-aware hoặc công thức cũ; không đổi hẳn mô hình như phần Tùng.
- Mốc chính xác của Đức là `A3_motion_attention` / `fixed` trong notebook 07 và `training/component_ablation/`, không phải checkpoint final 06 hoặc A3_cost_225; xem [nguồn/cấu hình mốc](tasks/vinhduc.md). Chạy lại bản gốc cùng bản thay đổi, không lấy số cũ coi là run đối chứng mới.
- Hai bạn giải thích phần giữ nguyên/thay đổi và lý do, ghi M0 trước full; cả hai tự khóa và kiểm test/smoke thật rồi chạy theo `GB-20261118-001` (Tùng) và `GB-20261116-002` (Đức). Không giao sẵn lời giải. Giữ bài làm và lỗi đã review nếu dùng tiếp mã; cả hai không chờ review lại bản sửa trước chạy.
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

- Bốn input chuẩn đã được cấp tại `data/gmdcsa24/` theo `GB-20261119-001`; nhận main an toàn rồi dùng `--input-root data/gmdcsa24`, xem [data/README](../data/README.md). Không chờ xin gửi file riêng hoặc chạy lại 01. Nếu vẫn thiếu/sai hash trên máy mình, báo đường dẫn/lệnh/lỗi cụ thể; không cần chạy PoseLab để nghiên cứu.
- Tạo môi trường Python riêng; xem [train](../training/component_ablation/README.md) và [Kaggle](notebook-guide.md). Không chạy lại 01 nếu đã có bốn input đúng hash.
- `python scripts/check_repository.py`: kiểm toàn vẹn repo/bằng chứng cũ, không chứng minh code mới đúng.
- `python -m training.component_ablation.validate_pipeline --input-root data/gmdcsa24`: cần dependencies; không phải full train. `data/local/` vẫn dành cho dữ liệu cục bộ bị ignore, không cần copy input chuẩn sang đó.
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

Người điều phối trả lời theo ID tại `docs/discussions/answers.md` trên main. Thành viên fetch/đọc `origin/main:docs/discussions/answers.md` đầu mỗi phiên và trước phần chờ duyệt; chưa trả lời thì tiếp tục việc độc lập, không tự xem là chấp thuận ngoài phạm vi. Cả hai không chờ trả lời cho phần đã được cấp quyền theo `GB-20261118-001` (Tùng) và `GB-20261116-002` (Đức). Ghi đè là commit mới cập nhật cùng file, không xóa lịch sử hoặc force-push. Chủ repo kiểm thủ công, không có monitor tự động.

## Checklist trước khi mời clone

- Người điều phối review, commit và push bộ bàn giao lên main khi đồng ý publish.
- Tạo/publish các nhánh giabao/dinhtung/vinhduc từ main đã có hướng dẫn, hoặc merge main vào nhánh tạo sớm khi an toàn; không ghi đè nhánh đã có bài làm.
- Hỗ trợ nguồn bốn input/kiểm quyền khi thành viên còn thiếu. Cả hai tự chọn GPU local/Kaggle có quyền trong hạn mức miễn phí/đã có, tự khóa cấu hình trong phạm vi đã cấp, không chờ duyệt thiết bị hoặc cấu hình.
- Bật bảo vệ main/yêu cầu PR review nếu có quyền. `AGENTS.md` không thay branch protection; lượt viết tài liệu không cấu hình bảo vệ remote.
- Bộ bàn giao mới trong working tree local thì chưa đến máy thành viên.
