# 🛡️ ME-STGCN-TA — Phát hiện té ngã từ khung xương

Dự án nghiên cứu phân loại **té ngã (Fall)** và **hoạt động thường ngày (ADL)** từ video một người. MediaPipe trích xuất tư thế; ME-STGCN-TA học quan hệ không gian–thời gian từ khung xương và đặc trưng chuyển động. **PoseLab đã dừng theo quyết định của Gia Bảo ngày 30/10/2026**; repo tập trung so sánh phương pháp và kiểm chứng thay đổi có kiểm soát.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-training-EE4C2C?logo=pytorch&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-notebooks-20BEFF?logo=kaggle&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX-inference-005CED?logo=onnx&logoColor=white)

[Báo cáo Word](docs/bao-cao.docx) · [Phương pháp](docs/methodology.md) · [Notebooks](docs/notebook-guide.md) · [Phân công nghiên cứu](docs/tasks/README.md) · [Hướng nghiên cứu tiếp](docs/training-roadmap.md)

## Thành viên và AI bắt đầu ở đây

Đọc [AGENTS.md](AGENTS.md) -> [bàn giao](docs/handoff.md) -> [nhiệm vụ](docs/tasks/README.md) -> [giao thức nghiên cứu](docs/research-protocol.md) trước khi sửa. **AI chưa biết người đang làm phải hỏi tên**, không đoán từ tài khoản Git.

| Thành viên | Nhánh | Công việc học thuật tiếp theo | Báo cáo |
|---|---|---|---|
| Tùng | `dinhtung` | [Giữ dataset, dùng mô hình hoặc phương pháp khác và so với ME-STGCN-TA](docs/tasks/dinhtung.md) | [Tiến độ Tùng](docs/reports/dinhtung.md) |
| Đức | `vinhduc` | [Thay đổi có kiểm soát ME-STGCN-TA hiện tại và so với bản gốc](docs/tasks/vinhduc.md) | [Tiến độ Đức](docs/reports/vinhduc.md) |
| Gia Bảo (người điều phối) | `giabao` | [Phát triển kỹ thuật, chuẩn bị tích hợp và điều phối](docs/tasks/giabao.md) | [Tiến độ Gia Bảo](docs/reports/giabao.md) |

`main` là bản chuẩn đã duyệt của nhóm, không phải nhánh thử nghiệm của Gia Bảo. Gia Bảo sửa code trên `giabao`, review/kiểm tra rồi mới tích hợp khi được yêu cầu; quy tắc và quyết định chung vẫn publish trên `main` để Tùng/Đức cùng đọc.

Phân công ngày **30/10/2026** thay việc chỉ định sẵn CTR-GCN/quality-aware: hai bạn tự đề xuất phương án, giải thích giữ nguyên/thay đổi gì và vì sao, thống nhất M0 trước full. Giữ cùng dataset và giao thức đánh giá, không coi là cải tiến đã chứng minh hoặc giao lại 07/08 như chưa làm. Giữ bài làm/review đã có; tiến độ xem báo cáo cá nhân. Không push trực tiếp main; xem [quy trình Git và input](docs/handoff.md).

**Mốc của Đức:** `A3_motion_attention` / policy `fixed` của notebook 07, tám kênh, attention bật, Fall multiplier 1.0; không phải model demo 06 hoặc A3_cost_225. [Nhiệm vụ Đức](docs/tasks/vinhduc.md) chỉ rõ mã, recipe và cách đối chiếu bản gốc.

**PoseLab và WEB-01/WEB-02 đã hủy**, không còn việc UI/QA khi rảnh hoặc chờ GPU. **Hỏi nhóm qua Markdown:** cập nhật cùng [file trao đổi cá nhân](docs/discussions/README.md), commit/push lên nhánh mình mỗi lần hỏi; người điều phối kiểm và trả lời trong [answers.md](docs/discussions/answers.md) trên main. Ghi đè nội dung file bằng commit mới, không force-push.

Câu khởi động có thể gửi AI sau clone:

> Đọc AGENTS.md, docs/handoff.md và quyết định GB-20261030-001 trong answers.md. Chưa biết tôi là ai thì hỏi tên trước khi sửa. Tùng dùng mô hình/phương pháp khác trên cùng dataset; Đức thay đổi có kiểm soát mô hình hiện tại. Tự đề xuất và giải thích, không áp sẵn cách làm; giữ cùng giao thức đánh giá. PoseLab đã dừng, không làm web. Kiểm tra nhánh/phạm vi, cập nhật báo cáo mỗi phiên; hỏi nhóm qua file trao đổi Markdown cá nhân và đọc trả lời trên main. Không bịa kết quả hoặc tự thay model gốc.

## 🧠 Phương pháp

![Quy trình xử lý và kiến trúc ME-STGCN-TA](docs/images/me-stgcn-ta-pipeline.png)

Video → MediaPipe Pose → xử lý thiếu tư thế → lấy 32 mốc gần đều → 33 khớp → 8 kênh đầu vào → các khối ST-GCN → chú ý thời gian → phân loại Fall/ADL.

Tám kênh gồm tọa độ tương đối x/y, visibility, vận tốc khớp x/y, góc thân và dịch chuyển tâm hông x/y. Đây là biểu diễn đầu vào; mạng tiếp tục học các biểu diễn bên trong. Vận tốc dùng tọa độ chuẩn hóa, không có đơn vị m/s.

## Trạng thái demo cũ

PoseLab **đã tắt**, không còn hướng dẫn khởi chạy hoặc nhiệm vụ phát triển web. Launcher và khởi động ASGI bị chặn. Nguồn `app/`, [model/checkpoint cũ](models/deployment/) và kết quả được giữ để truy vết nghiên cứu, không xóa dữ liệu lịch sử. Xem [quyết định dừng](docs/tasks/web-app.md).

## 📓 Notebooks

| Notebook | Chức năng |
|---|---|
| [01 · Chuẩn bị dữ liệu](notebooks/01-prepare-pose-data.ipynb) | Trích xuất pose, tạo đặc trưng và thông tin chia tập |
| [02 · Baseline Motion-SVM](notebooks/02-baseline-motion-svm.ipynb) | Phân loại từ thống kê chuyển động |
| [03 · Baseline ST-GCN](notebooks/03-baseline-stgcn.ipynb) | Học từ ba kênh khung xương cơ sở |
| [06 · Mô hình triển khai](notebooks/06-final-model-nested-loso.ipynb) | Hiệu chỉnh góc thân, nested LOSO, huấn luyện toàn dữ liệu và xuất ONNX |
| [07 · Kiểm chứng thành phần](notebooks/07-component-ablation.ipynb) | So sánh đặc trưng, chú ý thời gian và chi phí lớp ngã |
| [08 · Kiểm chứng thời gian](notebooks/08-timing-ablation.ipynb) | So sánh hai cách tính khoảng thời gian của vận tốc |

<details>
<summary>Các giai đoạn ban đầu: 04, 04b và 05</summary>

- [04 · ME-STGCN-TA ban đầu](notebooks/history/04-proposed-me-stgcn-ta.ipynb).
- [04b · Điều chỉnh ưu tiên độ nhạy](notebooks/history/04b-sensitivity-tuning.ipynb).
- [05 · Huấn luyện và xuất mô hình ban đầu](notebooks/history/05-initial-model-export.ipynb).

Các bản này lưu lịch sử phương pháp, không phải các bước bắt buộc trước 06–08. Mô hình bản 05 khác checkpoint 06 được lưu trong `models/deployment/`; PoseLab hiện đã dừng.

</details>

[Hướng dẫn Kaggle](docs/notebook-guide.md) ghi rõ input, GPU, cách chạy và lưu phiên bản. [data/README.md](data/README.md) mô tả bốn tệp đầu vào; video và tensor dữ liệu không được đóng gói trong repo.

## 📊 Kết quả đã đo

Thí nghiệm dùng **160 video, 79 Fall, 81 ADL và bốn chủ thể**. Với kiểm chứng thành phần và thời gian, số liệu dưới đây là trung bình ba seed tại ngưỡng cố định 0,5. Ba seed không tạo thành 480 video độc lập.

| Cấu hình | Accuracy ± SD (%) | Sensitivity (%) | Specificity (%) |
|---|---:|---:|---:|
| A0 · 3 kênh, trung bình | 84,58 ± 4,73 | 85,65 | 83,54 |
| A1 · 8 kênh, trung bình | 89,38 ± 1,08 | 91,14 | 87,65 |
| A2 · 3 kênh, chú ý | 87,71 ± 0,72 | 93,25 | 82,30 |
| A3 · 8 kênh, chú ý | 89,58 ± 3,44 | 91,98 | 87,24 |
| A3 · Tăng chi phí lớp ngã ×2,25 | 89,58 ± 0,36 | **96,62** | 82,72 |
| T0 · Khoảng thời gian chung | 89,58 ± 0,36 | 96,62 | 82,72 |
| T1 · Khoảng thời gian từng cặp | 87,71 ± 2,60 | 94,09 | 81,48 |

<details>
<summary>Xem sơ đồ cây các cấu hình A0–A3 và T0–T1</summary>

![Cây cấu hình kiểm chứng](docs/images/model-variants.png)

</details>

**Không có accuracy trung bình 95%.** Tăng chi phí lớp ngã giảm bỏ sót nhưng tăng báo nhầm; T1 chưa cải thiện ổn định. Các nhánh là phép kiểm chứng, không phải chuỗi nâng cấp mà mỗi bước chắc chắn tốt hơn.

- [Kiểm chứng thành phần](docs/component-ablation.md) · [Dự đoán và kết quả gốc](results/component_ablation/README.md).
- [Kiểm chứng thời gian](docs/timing-ablation.md) · [Dự đoán và kết quả gốc](results/timing_ablation/README.md).
- [Đánh giá mô hình triển khai](results/deployment_evaluation/README.md): accuracy 87,50%, sensitivity 100%, specificity 75,31% thuộc mô hình theo fold, **không phải test độc lập của mô hình đã học toàn bộ 160 video**.
- [Thử nghiệm video ngoài](results/external_evaluation/README.md) và [thời gian xử lý](results/benchmark/README.md) được báo cáo riêng.

## 🗂️ Cấu trúc repo

```text
app/                         Nguồn demo PoseLab cũ, đã tắt
models/deployment/           Checkpoint và cấu hình cũ được bảo toàn
notebooks/                   Các bước chuẩn bị, baseline và kiểm chứng
notebooks/history/           Giai đoạn đề xuất và xuất mô hình ban đầu
training/component_ablation/ Mã kiểm chứng thành phần
results/                     Dự đoán, bảng kết quả và cấu hình thực nghiệm
docs/                        Một báo cáo Word, đề cương và ảnh minh họa
docs/tasks/                  Nhiệm vụ Tùng/Đức và phạm vi sửa code
docs/reports/                Tiến độ và bằng chứng bàn giao từng người
docs/discussions/            Hỏi-đáp nhóm qua Markdown trên từng nhánh
AGENTS.md                    Quy tắc cho AI khi nhận việc
scripts/                     Kiểm tra repo và tạo biểu đồ
data/README.md               Hướng dẫn dữ liệu đầu vào
```


[Báo cáo hiện tại](docs/bao-cao.docx) đang hoàn thiện; các mục chưa có bằng chứng được để trống. Repo chỉ giữ một bản Word để cập nhật tại cùng đường dẫn; các phiên bản đã commit có thể xem trong lịch sử Git. [Đề cương ban đầu](docs/de-cuong.pdf) được giữ dưới dạng PDF.

Kiểm tra tính toàn vẹn của mô hình, notebooks, kết quả và liên kết:

```bash
python scripts/check_repository.py
```


## Phạm vi sử dụng

Ứng dụng là bản thử nghiệm nghiên cứu, chưa được kiểm chứng như thiết bị y tế hoặc hệ thống cấp cứu. Kết quả phát triển trên bốn người không bảo đảm hiệu quả cho mọi người và mọi cảnh quay. Mã nguồn, dữ liệu và thư viện có điều kiện sử dụng riêng; xem [NOTICE](NOTICE.md).
