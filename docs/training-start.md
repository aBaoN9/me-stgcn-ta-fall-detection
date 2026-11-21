# Bắt đầu làm bài và train - Tùng / Đức

**Nghiệm thu bổ sung 21/11/2026:** GRU/CTR-GCN adapted của Tùng đã được nhận qua PR #3 theo `GB-20261121-001`, kèm [hướng dẫn runner](../training/backbone_benchmark/supplement/README.md), checkpoint/kết quả và bản sửa monitor tạo thư mục cha. Không cần train lại để nhận bản sửa; chạy kiểm thử mới theo README. Chưa cập nhật Word/đổi deployment hoặc nghiệm thu bài Đức.

**Nghiệm thu 20/11/2026:** benchmark TCN–A3 của Tùng đã merge vào main qua PR #2 theo `GB-20261120-001`; code, checkpoint và kết quả [đã có sẵn](../results/backbone_benchmark/tung-m2-rtx3050-20261120-165523/README.md), không cần train lại để hoàn tất bàn giao này. Hướng dẫn bên dưới vẫn dùng cho kiểm tra/chạy mới có chủ đích; việc Tùng hoàn tất không đồng nghĩa phần Đức đã được nghiệm thu.

Cập nhật 19/11/2026 theo `GB-20261119-002` trong [answers](discussions/answers.md). Hai bạn tự triển khai, khóa phương án, test/smoke thật rồi full; **không chờ Gia Bảo duyệt từng bước**. Hạn model đã train + đánh giá thật: **25/11/2026 (UTC+7)**. Đây là hướng dẫn hỗ trợ chạy, không phải kết quả mới hay nghiệm thu code của thành viên.

## 1. Nhận cập nhật, không mất bài đang làm

```bash
git status --short
git fetch origin
git show origin/main:docs/discussions/answers.md
```

Giữ nhánh `dinhtung` hoặc `vinhduc`. Nếu worktree sạch, nhận tài liệu/input/tool bằng `git merge origin/main`; nếu có bài chưa commit thì lưu trên nhánh mình trước, không reset/force-push. Xử lý xung đột có chủ đích, không ghi đè bài người khác. Không merge nhánh bạn kia hoặc sửa helper/model gốc để lấy dữ liệu. Notebook đã pin commit cũ cần cách tách code/data ở mục 3; chỉ merge main ngoài notebook chưa đủ.

## 2. Kiểm input trên máy local

```bash
python scripts/training_inputs.py --input-root data/gmdcsa24
```

Tool chỉ dùng thư viện chuẩn Python, không cần Torch/GPU. Mặc định chỉ kiểm tra, không tải hoặc sửa file. Nếu thiếu file và có Internet:

```bash
python scripts/training_inputs.py --input-root data/gmdcsa24 --download
```

Chỉ tải file còn thiếu từ data commit bất biến `afc7370c324aabec1a6d1076b19d8247c818d96c`, kiểm dung lượng/SHA-256 trước khi lưu, không ghi đè file có sẵn. Nếu file cũ sai hash, lệnh FAIL trước khi tải; giữ bản đó riêng để truy nguyên rồi tự chọn thư mục sạch. Không sửa CSV hoặc đổi hash chuẩn cho qua guard. Xem [manifest/nguồn/giấy phép](../data/README.md).

Tool PASS chỉ chứng minh đúng bốn file. Tiếp theo, trong môi trường huấn luyện phù hợp:

```bash
python -m training.component_ablation.validate_pipeline --input-root data/gmdcsa24
```

Kiểm phiên bản theo [requirements hiện hành](../training/component_ablation/requirements.txt) và README phần riêng của mình. Dùng môi trường ảo/runtime riêng; không tự gỡ hoặc cài đè Torch/CUDA hệ thống. Ghi Python, Torch, NumPy, pandas, sklearn, thiết bị, CUDA/driver và số luồng CPU. Không coi máy điều phối PASS là máy mình PASS. Không sửa helper chung để vá dependency.

## 3. Kaggle: tách commit code và commit dữ liệu

Đã đọc `vinhduc` commit `eac041170f4d85a059386c87df2f1ad332a91a38`: notebook `training/contribution_study/kaggle_q0_q1.ipynb` checkout code `c72f33a547d3f130d3768611c08949cf45c37b56`; cell tìm input chỉ quét `/kaggle/input`. Commit code đó chưa có dữ liệu mới. Vì vậy **không chỉ chạy lại nguyên notebook rồi chờ dữ liệu tự xuất hiện**, và không đổi sang HEAD main để né code/config lock.

Chọn một trong hai cách, ghi cách đã dùng vào báo cáo:

- Không có Internet trong phiên chạy: tự tạo/gắn Kaggle Dataset chứa đúng bốn input đã được phép công khai trên main, giữ byte/tên. Đảm bảo chỉ có một `gmdcsa24_pose_32.npz` dưới `/kaggle/input`; input cùng một thư mục và hash đúng. Cách này dùng được cell tìm input cũ. Notebook còn cần code/dependency sẵn trong phiên; clone/fetch cũng cần Internet, nên chuẩn bị code đã khóa và môi trường từ trước nếu chạy offline.
- Có Internet: sau cell checkout code và kiểm M0, **thay toàn bộ cell tìm input cũ** bằng đoạn dưới. Đoạn này dùng `CODE`, `WORK` từ notebook của Đức, chỉ lấy helper độc lập, không checkout hay sửa mã khoa học. Tùng có thể dùng cùng helper với đường dẫn repo/runtime của mình.

```python
import subprocess
import sys
from pathlib import Path

subprocess.run(["git", "-C", str(CODE), "fetch", "origin", "main"], check=True)
helper = WORK / "training_inputs.py"
helper.write_bytes(subprocess.run(
    ["git", "-C", str(CODE), "show", "FETCH_HEAD:scripts/training_inputs.py"],
    capture_output=True, check=True,
).stdout)
INPUT_ROOT = WORK / "gmdcsa24"
subprocess.run([
    sys.executable, str(helper), "--input-root", str(INPUT_ROOT), "--download",
], check=True)
from training.component_ablation.features import load_inputs
from training.contribution_study.runner import verify_inputs
data = load_inputs(INPUT_ROOT)
print("Input guard:", verify_inputs(data, make_config("full")))
```

Giữ các assert code/config và guard bốn hash, validation, test và smoke thật của notebook. Không chạy lại cell cũ để ghi đè `INPUT_ROOT` về `/kaggle/input`. Lưu notebook đã chỉnh cell vận hành, HEAD code thực tế, helper/data commit và phiên bản môi trường. Nếu thay mã khoa học/config thì tự tạo M0 addendum, khóa hash mới **trước outer-test**, không bỏ guard.

## 4. Không có GPU vẫn được làm và full CPU

`GB-20261119-002` trả lời trực tiếp `DUC-20261117-001`: **cả Đức và Tùng được tự chọn full CPU** nếu không có GPU, không cần xin lại. Hai vế so sánh chạy cùng thiết bị/recipe/ngân sách; ghi CPU, RAM, số luồng, thời gian và giới hạn, không trộn kết quả CPU với một vế GPU rồi gọi là đối chứng cùng điều kiện. Ước tính synthetic của Đức không phải cam kết thời gian thực tế hay bằng chứng full thật.

Quyền mới không tự thay code cũ. Runner của Đức hiện chỉ có `--input-root`, `--mode`, `--output`, **chưa có `--allow-cpu-full`**; config khóa còn `allow_cpu_full=false`, notebook full còn assert CUDA. Đức được tự bổ sung lựa chọn CPU trong runner riêng/notebook của mình, cập nhật M0 addendum cùng code/config hash mới trước outer-test, giữ guard config/inputs/test. Không đổi `training/component_ablation/` để mở CPU và không chỉ xóa assert rồi coi đã hợp lệ. Tùng tự xử lý tương tự trong phần riêng. Không cần Gia Bảo duyệt addendum này.

Nếu chữ ký code/config/input/môi trường/thiết bị thay đổi: dùng **thư mục output mới**, không sửa lock của run đã bắt đầu hoặc ép resume khác chữ ký. Giữ đủ folds, seeds, nested LOSO và nguyên tắc chọn cấu hình/ngưỡng; không giảm thí nghiệm để gọi là full. Chỉ được dùng tài nguyên miễn phí/đã có quyền; quyết định này không cấp tiền dịch vụ hoặc tài khoản/quota.

## 5. Điểm bắt đầu và bằng chứng nộp

- Tùng: M0–M3 TCN–A3 đã nghiệm thu, runner nằm trong `training/backbone_benchmark/` trên main và run đã nộp giữ nguyên. Đọc README để kiểm checkpoint/dự đoán mà không train lại. Runner hiện yêu cầu CUDA/smoke thật cùng môi trường trước full, chưa có đường full CPU; nếu nghiên cứu tiếp theo CPU thì tự cập nhật mã/lock trong vùng riêng, không bỏ guard. Không bắt làm thêm để A3 thắng mới chấp nhận bài.
- Đức: tiếp tục phần riêng `training/contribution_study/` trên `vinhduc`; đọc README của nhánh mình. Kiểm `check_quality_pooling`, `check_review_fixes`, `check_analysis`, validation/input guard, báo cáo quality và smoke **dữ liệu thật**. Full Q0/Q1 theo M0 hiện hành phải đủ 24 outer / 96 fits; sau full chạy `audit_existing`, `evaluate_models`, `analyze_q0_q1`. Đây là lệnh của nhánh Đức, chưa tích hợp lên main, không phải check đã chạy thay Đức trong phiên này.
- Cả hai: nộp checkpoint/model load lại được, manifest SHA-256, config/code/input/environment lock, lệnh tái lập, raw OOF/predictions, metric theo fold/seed, threshold, log/completion, audit/reload và phân tích sai số/giới hạn. Xem checklist chính thức `GB-20261116-001` trong [answers](discussions/answers.md); kết quả không cải thiện vẫn hợp lệ nếu trung thực.
- Artifact nhỏ: được tự commit đích danh trên nhánh mình; với file bị ignore chỉ `git add -f` đúng từng đường dẫn checkpoint đã kiểm cùng manifest, **không `git add -f .`**, không mở ignore toàn bộ model/data. Artifact không phù hợp Git: tự dùng GitHub Release và ghi URL/hash/cách tải/nạp, không xin lại việc chọn cách lưu. Không cần gh CLI nếu dùng giao diện GitHub; không đưa token, tài khoản riêng, video hoặc file runtime thừa vào repo.
- Sau mỗi phiên cập nhật báo cáo cá nhân: commit/thiết bị/lệnh/PASS-FAIL-NOT_RUN/blocker, rồi push nhánh mình. Đọc quyết định mới trên main nhưng không tự đánh dấu người điều phối đã nghiệm thu. Có thể mở PR khi đủ bằng chứng, không chờ duyệt để bắt đầu làm.

## Còn phụ thuộc máy thành viên

Main đã cấp input, hướng dẫn và quyền; chưa chứng minh GPU/quota/tài khoản, dependency, smoke thật hoặc train trên máy hai bạn. Không có quyền cấp tài khoản Kaggle/GPU thay thành viên. Nếu hết RAM/quota, không có quyền phần cứng, input sai hash hoặc test lỗi: tự xử lý trong phạm vi và ghi blocker cụ thể, không bịa kết quả, không chờ duyệt phương án đã được cấp quyền. Không đổi dataset/giao thức chung/deployment và không bật lại PoseLab.
