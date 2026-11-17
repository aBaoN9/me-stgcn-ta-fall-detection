# Contribution study (Đức)

Code của DUC-01 đến DUC-03. Chỉ đọc/import `training/component_ablation/`, không sửa nguồn/kết quả cũ.

## DUC-01 audit

```bash
python -m training.contribution_study.audit_existing            # ghi results/contribution_study/audit
python -m training.contribution_study.audit_existing --output <dir>
```

Cần Python 3.10+, numpy, pandas; không cần torch/GPU/dataset. Exit code 0 khi mọi kiểm tra PASS. Đã chạy với Python 3.12.10, pandas 3.0.5, numpy 2.5.3 trên Windows 11.

## DUC-02/03 - quality-aware pooling (M0 đã khóa 17/11, `GB-20261116-002`)

| File | Vai trò |
|---|---|
| `quality_pooling.py` | `QualityAwareSkeletonModel`: A3 + prior `log(0,1 + q')` trong softmax attention; tắt prior = `SkeletonModel` |
| `runner.py` | Nested LOSO Q0/Q1 theo recipe 07, xuất predictions/fold/pooled/seed/errors/completion/locked_protocol |
| `check_quality_pooling.py` | Kiểm parity, shape sau stride, all-missing, q theo augmentation, gradient, smoke runner trên dữ liệu giả |
| `benchmark_latency.py` | Latency model-only PyTorch batch 1 và số tham số; đồng bộ đúng `cuda`/`cuda:N` |
| `check_review_fixes.py` | Test hồi quy dương/âm DUC-REV-01..04 và nạp lại model: guard hash input, nghiệm thu fold/resume, đồng bộ CUDA, audit NaN, reload checkpoint |
| `evaluate_models.py` | Nạp lại từng checkpoint `<run>/models/*.pt`, chấm lại subject outer, so với `predictions.csv`/metric đã lưu; ghi `models/manifest.json` và `model_reload_check.json` |
| `quality_report.py` | Phân bố q/q' và frame mất pose trên input thật, sau guard hash, trước full; chỉ mô tả input |
| `kaggle_q0_q1.ipynb` | Notebook Kaggle GPU: guard → validate → q → test → smoke thật → full → audit → reload → zip |

Cần môi trường riêng có torch (không cài đè hệ thống). Đã kiểm với `.venv` (bị Git bỏ qua): Python 3.12.10, torch 2.10.0+cpu, numpy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pip install numpy==2.2.6 pandas==2.3.3 scikit-learn==1.7.2
.venv/Scripts/python -m training.contribution_study.check_quality_pooling    # .runtime/contribution_study/quality_checks.json
.venv/Scripts/python -m training.contribution_study.check_review_fixes       # .runtime/contribution_study/review/review_checks.json
.venv/Scripts/python -m training.contribution_study.benchmark_latency        # results/contribution_study/latency/
```

M0 khóa trong `results/contribution_study/m0_locked.json` (config + code hash); notebook Kaggle dừng nếu code/config khác bản khóa. Theo `GB-20261116-002`, chạy smoke thật rồi full khi đủ bốn input đúng hash và GPU, không chờ duyệt riêng. Mỗi run một thư mục mới; runner từ chối ghi đè khi config/code/input/môi trường đổi. Runner dừng trước khi train nếu bốn input khác hash trong `results/component_ablation/locked_protocol.json`. Resume chỉ bỏ qua fold đã kiểm đủ; fold hỏng được chạy lại:

```bash
python -m training.contribution_study.runner --input-root data/local --mode smoke --output .runtime/contribution_study/smoke_real
python -m training.contribution_study.runner --input-root <input> --mode full --output results/contribution_study/runs/q0_q1_full_<YYYYMMDD>
python -m training.contribution_study.quality_report --input-root <input> --output <run>_precheck/quality
python -m training.contribution_study.audit_existing --run <run> --output <run>/audit   # tái tính metric + delta Q0→Q1
python -m training.contribution_study.evaluate_models --run <run> --input-root <input>  # nạp lại 24 checkpoint
```

Full cần GPU (24 outer, 96 fits). Checkpoint nằm ở `<run>/models/`, mỗi file là model refit trên ba subject và chỉ có metric outer của subject còn lại. Không sửa số tay; `completion.json` chỉ được ghi khi mọi fold, policy, OOF và artifact qua kiểm tra.
