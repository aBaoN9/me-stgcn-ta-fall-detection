# Contribution study (Đức)

Code của DUC-01 đến DUC-03. Chỉ đọc/import `training/component_ablation/`, không sửa nguồn/kết quả cũ.

## DUC-01 audit

```bash
python -m training.contribution_study.audit_existing            # ghi results/contribution_study/audit
python -m training.contribution_study.audit_existing --output <dir>
```

Cần Python 3.10+, numpy, pandas; không cần torch/GPU/dataset. Exit code 0 khi mọi kiểm tra PASS. Đã chạy với Python 3.12.10, pandas 3.0.5, numpy 2.5.3 trên Windows 11.

## DUC-02/03 - quality-aware pooling (M0 chờ duyệt)

| File | Vai trò |
|---|---|
| `quality_pooling.py` | `QualityAwareSkeletonModel`: A3 + prior `log(0,1 + q')` trong softmax attention; tắt prior = `SkeletonModel` |
| `runner.py` | Nested LOSO Q0/Q1 theo recipe 07, xuất predictions/fold/pooled/seed/errors/completion/locked_protocol |
| `check_quality_pooling.py` | Kiểm parity, shape sau stride, all-missing, q theo augmentation, gradient, smoke runner trên dữ liệu giả |
| `benchmark_latency.py` | Latency model-only PyTorch batch 1 và số tham số |

Cần môi trường riêng có torch (không cài đè hệ thống). Đã kiểm với `.venv` (bị Git bỏ qua): Python 3.12.10, torch 2.10.0+cpu, numpy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pip install numpy==2.2.6 pandas==2.3.3 scikit-learn==1.7.2
.venv/Scripts/python -m training.contribution_study.check_quality_pooling    # .runtime/contribution_study/quality_checks.json
.venv/Scripts/python -m training.contribution_study.benchmark_latency        # results/contribution_study/latency/
```

Chạy thật chỉ sau khi người điều phối duyệt M0 (`DUC-20261028-001`). Mỗi run một thư mục mới; runner từ chối ghi đè khi config/code/input/môi trường đổi:

```bash
python -m training.contribution_study.runner --input-root data/local --mode smoke --output .runtime/contribution_study/smoke_real
python -m training.contribution_study.runner --input-root <input> --mode full --output results/contribution_study/runs/q0_q1_full_<YYYYMMDD>
```

Full cần GPU (24 outer, 96 fits). Sau full, chạy audit cặp Q0→Q1 bằng cách đọc `predictions.csv` của run, không sửa số tay.
