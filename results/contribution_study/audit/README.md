# DUC-01 audit - tái tính từ OOF 07/08

Tạo bởi `python -m training.contribution_study.audit_existing` (28/10/2026). Không train, chỉ đọc `results/component_ablation/` và `results/timing_ablation/`. Phân tích: [contribution-study.md](../../../docs/research/contribution-study.md).

| Tệp | Nội dung |
|---|---|
| `audit_checks.json` | 22 kiểm tra PASS/FAIL, SHA-256 input/code, phiên bản môi trường |
| `{component,timing}_pooled_by_seed_recomputed.csv` | Metric pooled 160 clip mỗi variant/seed/policy, tính lại |
| `{component,timing}_seed_mean_std_recomputed.csv` | Mean/sample std (`ddof=1`) qua 3 seed |
| `paired_delta_by_seed[_policy].csv` | Delta từng seed cho 6 cặp một yếu tố, số clip sửa đúng/làm sai |
| `paired_delta_summary[_policy].csv` | Mean/std/min/max delta qua seed và số seed tăng/giảm |
| `paired_delta_by_subject.csv` | Delta theo subject outer-test (policy fixed) |
| `paired_clip_flips.csv` | Clip đổi đúng/sai giữa baseline và treatment, kèm score |
| `clip_error_frequency_fixed.csv`, `focus_clips_fixed.csv` | Số seed sai mỗi clip theo cấu hình (fixed) |

File không hậu tố là policy `fixed`. Delta là tỷ lệ (0-1); trong bản thảo đổi sang điểm phần trăm. Seed là lần lặp trên cùng 160 clip, không phải mẫu độc lập.
