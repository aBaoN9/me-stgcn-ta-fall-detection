# Modern benchmark — Đức (GB-20261124-001)

Năm model được giao: SkateFormer, BlockGCN, DeGCN, JT-GraphFormer, DS-GCN, so với A3 (`A3_motion_attention`, notebook 07) trên cùng hợp đồng v2. Thiết kế/M0 và mọi adaptation: [docs/research/modern-benchmark-duc.md](../../docs/research/modern-benchmark-duc.md). Giấy phép nguồn: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

| File | Vai trò |
|---|---|
| `graph.py` | Đồ thị 33 khớp MediaPipe từ cạnh A3: phân vùng spatial, hop, Dijkstra, bố cục 6×6 của SkateFormer |
| `skateformer.py`, `blockgcn.py`, `degcn.py`, `jt_graphformer.py` | Port từ repo tác giả @ SHA pin, mỗi file ghi rõ thay đổi |
| `models.py` | Registry ID → model, kiến trúc khóa, nguồn/giấy phép; `ds_gcn_v2_ce` dừng `BLOCKED` |
| `runner.py` | Nested LOSO một model/run: smoke, full, resume, kiểm fold, reload checkpoint, tổng hợp |
| `test_modern.py` | 21 test dương/âm; `--report` ghi JSON gắn hash code (full bắt buộc) |
| `tools/audit.py` | Kiểm chéo run-id, bảng chính fixed 0,5, delta cặp so A3, theo subject, FN/FP, clip đổi kết quả |
| `tools/latency.py` | Latency model-only batch 1 FP32, 30 warmup + 200 lần, cùng máy |

## Môi trường đã dùng

`.venv` riêng (Git bỏ qua): Python 3.12.10, torch 2.10.0+cpu, numpy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2; Windows 11, AMD Ryzen 7 5800H, CPU float32, **4 luồng torch cho mọi run**. Không cần dependency của repo tác giả (timm, einops, torch_topological, mmcv).

## Lệnh (chạy từ thư mục gốc repo; Windows dùng `.venv/Scripts/python`)

```bash
RUN=results/modern_benchmark_duc/duc-cpu-r7-5800h-20261126
python scripts/training_inputs.py --input-root data/gmdcsa24
python -B -m training.component_ablation.validate_pipeline --input-root data/gmdcsa24
python -B -m training.modern_benchmark_duc.test_modern --report $RUN/tests/test_report.json

# smoke thật mỗi model (seed 42, outer 1, 3 inner + 1 refit, tối đa 2 epoch)
python -B -m training.modern_benchmark_duc.runner --model skateformer_v2_ce --input-root data/gmdcsa24 \
    --output $RUN/smoke/skateformer_v2_ce --mode smoke --device cpu --threads 4

# full: cần smoke cùng model/code/env và báo cáo test PASS cùng code
python -B -m training.modern_benchmark_duc.runner --model skateformer_v2_ce --input-root data/gmdcsa24 \
    --output $RUN/full/skateformer_v2_ce --mode full --device cpu --threads 4 \
    --smoke-evidence $RUN/smoke/skateformer_v2_ce --test-report $RUN/tests/test_report.json

# tiếp tục sau khi bị ngắt: thêm --resume vào đúng lệnh full trên (cùng output/code/input/luồng)
# nếu tiến trình chết đột ngột, xác nhận không còn tiến trình ghi rồi mới xóa $RUN/full/<model>/.writer.lock

python -B -m training.modern_benchmark_duc.tools.audit --run $RUN --input-root data/gmdcsa24
python -B -m training.modern_benchmark_duc.tools.latency --output $RUN/latency --threads 4 --checkpoints $RUN/full
```

ID hợp lệ: `A3_motion_attention`, `skateformer_v2_ce`, `block_gcn_v2_ce`, `de_gcn_v2_ce`, `jt_graphformer_v2_ce`, `ds_gcn_v2_ce` (dừng `BLOCKED` vì không có giấy phép). ID khác bị từ chối, không fallback.

## Runner bảo đảm gì

- Dừng trước khi train nếu bốn input khác SHA-256 của lock 07, nhãn khác 81/79, mask sai kiểu hoặc split rò rỉ.
- `locked_protocol.json` ghi trước fit đầu: config, kiến trúc, nguồn/giấy phép, số tham số, hash input, hash từng file code (CRLF→LF), môi trường, thiết bị, số luồng, chữ ký smoke. Resume chỉ khi lock trùng hoàn toàn.
- Mỗi fold ghi vào `folds/.pending-<tag>`, được kiểm lại (chọn epoch từ lịch sử, early stopping, ngưỡng từ inner OOF, SHA checkpoint, nạp lại và chấm lại test) rồi mới đổi tên. Fold công bố hỏng làm run dừng, không bị ghi đè.
- `completion.json` chỉ ghi khi đủ 12 fold, 48 fit, 1.440 dòng dự đoán, mọi CSV tái tính khớp; smoke ghi `scientific_result=false`.
- Một tiến trình ghi mỗi thư mục (`.writer.lock`).

## Output mỗi model

`locked_protocol.json`, `folds/<model>_seed<s>_subject<o>/{fold.json,model.pt}`, `models/manifest.json` (SHA-256, byte, epoch, tham số, giây), `predictions.csv` (`sample_index,video_id,subject,label,score,prediction,threshold,variant,seed,policy`), `fold_metrics.csv`, `pooled_by_seed.csv`, `seed_mean_std.csv` (SD mẫu ddof=1), `errors.csv`, `inner_oof.csv`, `histories.csv`, `completion.json`. File `.pt` không vào Git; checkpoint full nằm ở GitHub Release asset ghi trong báo cáo.
