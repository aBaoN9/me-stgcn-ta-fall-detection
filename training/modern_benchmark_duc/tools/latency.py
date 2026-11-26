"""Model-only latency on one machine (GB-20261124-001 section 10): FP32, batch 1, eval/no_grad,
30 warmup + 200 timed calls, median/p95 milliseconds and parameter count, all models in one process.

    python -m training.modern_benchmark_duc.tools.latency --output <run>/latency --threads 4 [--checkpoints <run>/full]

Run on an otherwise idle machine. With --checkpoints, each model loads its first full checkpoint so the weights
are the trained ones (latency does not depend on them, but the loaded model is what is timed).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from training.component_ablation import training as original
from training.modern_benchmark_duc import runner
from training.modern_benchmark_duc.models import A3, ASSIGNED, DS_GCN, build_model, logits_of


def measure(model, device, warmup=30, runs=200):
    x = torch.randn(1, 8, 32, 33, device=device)
    model.eval()
    timings = []
    with torch.no_grad():
        for index in range(warmup + runs):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            logits_of(model, x)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if index >= warmup:
                timings.append((time.perf_counter() - started) * 1000)
    return dict(median_ms=float(np.median(timings)), p95_ms=float(np.percentile(timings, 95)),
                mean_ms=float(np.mean(timings)), runs=runs, warmup=warmup)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--checkpoints", type=Path, help="<run>/full; load the first checkpoint of each model")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    rows = []
    for model_id in (A3,) + tuple(m for m in ASSIGNED if m != DS_GCN):
        torch.manual_seed(0)
        source = "random init"
        if args.checkpoints and (args.checkpoints / model_id / "models/manifest.json").exists():
            first = runner.read_json(args.checkpoints / model_id / "models/manifest.json")[0]
            model = runner.load_model(args.checkpoints / model_id / first["file"], device)
            source = first["file"]
        else:
            model = build_model(model_id).to(device)
        row = dict(model=model_id, parameters=sum(p.numel() for p in model.parameters()), weights=source,
                   peak_memory_bytes=None, **measure(model, device))
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            measure(model, device, warmup=0, runs=5)
            row["peak_memory_bytes"] = torch.cuda.max_memory_allocated(device)
        rows.append(row)
        print(json.dumps(row), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    original.write_json(args.output / "model_only_latency.json", dict(
        scope="model-only forward, batch 1, FP32, eval + no_grad; excludes MediaPipe, feature building and I/O",
        environment=runner.environment(device, torch.get_num_threads()), models=rows,
        note="Peak memory is not measured on CPU. One machine only; do not rank against another machine's table."))


if __name__ == "__main__":
    main()
