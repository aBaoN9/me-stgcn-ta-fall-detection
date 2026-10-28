"""DUC-03 cost: model-only latency and parameters for Q0/Q1 (PyTorch, batch 1, eval, no grad, input (1,8,32,33)).

Weights are freshly initialised; latency does not depend on trained values. This is not ONNX, not MediaPipe/feature
extraction and not end-to-end pipeline latency.
"""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from training.contribution_study.quality_pooling import QualityAwareSkeletonModel

ROOT = Path(__file__).resolve().parents[2]


def measure(model, values, warmup, repeats, device):
    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
    with torch.no_grad():
        for _ in range(warmup):
            model(values)
        sync()
        times = []
        for _ in range(repeats):
            started = time.perf_counter()
            model(values)
            sync()
            times.append((time.perf_counter() - started) * 1000)
    return times


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=5, help="alternate Q0/Q1 rounds to reduce drift bias")
    parser.add_argument("--output", type=Path, default=ROOT / "results/contribution_study/latency/model_only_pytorch_cpu.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    values = torch.rand(1, 8, 32, 33, device=args.device)
    values[:, 2] = values[:, 2].clamp(0, 1)
    models = {name: QualityAwareSkeletonModel(8, True, quality_prior=prior).to(args.device).eval()
              for name, prior in [("Q0_A3_attention", False), ("Q1_A3_quality_prior", True)]}
    samples = {name: [] for name in models}
    for round_index in range(args.rounds):
        for name in (list(models) if round_index % 2 == 0 else list(models)[::-1]):
            samples[name].extend(measure(models[name], values, args.warmup, args.repeats, args.device))
    report = {"backend": f"PyTorch {torch.__version__} eager", "device": args.device, "threads": args.threads,
              "cpu": platform.processor(), "platform": platform.platform(), "python": platform.python_version(),
              "input_shape": list(values.shape), "precision": "float32", "batch": 1,
              "warmup_per_round": args.warmup, "repeats_per_round": args.repeats, "rounds": args.rounds,
              "models": {name: {"parameters": sum(p.numel() for p in models[name].parameters()),
                                "median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95)),
                                "measurements": len(times)} for name, times in samples.items()},
              "scope": "model-only; excludes MediaPipe, feature extraction, ONNX and app pipeline"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
