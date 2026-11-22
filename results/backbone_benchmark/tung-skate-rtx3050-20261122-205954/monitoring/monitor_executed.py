import ctypes
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from ctypes import wintypes
from pathlib import Path

class Memory(ctypes.Structure):
    _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD)] + [
        (key, ctypes.c_uint64) for key in ("total", "available", "page_total",
        "page_available", "virtual_total", "virtual_available", "extended_available")]

def memory_used():
    state = Memory()
    state.length = ctypes.sizeof(state)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
        raise ctypes.WinError()
    return (state.total - state.available) / 2**20

root = Path(".runtime")
tag = time.strftime("tung-skate-rtx3050-%Y%m%d-%H%M%S")
mode = sys.argv[1]
output = Path(sys.argv[2])
tag = output.name + "-monitor-" + tag.rsplit("-rtx3050-", 1)[-1]
telemetry = root / (tag + "-telemetry.csv")
summary = root / (tag + "-measurement.json")
query = [shutil.which("nvidia-smi"), "-i", "0",
         "--query-gpu=temperature.gpu,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"]
command = [sys.executable, "-B", "-m", "training.backbone_benchmark.skate.runner",
           "--input-root", "data/gmdcsa24", "--output", str(output),
           "--mode", mode, "--device", "cuda"]
if mode == "full":
    command.extend(["--smoke-evidence", sys.argv[3]])
if output.exists():
    command.append("--resume")
backup = root / (tag + "-backup")
backup.mkdir(parents=True)
def copy_completed():
    lock = output / "locked_protocol.json"
    if lock.is_file() and not (backup / lock.name).exists():
        shutil.copy2(lock, backup / lock.name)
    folds = output / "folds"
    if folds.is_dir():
        target = backup / "folds"
        target.mkdir(exist_ok=True)
        for fold in sorted(folds.iterdir()):
            if fold.name.startswith(".") or not (fold / "fold.json").is_file():
                continue
            final = target / fold.name
            if not final.exists():
                staged = target / (".copy-" + fold.name)
                shutil.copytree(fold, staged)
                staged.rename(final)
                print("Backed up:", fold.name, flush=True)

env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
start = time.monotonic()
rows = []
reason = None
process = None
with telemetry.open("x", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["elapsed_seconds", "temp_c", "gpu_used_mib", "gpu_util_pct", "system_ram_used_mib"])
    try:
        while process is None or process.poll() is None:
            gpu = [int(v.strip()) for v in subprocess.check_output(query, text=True, timeout=5).strip().split(",")]
            row = [round(time.monotonic() - start, 3), *gpu, round(memory_used(), 1)]
            writer.writerow(row)
            stream.flush()
            rows.append(row)
            if gpu[0] >= 87:
                reason = "Reached NVIDIA target temperature (87 C)"
                break
            if process is None:
                process = subprocess.Popen(command, env=env)
                start = time.monotonic()
            copy_completed()
            if shutil.disk_usage(root).free < 2**30:
                reason = "SSD free space below 1 GiB"
                break
            time.sleep(0.7)
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=20)
        copy_completed()
        report = dict(backup=str(backup), **dict(output=str(output), telemetry=str(telemetry), seconds=round(time.monotonic()-start, 2),
                      exit_code=None if process is None else process.returncode, thermal_stop=reason,
                      samples=len(rows), baseline=rows[0] if rows else None,
                      peak=[max(row[i] for row in rows) for i in range(1,5)] if rows else None))
        summary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report))
sys.exit(1 if process is None or reason else process.returncode)

