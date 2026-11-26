#!/usr/bin/env bash
# Run full nested LOSO for several models one after another (one "lane"), resuming a model whose run directory
# already exists. Every full run is guarded by the runner (test report + real smoke of the same model/code/env).
#   bash training/modern_benchmark_duc/tools/run_lane.sh <run-dir> <python> <model> [<model> ...]
# Logs: <run-dir>/logs/full_<model>.log. Stops the lane at the first failing model.
set -u
RUN="$1"; PY="$2"; shift 2
mkdir -p "$RUN/logs" "$RUN/full"
for MODEL in "$@"; do
  OUT="$RUN/full/$MODEL"
  RESUME=""
  [ -f "$OUT/locked_protocol.json" ] && RESUME="--resume"
  echo "$(date '+%F %T') start $MODEL $RESUME" >> "$RUN/logs/lanes.log"
  PYTHONIOENCODING=utf-8 "$PY" -B -u -m training.modern_benchmark_duc.runner --model "$MODEL" \
    --input-root data/gmdcsa24 --output "$OUT" --mode full --device cpu --threads 4 \
    --smoke-evidence "$RUN/smoke/$MODEL" --test-report "$RUN/tests/test_report.json" $RESUME \
    >> "$RUN/logs/full_$MODEL.log" 2>&1
  STATUS=$?
  echo "$(date '+%F %T') end $MODEL exit=$STATUS" >> "$RUN/logs/lanes.log"
  [ $STATUS -ne 0 ] && exit $STATUS
done
exit 0
