#!/bin/sh
# Headline timings, one configuration per process, each started only once the GPU looks idle
# (ioreg "Device Utilization %" below 10 for 3 s, waiting at most 10 min). Other agents share the
# GPU, so even these numbers are preliminary; each log line records whether idle was reached.
#
#   sh spikes/candle/scripts/headline.sh        -> spikes/candle/results/headline.{txt,jsonl}
set -eu
SPIKE="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$(cd "$SPIKE/../.." && pwd)"
OUT="$SPIKE/results"
BIN="$ROOT/target-candle/release/candle-spike"
MODELS="$ROOT/fixtures/models"
P8="$ROOT/fixtures/reference/encode/nanobrain-vits-p8-ref/mni152-t1w.safetensors"
P4="$ROOT/fixtures/reference/encode/nanobrain-vits-p4-ref/mni152-t1w.safetensors"
LOG="$OUT/headline.txt"
JSON="$OUT/headline.jsonl"
PY="$ROOT/tools/.venv/bin/python"
rm -f "$LOG" "$JSON"

CARGO_TARGET_DIR="$ROOT/target-candle" cargo build --release --manifest-path "$SPIKE/Cargo.toml"

gpu_util() {
    ioreg -r -c IOAccelerator -d 1 | grep -o '"Device Utilization %"=[0-9]*' | head -1 | cut -d= -f2
}

wait_idle() {
    waited=0
    quiet=0
    while [ "$quiet" -lt 3 ] && [ "$waited" -lt 600 ]; do
        if [ "$(gpu_util)" -lt 10 ]; then quiet=$((quiet + 1)); else quiet=0; fi
        sleep 1
        waited=$((waited + 1))
    done
    if [ "$quiet" -ge 3 ]; then echo "gpu idle after ${waited}s"; else echo "GPU STILL BUSY after ${waited}s"; fi
}

run() {
    state="$(wait_idle)"
    echo "== [$state] $*" | tee -a "$LOG"
    "$BIN" "$@" --json "$JSON" 2>&1 | tail -n +3 | tee -a "$LOG"
}

S8="--model $MODELS/nanobrain-vits-p8-ref.toml --input $P8"
S4="--model $MODELS/nanobrain-vits-p4-ref.toml --input $P4"
for opts in "f32 fused broadcast" "f32 fused folded" "f16 fused folded" "bf16 fused folded" \
    "f16-mixed fused folded" "bf16-mixed fused folded" "f32 chunked:2048 folded" "f16 chunked:2048 folded"; do
    set -- $opts
    run bench $S8 --precision "$1" --attention "$2" --bias "$3" --runs 7
done
for opts in "f32 fused folded" "f16 fused folded" "bf16 fused folded" "f16-mixed fused folded" \
    "bf16-mixed fused folded"; do
    set -- $opts
    run bench $S4 --precision "$1" --attention "$2" --bias "$3" --warmup 1 --runs 5
done
run bench $S4 --precision f16 --attention chunked:4096 --bias folded --warmup 1 --runs 3
for shape in 768x12x12 1024x24x16; do
    for precision in f32 f16 bf16-mixed; do
        run bench --shape $shape --patch 8 --input "$P8" --precision $precision --attention fused --bias folded --runs 5
    done
done
run attention --tokens 8151 --heads 6 --precision f32,f16,bf16 --attention fused,chunked:2048 --runs 10
run attention --tokens 59360 --heads 6 --precision f32,f16,bf16 --attention fused,chunked:2048 --warmup 1 --runs 5

state="$(wait_idle)"
echo "== [$state] torch MPS baseline, patch 8" | tee -a "$LOG"
"$PY" "$SPIKE/python/torch_mps_bench.py" "$MODELS/nanobrain-vits-p8-ref.toml" "$P8" --runs 7 | tee -a "$LOG"
state="$(wait_idle)"
echo "== [$state] torch MPS baseline, patch 4 (query chunks of 2048)" | tee -a "$LOG"
"$PY" "$SPIKE/python/torch_mps_bench.py" "$MODELS/nanobrain-vits-p4-ref.toml" "$P4" --chunk 2048 --warmup 1 --runs 3 \
    | tee -a "$LOG"
echo "done" | tee -a "$LOG"
