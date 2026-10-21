#!/bin/sh
# Full parity + timing matrix for the Candle spike. Results go to spikes/candle/results/:
# one .txt log per step and every configuration as a JSON line in results.jsonl.
#
#   sh spikes/candle/scripts/bench_all.sh            (from anywhere; takes ~30 min)
#
# The machine may be shared, so timings are preliminary.
set -eu
SPIKE="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$(cd "$SPIKE/../.." && pwd)"
OUT="$SPIKE/results"
BIN="$ROOT/target-candle/release/candle-spike"
MODELS="$ROOT/fixtures/models"
REF="$ROOT/fixtures/reference/encode"
JSON="$OUT/results.jsonl"
mkdir -p "$OUT"
rm -f "$JSON"

CARGO_TARGET_DIR="$ROOT/target-candle" cargo build --release --manifest-path "$SPIKE/Cargo.toml"

run() { # run NAME ARGS...: log to OUT/NAME.txt and echo
    name="$1"; shift
    echo "== $name: $*" | tee "$OUT/$name.txt"
    "$BIN" "$@" --json "$JSON" 2>&1 | tee -a "$OUT/$name.txt"
}

P8="$REF/nanobrain-vits-p8-ref/mni152-t1w.safetensors"
P4="$REF/nanobrain-vits-p4-ref/mni152-t1w.safetensors"
ALL=f32,f16,bf16,f16-mixed,bf16-mixed

run parity-p8 parity --model "$MODELS/nanobrain-vits-p8-ref.toml" --reference "$P8" \
    --precision $ALL --attention fused,chunked:2048 --bias broadcast,folded
run parity-p4 parity --model "$MODELS/nanobrain-vits-p4-ref.toml" --reference "$P4" \
    --precision $ALL --attention fused,chunked:2048 --bias folded

run attention-p8 attention --tokens 8151 --heads 6 --precision f32,f16,bf16 \
    --attention fused,chunked:1024,chunked:2048,chunked:4096,chunked:8192 --runs 10
run attention-p4 attention --tokens 59360 --heads 6 --precision f32,f16,bf16 \
    --attention fused,chunked:1024,chunked:2048,chunked:4096 --warmup 1 --runs 5

run bench-vits-p8 bench --model "$MODELS/nanobrain-vits-p8-ref.toml" --input "$P8" \
    --precision $ALL --attention fused,chunked:2048,chunked:8192 --bias broadcast,folded --runs 7
run bench-vits-p4-fused bench --model "$MODELS/nanobrain-vits-p4-ref.toml" --input "$P4" \
    --precision $ALL --attention fused --bias broadcast,folded --warmup 1 --runs 5
run bench-vits-p4-chunked bench --model "$MODELS/nanobrain-vits-p4-ref.toml" --input "$P4" \
    --precision $ALL --attention chunked:2048,chunked:4096 --bias folded --warmup 1 --runs 3

run bench-vitb-p8 bench --shape 768x12x12 --patch 8 --input "$P8" \
    --precision $ALL --attention fused,chunked:2048 --bias broadcast,folded --runs 5
run bench-vitl-p8 bench --shape 1024x24x16 --patch 8 --input "$P8" \
    --precision $ALL --attention fused,chunked:2048 --bias broadcast,folded --runs 5

# second pass over the headline configurations, to show run-to-run spread on a shared machine
run repeat-vits-p8 bench --model "$MODELS/nanobrain-vits-p8-ref.toml" --input "$P8" \
    --precision f32,f16,bf16,bf16-mixed --attention fused --bias folded --runs 7
run repeat-vits-p4 bench --model "$MODELS/nanobrain-vits-p4-ref.toml" --input "$P4" \
    --precision f32,f16,bf16,bf16-mixed --attention fused --bias folded --warmup 1 --runs 5
echo "done: $JSON"
