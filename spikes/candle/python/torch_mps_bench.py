"""PyTorch MPS baseline for the Candle spike: nanobrain's own forward_encoder on the same GPU.

    tools/.venv/bin/python spikes/candle/python/torch_mps_bench.py MODEL.toml REFERENCE.safetensors \
        [--chunk 2048] [--runs 5] [--warmup 2]

Times fp32, fp16 (model.half()) and bf16 autocast, checks fp32 against the reference tokens, and
prints one JSON line per configuration. With --chunk, F.scaled_dot_product_attention is called on
query chunks (MPS otherwise materializes the full score matrix, which at patch 4 is ~85 GB).
"""

import argparse
import copy
import json
import statistics
import sys
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[3] / "tools"
sys.path.insert(0, str(TOOLS))

import larmor_ref  # noqa: E402,F401  (puts nanobrain on sys.path)
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

import nanobrain.modules as nbmod  # noqa: E402
from make_reference import load_model  # noqa: E402


def chunked_sdpa(chunk: int):
    sdpa = F.scaled_dot_product_attention

    def run(q, k, v, is_causal=False, **kwargs):
        assert not is_causal
        return torch.cat([sdpa(q[:, :, i : i + chunk], k, v, **kwargs) for i in range(0, q.shape[2], chunk)], 2)

    return run


def time_forward(model, patches, coords, dtype, autocast, warmup, runs):
    def once():
        torch.mps.synchronize()
        start = time.perf_counter()
        with torch.inference_mode(), torch.autocast("mps", dtype=dtype, enabled=autocast):
            out = model.forward_encoder(patches, coords)
        torch.mps.synchronize()
        return (time.perf_counter() - start) * 1e3, out

    cold, _ = once()
    for _ in range(max(warmup - 1, 0)):
        once()
    times = []
    for _ in range(runs):
        ms, out = once()
        times.append(ms)
    return cold, times, out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    if args.chunk:
        nbmod.F.scaled_dot_product_attention = chunked_sdpa(args.chunk)
    _, model = load_model(args.model)
    ref = load_file(args.reference)
    device = torch.device("mps")
    patches = ref["patches"][None].to(device)
    coords = ref["coords"].long()[None].to(device)
    c = model.class_tokens

    configs = [("fp32", torch.float32, False), ("bf16-autocast", torch.bfloat16, True),
               ("fp16-autocast", torch.float16, True), ("fp16", torch.float16, False)]
    for name, dtype, autocast in configs:
        m = (copy.deepcopy(model).half() if name == "fp16" else model).to(device)
        x = patches.half() if name == "fp16" else patches
        torch.mps.empty_cache()
        cold, times, out = time_forward(m, x, coords, dtype, autocast, args.warmup, args.runs)
        tokens = out[0, c:].float().cpu()
        cos = F.cosine_similarity(tokens, ref["tokens"], dim=1)
        pooled = F.cosine_similarity(tokens.mean(0), ref["pooled"], dim=0)
        record = {
            "framework": f"torch {torch.__version__} mps",
            "model": args.model.stem,
            "tokens": int(tokens.shape[0]),
            "precision": name,
            "attention": f"sdpa chunked:{args.chunk}" if args.chunk else "sdpa",
            "median_ms": statistics.median(times),
            "min_ms": min(times),
            "cold_ms": cold,
            "driver_allocated_gib": torch.mps.driver_allocated_memory() / 2**30,
            "min_token_cosine": cos.min().item(),
            "pooled_cosine": pooled.item(),
            "max_abs_diff": (tokens - ref["tokens"]).abs().max().item(),
        }
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
