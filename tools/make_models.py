"""Write Larmor model manifests (model.toml) and encoder weights (safetensors).

    python make_models.py reference              # perturbed random-init ViT-S, patch 8 and 4
    python make_models.py export CKPT --id ID     # a real nanobrain checkpoint (.pth) -> ID.toml + ID.safetensors

Only the encoder is exported; the MAE decoder is never used for embeddings.
"""

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file

from larmor_ref import FIXTURES, sha256_file, toml_value

import nanobrain.model as nbm  # noqa: E402  (path set up by larmor_ref)

# nanobrain/config/default_pretrain.yaml, the ViT-S encoder every current checkpoint uses
VITS_ARGS = {
    "grid_size": [192, 240, 192],
    "patch_size": 8,
    "model_kwargs": {
        "depth": 12,
        "embed_dim": 384,
        "num_heads": 6,
        "decoder_depth": 4,
        "decoder_embed_dim": 512,
        "decoder_num_heads": 8,
        "class_tokens": 1,
        "drop_path_rate": 0.0,
    },
}

PREPROCESSING_STEPS = [
    {"op": "threshold_mask", "resolution": 2.0, "sigma": 4.0, "remove_islands": True, "fill_holes": True},
    {
        "op": "conform",
        "min_voxel_size": 1.0,
        "fov": [192.0, 240.0, 192.0],
        "top_margin": 5.0,
        "fixed_grid": True,
        "order": 1,
    },
    {"op": "quantize", "low": 0.005, "high": 0.995},
    {"op": "normalize", "method": "max_zscore", "std_floor": 1e-6},
]


def encoder_state(model: nbm.ViTMAE3D) -> dict[str, torch.Tensor]:
    state = model.state_dict()
    keep = {}
    for name, tensor in state.items():
        if name.startswith("decoder") or name == "mask_token":
            continue
        keep[name] = tensor.detach().to(torch.float32).contiguous()
    return keep


def write_model(model_id: str, args: dict, state: dict, out_dir: Path, description: str, source: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / f"{model_id}.safetensors"
    save_file(state, weights_path, metadata={"model_id": model_id, "naming": "nanobrain"})
    sha = sha256_file(weights_path)

    kw = args["model_kwargs"]
    arch = {
        "grid": list(args["grid_size"]),
        "patch": int(args["patch_size"]),
        "width": int(kw["embed_dim"]),
        "depth": int(kw["depth"]),
        "heads": int(kw["num_heads"]),
        "mlp_ratio": float(kw.get("mlp_ratio", 4)),
        "class_tokens": int(kw.get("class_tokens", 1)),
        "pos_embed": "separable",
        "qkv": "separate",
        "qkv_bias": bool(kw.get("qkv_bias", True)),
        "proj_bias": bool(kw.get("proj_bias", True)),
        "norm_eps": 1e-6,
        "activation": "gelu",
    }
    lines = [
        f"id = {toml_value(model_id)}",
        'family = "vit3d"',
        f"description = {toml_value(description)}",
        "",
        "[architecture]",
        *(f"{k} = {toml_value(v)}" for k, v in arch.items()),
        "",
        "[readout]",
        'tokens = "last"',
        'pool = "mean"',
        "",
        "[weights]",
        f"file = {toml_value(weights_path.name)}",
        f"sha256 = {toml_value(sha)}",
        'url = ""',
        'naming = "nanobrain"',
    ]
    for step in PREPROCESSING_STEPS:
        lines += ["", "[[preprocessing.steps]]", *(f"{k} = {toml_value(v)}" for k, v in step.items())]
    lines += ["", "[source]", *(f"{k} = {toml_value(v)}" for k, v in source.items()), ""]
    (out_dir / f"{model_id}.toml").write_text("\n".join(lines))
    print(f"wrote {out_dir / model_id}.toml  sha256 {sha[:12]}  ({len(state)} tensors)")


def perturbed_reference(patch_size: int) -> tuple[dict, nbm.ViTMAE3D]:
    """nanobrain's ViT-S init with every encoder parameter made non-trivial.

    A fresh init has zero biases and unit norms, so a bug that dropped a bias or a norm weight would
    still match. Perturbing them makes every tensor in the checkpoint matter for parity.
    """
    args = {**VITS_ARGS, "patch_size": patch_size}
    torch.manual_seed(0)
    model = nbm.ViTMAE3D.from_config(args)
    gen = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.startswith("decoder") or name == "mask_token":
                continue
            noise = torch.randn(p.shape, generator=gen)
            if name.endswith("bias"):
                p.add_(0.02 * noise)
            elif ".norm" in name or name.startswith("norm"):
                p.add_(0.1 * noise)  # LayerNorm weights around 1
            elif name.startswith(("pos_embed", "cls_token")):
                p.add_(0.02 * noise)
    model.eval()
    return args, model


def cmd_reference(args):
    out_dir = Path(args.out)
    for patch in (8, 4):
        model_args, model = perturbed_reference(patch)
        write_model(
            f"nanobrain-vits-p{patch}-ref",
            model_args,
            encoder_state(model),
            out_dir,
            description=f"Reference: nanobrain ViT-S, patch {patch}, seeded init with perturbed biases and norms",
            source={"model_class": "nanobrain.model:ViTMAE3D", "init": "manual_seed(0) + perturbation seed 1"},
        )


def cmd_export(args):
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    assert ckpt["model_class"] == "nanobrain.model:ViTMAE3D", f"unsupported {ckpt['model_class']}"
    model = nbm.ViTMAE3D.from_config(ckpt["args"])
    model.load_state_dict(ckpt["model"])
    write_model(
        args.id,
        ckpt["args"],
        encoder_state(model),
        Path(args.out),
        description=args.description or f"nanobrain checkpoint {Path(args.checkpoint).name}",
        source={
            "model_class": ckpt["model_class"],
            "checkpoint": Path(args.checkpoint).name,
            "git_sha": ckpt.get("git_sha", ""),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    ref = sub.add_parser("reference")
    ref.add_argument("--out", default=str(FIXTURES / "models"))
    ref.set_defaults(func=cmd_reference)
    exp = sub.add_parser("export")
    exp.add_argument("checkpoint")
    exp.add_argument("--id", required=True)
    exp.add_argument("--out", default=".")
    exp.add_argument("--description", default="")
    exp.set_defaults(func=cmd_export)
    args = parser.parse_args()
    args.func(args)
