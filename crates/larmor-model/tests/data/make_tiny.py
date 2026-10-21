"""Write the tiny nanobrain encoder and its PyTorch outputs that larmor-model's unit tests check.

    cd tools && .venv/bin/python ../crates/larmor-model/tests/data/make_tiny.py

Writes, next to this script:
  tiny-vit.toml / tiny-vit.safetensors   a 2-block ViTMAE3D on a 16x24x12 grid, patch 4, two class tokens
  tiny-vit-encode.safetensors            patches, coords, tokens, cls, pooled for 50 shuffled patches

The grid's axes differ so a mixed-up position table fails, the patch subset is shuffled so coordinates
must be honoured, and every parameter is perturbed so a dropped bias or norm weight fails.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3] / "tools"))

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

import larmor_ref  # noqa: E402,F401  (puts nanobrain on sys.path)
import make_models  # noqa: E402
import nanobrain.model as nbm  # noqa: E402

ARGS = {
    "grid_size": [16, 24, 12],
    "patch_size": 4,
    "model_kwargs": {
        "depth": 2,
        "embed_dim": 32,
        "num_heads": 4,
        "decoder_depth": 1,
        "decoder_embed_dim": 16,
        "decoder_num_heads": 2,
        "class_tokens": 2,
        "drop_path_rate": 0.0,
    },
}


def main():
    torch.manual_seed(0)
    model = nbm.ViTMAE3D.from_config(ARGS)
    gen = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.startswith("decoder") or name == "mask_token":
                continue
            noise = torch.randn(p.shape, generator=gen)
            # larger than the reference models' perturbation so every tensor moves the output visibly
            if name.endswith("bias"):
                p.add_(0.1 * noise)
            elif ".norm" in name or name.startswith("norm"):
                p.add_(0.2 * noise)
            elif name.startswith(("pos_embed", "cls_token")):
                p.add_(0.5 * noise)
    model.eval()

    make_models.write_model(
        "tiny-vit",
        ARGS,
        make_models.encoder_state(model),
        HERE,
        description="Unit-test model: 2-block ViTMAE3D, 16x24x12 grid, patch 4, 2 class tokens",
        source={"model_class": "nanobrain.model:ViTMAE3D", "script": "tests/data/make_tiny.py"},
    )

    image = torch.randn((1, *ARGS["grid_size"]), generator=gen)
    patches = nbm.patchify3d(image, model.patch_size)
    order = torch.randperm(patches.shape[1], generator=gen)[:50]
    coords = model.coord_grid[:, order]
    with torch.inference_mode():
        x = model.forward_encoder(patches[:, order], coords)[0]
    c = model.class_tokens
    tokens, cls = x[c:], x[:c]
    save_file(
        {
            "patches": patches[0, order].contiguous(),
            "coords": coords[0].to(torch.int32).contiguous(),
            "tokens": tokens.contiguous(),
            "cls": cls.contiguous(),
            "pooled": tokens.mean(dim=0).contiguous(),
        },
        HERE / "tiny-vit-encode.safetensors",
        metadata={"model_id": "tiny-vit", "torch": torch.__version__},
    )
    print(f"wrote {HERE / 'tiny-vit-encode.safetensors'} ({len(order)} tokens)")


if __name__ == "__main__":
    main()
