# Models

A Larmor model is two files side by side: a `model.toml` manifest (what the network is, how scans are
prepared for it, and the SHA-256 of its weights) and a `.safetensors` file holding the encoder weights.
A new model is a new manifest, not new code, as long as its architecture and preprocessing use values
Larmor supports. The manifest format is in `docs/DESIGN.md`, "Model manifest".

Weights are never committed (`*.safetensors` is git-ignored), so this folder holds manifests and this
README; put the weights next to their manifest.

## Where `larmor` looks

`--model` takes either a path to a `.toml` file or a model id. An id is looked up in these folders, in
order, and the first manifest whose `id` matches wins (`<id>.toml` first, then any `.toml` in the folder):

1. every folder in `$LARMOR_MODELS` (colon-separated, like `PATH`),
2. the user data folder, `~/Library/Application Support/larmor/models`,
3. `./models` in the current folder (this folder, when you run `larmor` from the repository root).

`larmor models list` prints the folders and every manifest it finds, with the weights status;
`larmor models verify [ID]` recomputes the weights SHA-256.

The embedding cache key includes an encoder key: the weights SHA-256 and every `[architecture]` and
`[readout]` value, so replacing the weights, or correcting a value such as `heads` or `norm_eps`, never
mixes old and new embeddings, and a probe refuses embeddings from another encoder than the one it was
trained on.

A manifest's `id` is letters, digits, `.`, `_` and `-` (starting with a letter or digit, at most 64): it
names cache folders and output files, so write `org--name` rather than `org/name`. `[weights] file` is a
plain file name: the weights always sit next to the manifest.

## nanobrain checkpoints

nanobrain's trained checkpoints are PyTorch `.pth` files (`{"model_class", "args", "model", ...}`) that
live with the training runs (the cluster). Export the encoder half to Larmor's format with
`tools/make_models.py` from the PyTorch reference environment:

```sh
cd tools
uv sync --frozen --inexact                       # once; or use an existing tools/.venv
.venv/bin/python make_models.py export /path/to/vits_run2_epoch40.pth \
    --id nanobrain-vits-p8 --out ../models \
    --description "nanobrain ViT-S MAE encoder, 8 mm patches, run 2 epoch 40"
```

This writes `../models/nanobrain-vits-p8.toml` and `../models/nanobrain-vits-p8.safetensors`: the
encoder's parameters in float32 with nanobrain's tensor names (the MAE decoder and mask token are left
out), the `[architecture]` read from the checkpoint's `args`, the `nanobrain-eval` preprocessing recipe,
and the weights SHA-256. Then:

```sh
larmor models verify nanobrain-vits-p8
larmor encode /data/bids --model nanobrain-vits-p8
```

To share a model, upload the `.safetensors` somewhere and put its address in the manifest's
`[weights] url`; others copy the manifest into one of the folders above and run
`larmor models download nanobrain-vits-p8`, which fetches the file next to the manifest and refuses it
unless its SHA-256 matches. The URL must be `https://` (`--allow-http` for a local server), and an existing
file with another SHA-256 is only replaced with `--force`. For Hugging Face URLs
(`https://huggingface.co/...`, `https://hf.co/...`), `HF_TOKEN`, when set, is sent as a bearer token so
private repositories work; it is never sent to any other host (the host is parsed, so
`https://huggingface.co@other.host/` does not count, and such URLs are refused).

## Reference models

`tools/make_models.py reference` writes `nanobrain-vits-p8-ref` and `nanobrain-vits-p4-ref` into
`fixtures/models/`: nanobrain's ViT-S with a seeded random init and every bias, norm, position table and
class token perturbed. They exist to check parity with PyTorch (`larmor check-parity fixtures`), not to
embed scans; their embeddings carry no anatomy.
