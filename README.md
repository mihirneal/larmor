# Larmor

MRI foundation-model embeddings and linear probes on a Mac, in one Rust program.

Larmor turns an MRI scan into embeddings from a 3D ViT foundation model and fits and applies linear probes
on them, in the style of [Onset](https://github.com/mihirneal/onset). The forward pass is the easy part.
The hard part is matching the training preprocessing exactly, so every piece is checked against the
PyTorch code it mirrors ([MedARC-AI/nanobrain](https://github.com/MedARC-AI/nanobrain) at `a1bfdd1`).

- **One program, nothing else to install.** Preprocessing, encoding and probes are all Rust, with no
  Python at run time.
- **Model-agnostic.** A model is a `model.toml` manifest (architecture, preprocessing recipe, weights and
  their SHA-256), not code. nanobrain's ViT is the first one.
- **Encode once.** Embeddings are cached by scan hash, preprocessing version and model hash, so a new
  probe trains on the cache and never re-runs the encoder.
- **Same numbers as the eval framework.** Probes reproduce nanobrain's eval: the same CV folds, solvers,
  metrics and bootstrap resamples.

## Status

Work in progress (phases 0–2 of [docs/DESIGN.md](docs/DESIGN.md)).

| Piece | Crate | State |
|---|---|---|
| NIfTI I/O, resampling, filters, morphology | `larmor-volume` | Done |
| nanobrain eval preprocessing recipe | `larmor-prep` | Done: bit-identical to nibabel/scipy/numpy at every stage on 25 fixture scans, about 9× faster than the Python pipeline |
| 3D ViT encoder | `larmor-model` | Parity passes against PyTorch (min token cosine > 0.99999). Moving from Burn to Candle, which is 2–3× faster on Metal |
| Ridge, logistic and segmentation probes | `larmor-probe` | Matches nanobrain's eval framework (folds, chosen hyperparameters, metrics, bootstrap CIs) |
| `larmor` command line | `larmor` | In progress: encode, probe fit/apply, models, check-parity, bench |

Not yet: DICOM, watching a Box Drive folder, and the viewer.

## Build

```bash
# install Rust once: https://rustup.rs
cargo build --release
cargo test
```

The parity tests that need reference data are `#[ignore]`d by default. To build the 25-scan fixture set
and its PyTorch references (public OpenNeuro scans, TemplateFlow templates and synthetic edge cases), see
[tools/README.md](tools/README.md), then run `cargo test --release -- --ignored`.

## License

MIT OR Apache-2.0
