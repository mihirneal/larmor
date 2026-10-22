# Larmor

MRI foundation-model embeddings and linear probes on a Mac, in one Rust program.

Larmor turns an MRI scan (a NIfTI file or a DICOM series) into embeddings from a 3D ViT foundation model
and fits and applies linear probes on them, in the style of [Onset](https://github.com/mihirneal/onset). The forward pass is the easy part.
The hard part is matching the training preprocessing exactly, so every piece is checked against the
PyTorch code it mirrors ([MedARC-AI/nanobrain](https://github.com/MedARC-AI/nanobrain) at `a1bfdd1`).

- **One program, nothing else to install.** Preprocessing, encoding and probes are all Rust, with no
  Python at run time.
- **Model-agnostic.** A model is a `model.toml` manifest (architecture, preprocessing recipe, weights and
  their SHA-256), not code. nanobrain's ViT is the first one.
- **Encode once.** Embeddings are cached by scan hash, preprocessing version, encoder (weights and every
  architecture value) and precision, so a new probe trains on the cache and never re-runs the encoder.
- **Same numbers as the eval framework.** Probes reproduce nanobrain's eval: the same CV folds, solvers,
  metrics and bootstrap resamples.
- **DICOM in, as dcm2niix would.** A DICOM series is converted to exactly the NIfTI image dcm2niix writes
  for it, bit for bit, so it embeds exactly as that file would. `larmor watch` encodes scans as they
  arrive in a folder (a Box Drive folder, say).

## Status

Work in progress (phases 0–3 of [docs/DESIGN.md](docs/DESIGN.md)).

| Piece | Crate | State |
|---|---|---|
| NIfTI I/O, resampling, filters, morphology | `larmor-volume` | Done |
| DICOM series to dcm2niix's NIfTI | `larmor-dicom` | Done for classic DICOM (one slice per file), uncompressed, RLE, JPEG and JPEG 2000: bit-identical to dcm2niix v1.0.20260724 on 12 Siemens, GE, Philips and UIH series, 2-D and 3-D, in four patient positions |
| nanobrain eval preprocessing recipe | `larmor-prep` | Done: bit-identical to nibabel/scipy/numpy at every stage on 25 fixture scans, about 9× faster than the Python pipeline |
| 3D ViT encoder | `larmor-model` | Done, on Candle (Metal or the CPU): parity passes against PyTorch on all 29 references in f32, f16 and bf16 (f32 min token cosine > 0.99999); ViT-S patch 8 encodes in about 0.15 s on an M3 Ultra |
| Ridge, logistic and segmentation probes | `larmor-probe` | Matches nanobrain's eval framework (folds, chosen hyperparameters, metrics, bootstrap CIs) |
| `larmor` command line | `larmor` | Done for phases 0–3: encode (NIfTI files and DICOM series, with the embedding cache, and multi-modal input on a reference scan's head mask), probe fit/apply, watch (a site's folders, encoding scans as they settle), models, check-parity and bench ([docs/CLI.md](docs/CLI.md)); every parity gate passes end to end on the 36 fixture scans and 15 DICOM sets |

Not yet: enhanced (multi-frame) DICOM, JPEG-LS compressed DICOM, and the viewer.

## Build

Larmor runs on Apple silicon Macs, on the GPU through Metal (or the CPU where Metal is missing).

```bash
# install Rust once: https://rustup.rs
cargo build --release
cargo test
```

The parity tests that need reference data are `#[ignore]`d by default. To build the 36-scan fixture set
and its PyTorch references (public OpenNeuro scans, TemplateFlow templates, synthetic edge cases, and DICOM
series from dcm2niix's validation sets with dcm2niix's NIfTI of each), see [tools/README.md](tools/README.md),
then run `cargo test --release -- --ignored`.

```bash
larmor encode /data/exam --model nanobrain-vits-p8          # NIfTI files and DICOM series below /data/exam
larmor watch --site examples/site.toml                      # edit the paths first
```

## License

MIT OR Apache-2.0
