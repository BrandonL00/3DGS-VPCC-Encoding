# https://brandonl00.github.io/3DGS-VPCC-Encoding

# E19 — 3D Gaussian scene compression with V-PCC

E19 compresses an existing INRIA-style 3D Gaussian Splatting PLY into a
directory package and decodes it back into a renderer-ready PLY. It combines
genuine MPEG TMC2 V-PCC geometry coding with ffsplat SOG attribute coding.
Encoding an already trained scene requires no scene training.

**Release candidate 1.** Supported validation target: Ubuntu 24.04 / WSL2,
NVIDIA CUDA GPU, Python 3.12, Rust 1.92.0. Read `VALIDATION.md` for the actual
checks performed; a successful run on one GPU does not establish all-GPU support.

## Install

On Windows, run these commands **inside WSL2 Ubuntu**. Install a compatible
NVIDIA host driver first. This setup does not install or replace GPU drivers.
For WSL, use the Windows NVIDIA driver; see NVIDIA's WSL guide.

Install Git, a C++ toolchain, CMake, Ninja and Make. On Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y git build-essential cmake ninja-build pkg-config
```

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and
[Rust](https://rustup.rs/), then run from this release's root:

```bash
python3 setup.py
python3 e19.py doctor
```

Setup clones pinned ffsplat/TMC2 sources, applies the included dependency
overlay and TMC2 build patches, installs the frozen Python environment, builds
the adapter, and tests both PyTorch and CuPy on the GPU. This downloads large
CUDA/PyTorch packages and compiles TMC2/HM; the first installation takes time.
Install on the Linux filesystem for faster builds in WSL.

This candidate uses the tested **Torch 2.11 / CUDA 13 / CuPy 14** profile.
It checks the GPU rather than guessing arbitrary CUDA versions. An incompatible
driver or architecture must be resolved before encoding. No standalone CUDA
compiler is required by the documented E19 path; optional rendering/training
extensions have separate requirements.

Dependency destinations default to `~/.local/share/e19/ffsplat` and
`~/tools/mpeg-pcc-tmc2`. Setup refuses to alter existing dependency directories.
To reuse a working environment without downloading or reinstalling it:

```bash
python3 setup.py --reuse-existing \
  --ffsplat-root /path/to/ffsplat \
  --tmc2-root /path/to/mpeg-pcc-tmc2
```

The local configuration `e19.local.json` is excluded from Git. Environment
overrides: `FFSPLAT_ROOT`, `TMC2_ROOT`, `E19_PYTHON`, `E19_ADAPTER_BINARY`.

## Encode and decode

```bash
python3 e19.py encode /data/input.ply outputs/example --preset compact
python3 e19.py decode outputs/example outputs/decoded.ply
```

Use fresh output paths. Encode performs an immediate validation decode and
writes `restored_0000.ply`; the separate decode demonstrates that the source
PLY is unnecessary for restoration. Dependencies remain necessary.

- `compact` is the launcher default: adaptive KISS-inspired SH codebook, 100
  centroid levels. It matches the policy used for our combined experiments.
- `quality` uses up to 65,536 SH entries and 32 centroid levels; these are
  historical preset names, not a guaranteed quality ranking on every scene.
- For our explicit preservation Maximum configuration, use
  `--sh-clusters 65536 --sh-levels 100` with an input large enough for that grid.
- Geometry stays at 8+8 coarse/fine coordinates, Morton ordering and residual
  bitplanes. This is lossy scene compression, not preservation of float32 data.
- Inputs are binary little-endian INRIA 3DGS PLYs with SH degree 3 and the
  standard property layout. The small SOG square-grid trim remains; tiny scenes
  can be too small for PLAS. Test with at least 16,384 records.
- Training/compaction and the separate benchmark application are not required
  for encoding. You may encode externally compacted PLYs and label their
  lineage with `--compaction-version 18.3`; that flag does not run compaction.

## Smoke test and tests

```bash
python3 examples/make_fixture.py outputs/fixture.ply
python3 e19.py encode outputs/fixture.ply outputs/smoke --preset compact
python3 e19.py decode outputs/smoke outputs/smoke-decoded.ply
# Run these with the configured ffsplat Python:
/path/to/ffsplat/.venv/bin/python -m unittest discover \
  -s pipeline/tmc2_adapter/scripts -p 'test_*.py'
cargo test --locked -p tmc2_adapter
```

The fixture is generated synthetic data; no downloaded scene is redistributed.
It checks the codec path, not visual quality. Record counts and SHA-256 of
immediate versus independent decoding are checked in release validation.

## Format and research results

See [format documentation](docs/FORMAT.md) and [results guide](results/README.md).
`manifest.json` joins geometry streams, associations, residuals and SOG semantics.
This directory package is a hybrid format; the entire package is not itself a
single standard V-PCC stream. The internal V-PCC stream is produced by TMC2.
Preservation settings and C18.3 budgets are documented in the supplied CSVs.

No dataset, trained scene, benchmark database, API secret, or local environment
is included. Full workstation runners and the separate `3dgsCompare` service
are outside this initial codec release; benchmark installation and C18.3
training integration are deferred. Older Rust atlas binaries remain in the
workspace solely to preserve the existing locked build graph.

Our code is MIT licensed. Dependencies retain their own licenses; see
[THIRD_PARTY.md](THIRD_PARTY.md). Optional Discord messages are disabled by the
release launcher unless you explicitly set `VPCC_DISCORD_DISABLED=0` and
configure your webhook.
