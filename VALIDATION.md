# E19 release candidate 1 — validation, November 9, 2026

## Passed

- Source-only release assembled using an explicit allowlist.
- Fresh ffsplat checkout at the pinned revision, with the captured metadata
  overlay and `uv sync --frozen --offline --no-dev --link-mode symlink`.
  107 packages installed into a new environment from the existing cache;
  the original working environment was not changed.
- Rust adapter built from the release copy with `cargo build --locked --release`.
- `e19.py doctor`: PyTorch CUDA tensor operation, CuPy GPU operation, SOG profile
  import, and required TMC2/HM binary/config presence checks passed.
- Four launcher configuration tests, eight SH/residual/geometry tests and five
  Rust adapter tests passed (17 total).
- Actual encode of a deterministic synthetic 16,384-Gaussian SH-degree-3 PLY
  using compact SH settings, V-PCC 8+8 and residual bitplanes.
- Independent decode succeeded; its complete PLY SHA-256 matched encode's
  validation decode: `ad94059db4c5a22d321d7c36fb881611e7a1ad5500483fdc9ba28c51c238b779`.
- Output records: 16,384. Encoded payload: 307,774 bytes. This synthetic size
  is a smoke result, not a representative compression or quality benchmark.

Host: WSL Ubuntu, RTX 5070 (compute capability 12.0), driver 610.88,
Python 3.12.3, Torch 2.11.0+cu130, CuPy 14.2.0, Rust 1.92.0.
Pinned existing TMC2/HM binaries were reused; this validation did not rebuild
TMC2/HM on a fresh OS. Known setup fixes include preserving venv interpreter
symlinks, portable dependency roots, explicit adapter selection, and decode
overwrite protection.

## Scope limits

The new environment was installed from a local cache rather than a clean
network download. The fresh-install branch of setup.py and its TMC2 build
invocation have not been exercised end-to-end on a new machine. No Docker
image or non-NVIDIA backend is provided. No rendering/training extensions
are required or installed by the codec smoke test. CI performs CPU checks
only; its first hosted run is pending.

The October result CSVs describe the archived study and were not regenerated
by this release smoke test. Optional compaction training and benchmarking
application integration are outside the first candidate. Full transitive
license review and testing on another machine remain recommended before
claiming general installation support.
