# Third-party components

The root MIT license covers this project's own code. It does not replace
dependency licenses or license the separately downloaded datasets/models.
Source dependencies are installed separately, not vendored into this release.

| Dependency | Pinned source / record | License notice |
|---|---|---|
| ffsplat | `3fb6444c5faed6de9aa55fa2313df6c2a4104193`, https://github.com/w-m/ffsplat | Apache-2.0 text in `dependencies/ffsplat/LICENSE` |
| MPEG TMC2 | `d47e6ce879206b8a7881ecabce9e1ad2e8254a38`, https://github.com/MPEGGroup/mpeg-pcc-tmc2 | Upstream notice copied to `dependencies/TMC2-COPYING`; build script applies local integration patches |
| HM application codec | `cae35dba3dcf8e1b6121a70b397000af36301045` | Downloaded/build dependency; retain its upstream license and source notices |
| PLAS, torchpq, CuPy, PyTorch and other Python packages | Git commits, wheel sources and versions in `dependencies/ffsplat/uv.lock` | Each dependency retains its own package license |
| Rust dependencies | Versions and checksums in `Cargo.lock` | Each dependency retains its own crate license |

The ffsplat overlay changes dependency metadata from upstream: Torch 2.6 to
2.11, CuPy CUDA 12 to CUDA 13, and the named PyTorch index. No ffsplat source
algorithm changes are applied. Both the modified metadata and its frozen lock
are supplied. The full development lock is retained, but setup excludes the
development group.

The E19 semantic transforms compose with ffsplat's extension interfaces. The
package uses TMC2 for its geometry stream and additional project-specific
streams for the remaining Gaussian data. MIT licensing of our orchestration
does not grant rights to any external standard, dependency or dataset.

This is a dependency inventory, not a completed automated license audit of
all transitive dependencies. Public-release review should retain upstream
notices and resolve any intended redistribution of binaries separately.
