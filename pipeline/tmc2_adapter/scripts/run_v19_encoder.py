#!/usr/bin/env python3
"""V19 encoder: Morton/V-PCC geometry plus modular SH codebook storage.

The encoder treats the local ffsplat SOG-web implementation as an
external dependency.  It produces a fresh Morton SOG semantic container, a
fresh signed-log V-PCC geometry stream, then a self-decodable hybrid archive.

Run through ffsplat's environment and pass dependency roots explicitly, or use
the portable release's e19.py launcher. See tools/release/README.release.md.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from package_geometry_probe import convert_residual_bitplanes


REPOSITORY = Path(__file__).resolve().parents[3]
SCRIPTS = REPOSITORY / "pipeline" / "tmc2_adapter" / "scripts"
DUAL_BENCHMARK = REPOSITORY / "evaluation" / "benchmark_3dgscompare.py"


def run(command: list[str], cwd: Path) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    # Keep existing research installations working without embedding a username.
    installed = Path.home() / ".local/share/e19/ffsplat"
    legacy = Path.home() / "ffsplat"
    ffsplat_default = installed if installed.is_dir() or not legacy.is_dir() else legacy
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="input INRIA 3DGS PLY")
    parser.add_argument("output", type=Path, help="new V19 package directory")
    parser.add_argument("--preset", choices=["quality", "compact"], default="quality",
                        help="quality: existing K/32 levels; compact: KISS-sized K/100 levels. Both use SH PLAS/tiling.")
    parser.add_argument("--compaction-version", default="untracked",
                        help="Independent source compaction version, e.g. 18.3, or none for a raw source.")
    parser.add_argument("--compaction-manifest", type=Path)
    parser.add_argument("--ffsplat-root", type=Path,
                        default=Path(os.environ.get("FFSPLAT_ROOT", str(ffsplat_default))))
    parser.add_argument("--tmc2-root", type=Path,
                        default=Path(os.environ.get("TMC2_ROOT", str(Path.home() / "tools/mpeg-pcc-tmc2"))))
    parser.add_argument("--adapter-binary", type=Path,
                        help="Use a prebuilt tmc2_adapter instead of cargo run.")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--geometry-high-bits", type=int, choices=range(6, 13), default=8,
                        help="V-PCC coarse width; defaults to 8+8, total precision remains 16 bits.")
    parser.add_argument("--residual-layout", choices=["packed", "bitplanes"], default="bitplanes",
                        help="Reversible low-coordinate storage; packed reproduces the earlier layout.")
    parser.add_argument("--vpcc-atlas-width", type=int, default=4096)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sog-control-output", type=Path,
                        help="Retain the matched standalone SOG control without repeating SH encoding.")
    parser.add_argument("--sh-policy", choices=["legacy", "kiss"], help="Optional override of the preset's codebook policy.")
    parser.add_argument("--sh-clusters", type=int)
    parser.add_argument("--sh-levels", type=int,
                        help="Centroid precision; defaults to 32 for legacy or 100 for kiss.")
    parser.add_argument("--sh-layout", choices=["none", "lexicographic", "plas"], default="plas")
    parser.add_argument("--sh-tiled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--benchmark-backend", type=Path,
                        help="3dgsCompare backend directory; enables the dual-quality benchmark when paired with --photo-reference-frame-set-id.")
    parser.add_argument("--benchmark-python", type=Path,
                        default=Path("/mnt/d/3dgsCompare/backend/.venv/Scripts/python.exe"))
    parser.add_argument("--benchmark-camera-bundle-id")
    parser.add_argument("--benchmark-original-scene-id")
    parser.add_argument("--photo-reference-frame-set-id")
    args = parser.parse_args()
    if args.sh_policy is None:
        args.sh_policy = "kiss" if args.preset == "compact" else "legacy"
    if args.sh_levels is None:
        args.sh_levels = 100 if args.sh_policy == "kiss" else 32
    benchmark_args = (args.benchmark_backend, args.benchmark_camera_bundle_id,
                      args.benchmark_original_scene_id, args.photo_reference_frame_set_id)
    if any(value is not None for value in benchmark_args) and not all(value is not None for value in benchmark_args):
        parser.error("dual benchmark requires --benchmark-backend, --benchmark-camera-bundle-id, "
                     "--benchmark-original-scene-id, and --photo-reference-frame-set-id together")
    source = args.source.resolve()
    output = args.output.resolve()
    args.ffsplat_root = args.ffsplat_root.expanduser().resolve()
    args.tmc2_root = args.tmc2_root.expanduser().resolve()
    if args.sog_control_output:
        args.sog_control_output = args.sog_control_output.resolve()
    if not source.is_file():
        raise SystemExit(f"input PLY not found: {source}")
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output}")
    if args.sog_control_output and args.sog_control_output.exists():
        raise SystemExit(f"refusing to overwrite existing SOG control: {args.sog_control_output}")
    if not (args.ffsplat_root / "pyproject.toml").is_file():
        raise SystemExit(f"ffsplat checkout not found: {args.ffsplat_root}")
    if not (args.tmc2_root / "bin/PccAppEncoder").is_file():
        raise SystemExit(f"TMC2 encoder not found: {args.tmc2_root}")

    if args.compaction_manifest and not args.compaction_manifest.is_file():
        raise SystemExit(f"compaction manifest not found: {args.compaction_manifest}")

    with tempfile.TemporaryDirectory(prefix="v19-encoder-") as temporary:
        work = Path(temporary)
        sog_root = work / "sog"
        sog_root.mkdir()
        sog_dir = sog_root / "sog_web_morton"
        order = work / "morton_source_indices.npy"
        geometry = work / "vpcc_geometry"

        # ffsplat owns SOG semantics and the canonical full-16-bit Morton
        # ordering.  The saved permutation keeps our geometry records aligned
        # with those semantic image planes.
        sh_options = ["--sh-policy", args.sh_policy, "--sh-levels", str(args.sh_levels),
                      "--sh-layout", args.sh_layout]
        if args.sh_clusters is not None:
            sh_options.extend(["--sh-clusters", str(args.sh_clusters)])
        if args.sh_tiled:
            sh_options.append("--sh-tiled")
        run([
            sys.executable, str(SCRIPTS / "encode_sog_morton.py"), str(source), str(sog_dir),
            "--order-output", str(order), "--device", args.device,
            *sh_options,
        ], args.ffsplat_root)

        # Our adapter owns signed-log geometry and genuine V-PCC coding.  No
        # generic semantic sidecar is requested: ffsplat has already created it.
        adapter_command = ([str(args.adapter_binary.resolve())] if args.adapter_binary else
                           [os.environ.get("E19_CARGO", "cargo"), "run", "--release", "-p", "tmc2_adapter", "--"])
        run([
            *adapter_command, str(source), str(geometry),
            "--tmc2-root", str(args.tmc2_root), "--geometry-bitdepth", str(args.geometry_high_bits),
            "--high-low-geometry", "--signed-log-geometry",
            "--vpcc-disable-patch-splitting", "--vpcc-atlas-width", str(args.vpcc_atlas_width),
            "--threads", str(args.threads),
        ], REPOSITORY)

        # Build the portable package and immediately prove that it decodes with
        # package files alone.
        run([
            sys.executable, str(SCRIPTS / "build_v17_hybrid.py"), str(geometry), str(sog_root), str(output),
            "--order", str(order), "--vpcc-geometry-bitdepth", str(args.geometry_high_bits),
        ], REPOSITORY)
        if args.residual_layout == "bitplanes":
            convert_residual_bitplanes(output, args.geometry_high_bits)
        run([
            sys.executable, str(SCRIPTS / "decode_v18_hybrid.py"), str(output), str(output / "restored_0000.ply"),
            "--tmc2-root", str(args.tmc2_root), "--threads", str(args.threads),
        ], args.ffsplat_root)
        if args.sog_control_output:
            shutil.copytree(sog_dir, args.sog_control_output)

    manifest = json.loads((output / "manifest.json").read_text())
    expected_policy = "kiss" if args.preset == "compact" else "legacy"
    expected_levels = 100 if args.preset == "compact" else 32
    preset_matches = (args.sh_policy == expected_policy and args.sh_levels == expected_levels
                      and args.sh_clusters is None and args.sh_layout == "plas" and args.sh_tiled)
    versions = {"encoder_version": "19", "encoder_preset": args.preset if preset_matches else "custom",
                "requested_preset": args.preset,
                "compaction_version": args.compaction_version,
                "compaction_manifest": str(args.compaction_manifest.resolve()) if args.compaction_manifest else None}
    manifest.update(versions)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = {
        "mode": "v19_encoder_source_to_package_to_decode",
        **versions,
        "source_ply": str(source),
        "ffsplat_root": str(args.ffsplat_root),
        "tmc2_root": str(args.tmc2_root),
        "vpcc_geometry_bitdepth": args.geometry_high_bits,
        "low_coordinate_bits": 16 - args.geometry_high_bits,
        "residual_layout": args.residual_layout,
        "vpcc_atlas_width": args.vpcc_atlas_width,
        "payload_bytes": manifest["payload_bytes"],
        "restored_ply": "restored_0000.ply",
        "sh_encoding": json.loads((output / "semantic_sog" / "sh_encoding_report.json").read_text()),
    }
    (output / "v19_encoder_run.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.photo_reference_frame_set_id:
        run([
            # WSL's Windows-process bridge treats an absolute /mnt/c script
            # path as C:\\mnt\\c\\..., while a repository-relative path is
            # resolved correctly from the inherited working directory.
            str(args.benchmark_python), str(DUAL_BENCHMARK.relative_to(REPOSITORY)),
            "--candidate", str(output / "restored_0000.ply"),
            "--output", str(output / "benchmark_3dgscompare_dual.json"),
            "--backend", str(args.benchmark_backend),
            "--name", f"{output.name} dual quality",
            "--original-scene-id", args.benchmark_original_scene_id,
            "--camera-bundle-id", args.benchmark_camera_bundle_id,
            "--photo-reference-frame-set-id", args.photo_reference_frame_set_id,
        ], REPOSITORY)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
