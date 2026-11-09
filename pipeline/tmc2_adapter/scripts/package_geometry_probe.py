"""Package a measured geometry split with fixed SOG semantics and audit PLY equality."""
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
PROJECT = SCRIPTS.parents[2]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def convert_residual_bitplanes(package, high_bits):
    """Reversibly replace a newly built experiment's packed residual file."""
    import numpy as np
    import zlib
    from decode_v18_hybrid import unpack_low_coordinates
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    packed_path = package / manifest["low_bits"]
    values = unpack_low_coordinates(packed_path, manifest["records"], 16 - high_bits)
    paths = []
    before = packed_path.stat().st_size
    after = 0
    for bit in range(16 - high_bits):
        name = f"low_plane_{bit}.deflate"
        blob = zlib.compress(np.packbits(((values.reshape(-1) >> bit) & 1).astype(np.uint8),
                                        bitorder="little").tobytes(), 9)
        (package / name).write_bytes(blob)
        paths.append(name)
        after += len(blob)
    packed_path.unlink()  # Only the just-generated, reproducible experiment stream.
    manifest.pop("low_bits")
    manifest.update(low_bits_layout="bitplanes", low_bitplanes=paths,
                    payload_bytes=manifest["payload_bytes"] - before + after)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--sog", type=Path, required=True)
    parser.add_argument("--order", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--high-bits", type=int, required=True)
    parser.add_argument("--residual-bitplanes", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    with tempfile.TemporaryDirectory(prefix="e19-geometry-package-") as name:
        root = Path(name)
        (root / "sog_web_morton").symlink_to(args.sog.resolve(), target_is_directory=True)
        subprocess.run([sys.executable, str(SCRIPTS / "build_v17_hybrid.py"), str(args.geometry),
            str(root), str(args.output), "--order", str(args.order),
            "--vpcc-geometry-bitdepth", str(args.high_bits)], cwd=PROJECT, check=True)
    if args.residual_bitplanes:
        convert_residual_bitplanes(args.output, args.high_bits)
    restored = args.output / "restored_0000.ply"
    subprocess.run([sys.executable, str(SCRIPTS / "decode_v18_hybrid.py"), str(args.output), str(restored)], check=True)
    identical = digest(restored) == digest(args.baseline / "restored_0000.ply")
    report = {"encoder_version": "19", "compaction_version": "18.3",
              "geometry_high_bits": args.high_bits, "low_coordinate_bits": 16 - args.high_bits,
              "baseline": str(args.baseline), "identical_restored_ply": identical,
              "restored_sha256": digest(restored)}
    (args.output / "geometry_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    manifest = json.loads((args.output / "manifest.json").read_text())
    manifest.update(encoder_version="19", compaction_version="18.3", encoder_preset="compact",
                    geometry_experiment=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({**report, "payload_bytes": manifest["payload_bytes"]}, indent=2))
    if not identical:
        raise SystemExit("Package differs from baseline: render quality must be remeasured before accepting it")


if __name__ == "__main__":
    main()
