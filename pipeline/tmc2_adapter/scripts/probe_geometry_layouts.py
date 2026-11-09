"""Measure reversible residual layouts and geometry-only V-PCC splits.

All candidates target the same 16-bit positions in the same Gaussian order.
No quantization or rendering settings change. Outputs are isolated diagnostics.
"""
from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import lzma
import os
import subprocess
import zlib
from pathlib import Path

import numpy as np

from build_v17_hybrid import morton, pack_low_coordinates, quantize, xyz
from decode_v18_hybrid import decode_vpcc, read_ascii_anchor_ply, unpack_low_coordinates

PROJECT = Path(__file__).resolve().parents[3]


def packed(values, bits):
    values = values.reshape(-1).astype(np.uint16)
    return np.packbits(((values[:, None] >> np.arange(bits)) & 1).reshape(-1), bitorder="little").tobytes()


def unpacked(raw, shape, bits):
    binary = np.unpackbits(np.frombuffer(raw, np.uint8), bitorder="little")[:int(np.prod(shape)) * bits]
    return (binary.reshape(-1, bits) * (1 << np.arange(bits))).sum(1).astype(np.uint16).reshape(shape)


def residual_trials(low, bits):
    results = []
    raw = packed(low, bits)
    for name, encode, decode in (("deflate", lambda data: zlib.compress(data, 9), zlib.decompress),
                                 ("lzma", lambda data: lzma.compress(data, preset=6), lzma.decompress),
                                 ("bz2", bz2.compress, bz2.decompress)):
        encoded = encode(raw)
        assert np.array_equal(unpacked(decode(encoded), low.shape, bits), low)
        results.append({"layout": "packed_xyz", "codec": name, "bytes": len(encoded), "exact": True})
    candidates = {"channel_packed": low.T, "bitplanes": ((low.reshape(-1, 1) >> np.arange(bits)) & 1).T.astype(np.uint8)}
    for name, values in candidates.items():
        if name == "bitplanes":
            streams = [np.packbits(plane, bitorder="little").tobytes() for plane in values]
            restored = np.stack([np.unpackbits(np.frombuffer(zlib.decompress(zlib.compress(stream, 9)), np.uint8),
                bitorder="little")[:low.size] for stream in streams], axis=1)
            restored = (restored * (1 << np.arange(bits))).sum(1).astype(np.uint16).reshape(low.shape)
        else:
            streams = [packed(channel, bits) for channel in values]
            restored = np.stack([unpacked(zlib.decompress(zlib.compress(stream, 9)), (len(low),), bits)
                                 for stream in streams], axis=1)
        assert np.array_equal(restored, low)
        results.append({"layout": name, "codec": "deflate", "bytes": sum(len(zlib.compress(s, 9)) for s in streams), "exact": True})
    delta = ((low.astype(np.int32) - np.vstack([np.zeros((1, 3)), low[:-1]]).astype(np.int32)) & ((1 << bits) - 1)).astype(np.uint16)
    stream = zlib.compress(packed(delta, bits), 9)
    restored = np.cumsum(unpacked(zlib.decompress(stream), low.shape, bits), axis=0) & ((1 << bits) - 1)
    assert np.array_equal(restored, low)
    results.append({"layout": "modulo_delta_xyz", "codec": "deflate", "bytes": len(stream), "exact": True})
    return sorted(results, key=lambda row: row["bytes"])


def main():
    os.environ["PATH"] = "/home/email/.cargo/bin:" + os.environ.get("PATH", "")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--baseline-package", type=Path, required=True)
    parser.add_argument("--sog", type=Path, required=True)
    parser.add_argument("--order", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", type=int, nargs="+", default=[9, 8, 11, 12])
    parser.add_argument("--attempt-label", default="", help="Use a fresh geometry directory for a revised codec configuration.")
    parser.add_argument("--tmc2-root", type=Path, default=Path("/home/email/tools/mpeg-pcc-tmc2"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.baseline_package / "manifest.json").read_text())
    order = np.load(args.order).reshape(-1)
    all_q = quantize(xyz(args.source).astype(np.float64), manifest["transform"])
    q = all_q[order]
    if len(q) != manifest["records"]:
        raise ValueError("source/order do not match baseline record count")
    low = (q & 63).astype(np.uint8)
    baseline_low = unpack_low_coordinates(args.baseline_package / "low_bits.deflate", len(q), 6)
    if not np.array_equal(low, baseline_low):
        raise ValueError("source/order do not match the baseline's fine positions")
    result = {"encoder_version": "19", "compaction_version": "18.3", "source": str(args.source),
        "baseline_package": str(args.baseline_package), "records": len(q),
        "quantized_xyz_sha256": hashlib.sha256(q.tobytes()).hexdigest(),
        "matched_sog_positions_bytes": sum((args.sog / f"means_bytes_{b}.avif").stat().st_size for b in (0, 1)),
        "baseline_positions_bytes": sum((args.baseline_package / name).stat().st_size for name in ("stream.bin", "association.deflate", "low_bits.deflate")),
        "residual_trials_10plus6": residual_trials(low, 6), "splits": []}
    report_path = args.output / "geometry_probe.json"
    if report_path.is_file():
        prior = json.loads(report_path.read_text())
        if prior.get("quantized_xyz_sha256") != result["quantized_xyz_sha256"]:
            raise ValueError("refusing to mix geometry probes from different position sets")
        result["splits"] = prior.get("splits", [])

    def save():
        report_path.write_text(json.dumps(result, indent=2) + "\n")

    save()
    print(json.dumps({key: value for key, value in result.items() if key != "splits"}, indent=2), flush=True)
    for high_bits in args.splits:
        if not 6 <= high_bits <= 12:
            raise ValueError("coarse bitdepth must be 6..12")
        if any(row.get("high_bits") == high_bits and row.get("exact") for row in result["splits"]):
            continue
        result["splits"] = [row for row in result["splits"] if row.get("high_bits") != high_bits]
        suffix = f"_{args.attempt_label}" if args.attempt_label else ""
        root = args.output / f"vpcc_{high_bits}plus{16-high_bits}{suffix}"
        try:
            if not (root / "manifest.json").exists():
                command = ["cargo", "run", "--release", "-p", "tmc2_adapter", "--", str(args.source), str(root),
                    "--tmc2-root", str(args.tmc2_root), "--geometry-bitdepth", str(high_bits),
                    "--high-low-geometry", "--signed-log-geometry", "--vpcc-disable-patch-splitting",
                    "--vpcc-atlas-width", "4096", "--threads", "8"]
                print("+", " ".join(command), flush=True)
                with root.with_suffix(".log").open("a") as log:
                    subprocess.run(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, check=True)
            stream = root / "stream.bin"
            if not stream.is_file():
                raise RuntimeError(f"incomplete geometry directory: {root}")
            split_manifest = json.loads((root / "manifest.json").read_text())
            if split_manifest["transform"] != manifest["transform"]:
                # Conversion manifests can carry extra fields. The integer target
                # itself, not dictionary formatting, is the compatibility contract.
                candidate_q = quantize(xyz(args.source).astype(np.float64), split_manifest["transform"])[order]
                if not np.array_equal(candidate_q, q):
                    raise ValueError("split changed the 16-bit quantization")
            decoded_path = root / "decoded_0000.ply"
            anchors = (read_ascii_anchor_ply(decoded_path) if decoded_path.is_file() and decoded_path.stat().st_size
                       else decode_vpcc(root, root / "probe_decoded.ply", args.tmc2_root, 8))
            codes_decoded = morton(anchors, high_bits)
            anchors = anchors[np.argsort(codes_decoded, kind="stable")]
            codes_decoded = np.sort(codes_decoded)
            low_bits = 16 - high_bits
            expected_codes = np.unique(morton(all_q >> low_bits, high_bits))
            if not np.array_equal(codes_decoded, expected_codes):
                raise ValueError("V-PCC anchor set is not exact")
            ids = np.searchsorted(expected_codes, morton(q >> low_bits, high_bits))
            delta = np.diff(ids.astype(np.int64), prepend=0)
            zz = ((delta << 1) ^ (delta >> 63)).astype("<u4")
            association_bytes = len(zlib.compress(zz.tobytes(), 9))
            low = (q & ((1 << low_bits) - 1)).astype(np.uint16)
            reconstructed = (anchors[ids].astype(np.uint16) << low_bits) | low
            if not np.array_equal(reconstructed, q):
                raise ValueError("split reconstruction changed positions")
            trials = residual_trials(low, low_bits)
            raw_residual_bytes = len(zlib.compress(pack_low_coordinates(low, low_bits), 9))
            row = {"high_bits": high_bits, "low_bits": low_bits, "geometry_directory": str(root),
                   "vpcc_bytes": stream.stat().st_size,
                   "association_bytes": association_bytes, "current_residual_bytes": raw_residual_bytes,
                   "positions_bytes": stream.stat().st_size + association_bytes + raw_residual_bytes,
                   "residual_trials": trials, "anchors": len(anchors), "exact": True}
            result["splits"].append(row)
            print(json.dumps(row, indent=2), flush=True)
        except Exception as error:
            result["splits"].append({"high_bits": high_bits, "error": repr(error), "exact": False})
            print(f"FAILED {high_bits}+{16-high_bits}: {error}", flush=True)
        save()


if __name__ == "__main__":
    main()
