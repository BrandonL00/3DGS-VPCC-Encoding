#!/usr/bin/env python3
"""Decode a SOG hybrid without the original source PLY.

The container holds SOG semantics without its two means images.  Geometry is
instead recovered from TMC2's V-PCC anchor stream, the Morton association
stream, and packed signed-log low coordinates. Experimental E19 packages
can replace V-PCC with a self-contained lossless anchor or position stream.

Run through ffsplat's environment with --tmc2-root, or use the portable
release's e19.py decode command.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import zlib
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from ffsplat.coding.scene_decoder import decode_gaussians
from lossless_geometry import decode_positions


def read_ascii_anchor_ply(path: Path) -> np.ndarray:
    """Read the x/y/z high-coordinate PLY emitted by TMC2's decoder."""
    with path.open("rb") as handle:
        lines = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"truncated PLY header: {path}")
            lines.append(line)
            if line == b"end_header\n":
                break
    header = b"".join(lines).decode("ascii")
    if "format ascii 1.0" not in header:
        raise ValueError("TMC2 decoded anchors must be ascii PLY")
    count = next(int(line.split()[2]) for line in header.splitlines() if line.startswith("element vertex "))
    anchors = np.loadtxt(path, skiprows=len(lines), max_rows=count, usecols=(0, 1, 2), dtype=np.uint16)
    if anchors.shape != (count, 3):
        raise ValueError(f"expected {count} decoded anchors, got {anchors.shape}")
    return anchors


def morton(points: np.ndarray, bits: int) -> np.ndarray:
    out = np.zeros(len(points), dtype=np.uint64)
    for bit in range(bits):
        for axis in range(3):
            out |= ((points[:, axis] >> bit) & 1).astype(np.uint64) << (3 * bit + axis)
    return out


def unpack_association(path: Path, count: int, anchor_count: int) -> np.ndarray:
    packed = np.frombuffer(zlib.decompress(path.read_bytes()), dtype="<u4")
    if len(packed) != count:
        raise ValueError(f"association records={len(packed)}, expected={count}")
    deltas = (packed >> 1).astype(np.int64) ^ -((packed & 1).astype(np.int64))
    ids = np.cumsum(deltas, dtype=np.int64)
    if ids.min(initial=0) < 0 or ids.max(initial=0) >= anchor_count:
        raise ValueError("association contains an out-of-range anchor id")
    return ids


def unpack_low_coordinates(path: Path, count: int, bits: int) -> np.ndarray:
    raw = zlib.decompress(path.read_bytes())
    required_bits = count * 3 * bits
    if len(raw) * 8 < required_bits:
        raise ValueError("low-coordinate stream ended early")
    values = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little")[:required_bits]
    weights = (1 << np.arange(bits, dtype=np.uint16))
    return (values.reshape(-1, bits).astype(np.uint16) * weights).sum(axis=1, dtype=np.uint16).reshape(count, 3)


def dequantize(quantized: np.ndarray, transform: dict) -> np.ndarray:
    domain = quantized.astype(np.float64) / float(transform["scale_to_quantized"])
    if transform.get("signed_log_coordinates", False):
        domain += float(transform["coordinate_domain_min"])
        return (np.sign(domain) * np.expm1(np.abs(domain))).astype(np.float32)
    return (domain + np.asarray(transform["source_min"], dtype=np.float64)).astype(np.float32)


def unpack_bitplanes(package, paths, count, bits):
    if len(paths) != bits:
        raise ValueError("residual bitplane count mismatch")
    low = np.zeros(count * 3, dtype=np.uint16)
    for bit, path in enumerate(paths):
        raw = zlib.decompress((package / path).read_bytes())
        if len(raw) != (count * 3 + 7) // 8:
            raise ValueError("residual bitplane size mismatch")
        plane = np.unpackbits(np.frombuffer(raw, np.uint8), bitorder="little")[:count * 3]
        low |= plane.astype(np.uint16) << bit
    return low.reshape(count, 3)


def decode_vpcc(package: Path, destination: Path, tmc2_root: Path, threads: int) -> np.ndarray:
    executable = tmc2_root / "bin/PccAppDecoder"
    hm_decoder = tmc2_root / "dependencies/HM-16.20+SCM-8.8-app/bin/TAppDecoderStatic"
    log = destination.with_suffix(".log")
    command = [
        str(executable),
        f"--compressedStreamPath={package / 'stream.bin'}",
        f"--inverseColorSpaceConversionConfig={tmc2_root / 'cfg/hdrconvert/yuv420toyuv444_16bit.cfg'}",
        "--startFrameNumber=0",
        "--frameCount=1",
        f"--nbThread={threads}",
        f"--reconstructedDataPath={destination.with_name('anchors_%04d.ply')}",
        "--computeChecksum=0",
        f"--videoDecoderOccupancyPath={hm_decoder}",
        f"--videoDecoderGeometryPath={hm_decoder}",
        f"--videoDecoderAttributePath={hm_decoder}",
    ]
    with log.open("w", encoding="utf-8") as handle:
        subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT)
    return read_ascii_anchor_ply(destination.with_name("anchors_0000.ply"))


def semantic_ply(package: Path, temp: Path, output: Path) -> None:
    """Decode SOG semantic planes after supplying disposable placeholder means.

    ffsplat needs means images to instantiate a SOG scene.  Both placeholders
    have the right grid shape and are overwritten by package geometry before
    the final PLY is written.
    """
    source = package / "semantic_sog"
    staging = temp / "semantic_sog"
    shutil.copytree(source, staging)
    placeholder = staging / "f_dc.avif"
    for name in ("means_bytes_0.avif", "means_bytes_1.avif"):
        shutil.copy2(placeholder, staging / name)
    gaussians = decode_gaussians(input_path=staging, input_format="smurfx", verbose=False)
    means = gaussians.means.data.detach().cpu().numpy()
    scales = np.log(np.clip(gaussians.scales.data.detach().cpu().numpy(), 1e-7, None))
    opacity = np.squeeze(gaussians.opacities.data.detach().cpu().numpy())
    opacity = np.clip(opacity, 1e-7, 1.0 - 1e-7)
    opacity = np.log(opacity / (1.0 - opacity))
    rotations = gaussians.quaternions.data.detach().cpu().numpy()
    sh = gaussians.sh.data.detach().cpu().numpy()
    dc = sh[:, 0, :]
    rest = np.transpose(sh[:, 1:, :], (0, 2, 1)).reshape(len(sh), -1)
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("nx", "f4"), ("ny", "f4"), ("nz", "f4")]
    dtype += [(f"f_dc_{index}", "f4") for index in range(3)]
    dtype += [(f"f_rest_{index}", "f4") for index in range(rest.shape[1])]
    dtype += [("opacity", "f4"), ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"), ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4")]
    vertex = np.empty(len(sh), dtype=dtype)
    vertex["x"], vertex["y"], vertex["z"] = means[:, 0], means[:, 1], means[:, 2]
    vertex["nx"], vertex["ny"], vertex["nz"] = 0.0, 0.0, 0.0
    for index in range(3): vertex[f"f_dc_{index}"] = dc[:, index]
    for index in range(rest.shape[1]): vertex[f"f_rest_{index}"] = rest[:, index]
    vertex["opacity"] = opacity
    for index in range(3): vertex[f"scale_{index}"] = scales[:, index]
    for index in range(4): vertex[f"rot_{index}"] = rotations[:, index]
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(str(output))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tmc2-root", type=Path,
                        default=Path(os.environ.get("TMC2_ROOT", str(Path.home() / "tools/mpeg-pcc-tmc2"))))
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    package = args.package.resolve()
    args.output = args.output.resolve()
    args.tmc2_root = args.tmc2_root.expanduser().resolve()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((package / "manifest.json").read_text())
    if manifest.get("mode") not in {"v18_vpcc_signed_log_morton_sog_hybrid", "e19_modular_geometry_sog_hybrid"}:
        raise SystemExit(f"unsupported package mode: {manifest.get('mode')}")
    high_bits = int(manifest.get("geometry_anchor_bits", manifest.get("vpcc_geometry_bitdepth", 16)))
    low_bits = int(manifest.get("low_coordinate_bits", 0))
    if high_bits + low_bits != 16:
        raise SystemExit("V18 geometry must total 16 bits")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="v18-hybrid-decode-") as temporary:
        temporary_path = Path(temporary)
        backend = manifest.get("geometry_backend", "vpcc")
        if backend == "lossless_positions":
            quantized = decode_positions((package / manifest["position_stream"]).read_bytes())
        elif backend in {"vpcc", "vpcc_lossy_nearest", "lossless_anchors"}:
            quantized = decode_anchor_geometry(package, manifest, temporary_path, args.tmc2_root, args.threads)
        else:
            raise SystemExit(f"unsupported geometry backend: {backend}")
        if quantized.shape != (int(manifest["records"]), 3):
            raise SystemExit("geometry record count mismatch")
        geometry = dequantize(quantized, manifest["transform"])
        semantic = temporary_path / "semantic.ply"
        semantic_ply(package, temporary_path, semantic)
        ply = PlyData.read(str(semantic))
        vertex = ply["vertex"].data
        if len(vertex) != len(geometry):
            raise SystemExit(f"semantic records={len(vertex)}, geometry records={len(geometry)}")
        vertex["x"], vertex["y"], vertex["z"] = geometry[:, 0], geometry[:, 1], geometry[:, 2]
        PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(str(args.output))
    print(args.output)


def decode_anchor_geometry(package, manifest, temporary_path, tmc2_root, threads):
    high_bits = int(manifest.get("geometry_anchor_bits", manifest.get("vpcc_geometry_bitdepth")))
    low_bits = int(manifest["low_coordinate_bits"])
    decoded = (decode_positions((package / manifest["anchor_stream"]).read_bytes())
               if manifest.get("geometry_backend") == "lossless_anchors" else
               decode_vpcc(package, temporary_path / "decoded.ply", tmc2_root, threads))
    if manifest.get("geometry_backend") == "vpcc_lossy_nearest":
        # The encoder maps original anchors to this canonical decoded set and
        # writes final per-Gaussian associations. No source geometry is needed.
        if np.any(decoded >= (1 << high_bits)):
            raise ValueError("lossy decoded anchor exceeds the declared coarse grid")
        decoded = np.unique(decoded, axis=0)
    codes = morton(decoded, high_bits)
    order = np.argsort(codes, kind="stable")
    codes = codes[order]
    anchors = decoded[order]
    if len(np.unique(codes)) != len(codes):
        raise SystemExit("decoded anchors are not unique")
    expected = int(manifest.get("geometry_anchors", manifest.get("vpcc_anchors")))
    if len(anchors) != expected:
        raise SystemExit(f"decoded anchors={len(anchors)}, expected={expected}")
    association = unpack_association(package / manifest["association"], int(manifest["records"]), len(anchors))
    low = (unpack_bitplanes(package, manifest["low_bitplanes"], len(association), low_bits)
           if manifest.get("low_bits_layout") == "bitplanes" else
           unpack_low_coordinates(package / manifest["low_bits"], len(association), low_bits))
    return (anchors[association].astype(np.uint16) << low_bits) | low


if __name__ == "__main__":
    main()
