"""Experimental lossless geometry backends with self-describing byte accounting.

These are controls for V-PCC, not production replacements. All layouts retain
integer XYZ values and record order. Morton/octree layouts require Morton order.
"""
from __future__ import annotations

import json
import lzma
import struct
import zlib

import numpy as np


def morton(xyz, bits):
    codes = np.zeros(len(xyz), np.uint64)
    for bit in range(bits):
        for axis in range(3):
            codes |= ((xyz[:, axis].astype(np.uint64) >> bit) & 1) << (3 * bit + axis)
    return codes


def unmorton(codes, bits):
    xyz = np.zeros((len(codes), 3), np.uint16)
    for bit in range(bits):
        for axis in range(3):
            xyz[:, axis] |= (((codes >> (3 * bit + axis)) & 1) << bit).astype(np.uint16)
    return xyz


def varints(values):
    out = bytearray()
    for number in values:
        value = int(number)
        while value >= 128:
            out.append((value & 127) | 128)
            value >>= 7
        out.append(value)
    return bytes(out)


def unvarints(raw, count):
    out = np.empty(count, np.uint64)
    index = shift = value = 0
    for byte in raw:
        value |= (byte & 127) << shift
        if byte & 128:
            shift += 7
            if shift > 63:
                raise ValueError("invalid varint")
        else:
            if index >= count:
                raise ValueError("too many varints")
            out[index] = value
            index += 1
            shift = value = 0
    if shift or index != count:
        raise ValueError("truncated varints")
    return out


def octree(codes, bits):
    if len(np.unique(codes)) != len(codes):
        raise ValueError("octree control requires unique positions")
    out = bytearray()
    for level in range(bits):
        prefixes = np.unique(codes >> (3 * (bits - level - 1)))
        _, inverse = np.unique(prefixes >> 3, return_inverse=True)
        masks = np.zeros(int(inverse.max()) + 1, np.uint8)
        np.bitwise_or.at(masks, inverse, (1 << (prefixes & 7)).astype(np.uint8))
        out.extend(masks.tobytes())
    return bytes(out)


def unoctree(raw, bits):
    prefixes = np.zeros(1, np.uint64)
    offset = 0
    for _ in range(bits):
        masks = np.frombuffer(raw, np.uint8, count=len(prefixes), offset=offset)
        offset += len(prefixes)
        active = ((masks[:, None] >> np.arange(8)) & 1).astype(bool)
        prefixes = ((prefixes[:, None] << 3) | np.arange(8, dtype=np.uint64))[active]
    if offset != len(raw):
        raise ValueError("octree length mismatch")
    return prefixes


def encode_positions(xyz, bits, layout, codec):
    xyz = np.asarray(xyz, dtype=np.uint16)
    if not 1 <= bits <= 16 or xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz):
        raise ValueError("expected nonempty XYZ and 1..16 bits")
    if int(xyz.max()) >= 1 << bits:
        raise ValueError("coordinate exceeds bit depth")
    dtype = np.dtype("<u2" if bits > 8 else "u1")
    if layout == "xyz_planar":
        raw = xyz.T.astype(dtype).tobytes()
    elif layout == "xyz_moddelta":
        delta = np.diff(xyz.astype(np.int32), axis=0, prepend=np.zeros((1, 3), np.int32)) & ((1 << bits) - 1)
        raw = delta.T.astype(dtype).tobytes()
    elif layout == "xyz_byteplanes":
        raw = xyz.astype(dtype).view(np.uint8).reshape(len(xyz), 3, dtype.itemsize).transpose(2, 1, 0).tobytes()
    else:
        codes = morton(xyz, bits)
        if np.any(codes[1:] < codes[:-1]):
            raise ValueError("Morton/octree layouts require sorted input")
        if layout == "morton_delta_varint":
            raw = varints(np.diff(codes, prepend=np.uint64(0)))
        elif layout == "octree":
            raw = octree(codes, bits)
        else:
            raise ValueError(f"unknown layout: {layout}")
    if codec == "deflate":
        compressed = zlib.compress(raw, 9)
    elif codec == "lzma":
        compressed = lzma.compress(raw, preset=6)
    else:
        raise ValueError(f"unknown codec: {codec}")
    header = json.dumps({"bits": bits, "records": len(xyz), "layout": layout, "codec": codec},
                        separators=(",", ":")).encode()
    return b"EG01" + struct.pack("<I", len(header)) + header + compressed


def decode_positions(blob):
    if blob[:4] != b"EG01":
        raise ValueError("unknown geometry container")
    size, = struct.unpack("<I", blob[4:8])
    if not 0 < size < 65536:
        raise ValueError("invalid geometry header")
    header = json.loads(blob[8:8+size])
    bits, count, layout, codec = (header[key] for key in ("bits", "records", "layout", "codec"))
    if not 1 <= bits <= 16 or count < 1:
        raise ValueError("invalid geometry shape")
    payload = blob[8+size:]
    if codec == "deflate":
        raw = zlib.decompress(payload)
    elif codec == "lzma":
        raw = lzma.decompress(payload)
    else:
        raise ValueError("unknown geometry compression")
    dtype = np.dtype("<u2" if bits > 8 else "u1")
    if layout in {"xyz_planar", "xyz_moddelta"}:
        xyz = np.frombuffer(raw, dtype).reshape(3, count).T.astype(np.uint16)
        if layout == "xyz_moddelta":
            xyz = (np.cumsum(xyz, axis=0) & ((1 << bits) - 1)).astype(np.uint16)
    elif layout == "xyz_byteplanes":
        xyz = np.frombuffer(raw, np.uint8).reshape(dtype.itemsize, 3, count).transpose(2, 1, 0).copy().view(dtype).reshape(count, 3).astype(np.uint16)
    elif layout == "morton_delta_varint":
        xyz = unmorton(np.cumsum(unvarints(raw, count), dtype=np.uint64), bits)
    elif layout == "octree":
        xyz = unmorton(unoctree(raw, bits), bits)
    else:
        raise ValueError("unknown geometry layout")
    if xyz.shape != (count, 3) or int(xyz.max()) >= 1 << bits:
        raise ValueError("geometry reconstruction out of bounds")
    return xyz
