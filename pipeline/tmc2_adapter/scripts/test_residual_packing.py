"""Regression coverage for existing and wider experimental residuals."""
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np

from build_v17_hybrid import pack_low_coordinates
from decode_v18_hybrid import unpack_low_coordinates, unpack_bitplanes
from probe_geometry_layouts import residual_trials


class ResidualPackingTest(unittest.TestCase):
    def test_widths_and_existing_byte_layout(self):
        rng = np.random.default_rng(42)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "low.deflate"
            for bits in range(1, 11):
                values = rng.integers(0, 1 << bits, (29, 3), dtype=np.uint16)
                raw = pack_low_coordinates(values, bits)
                expected = np.packbits(((values.reshape(-1, 1) >> np.arange(bits)) & 1).reshape(-1),
                                       bitorder="little").tobytes()
                self.assertEqual(raw, expected)
                path.write_bytes(zlib.compress(raw))
                np.testing.assert_array_equal(unpack_low_coordinates(path, len(values), bits), values)

    def test_wide_residual_layouts(self):
        rng = np.random.default_rng(43)
        for bits in (8, 9, 10):
            values = rng.integers(0, 1 << bits, (59, 3), dtype=np.uint16)
            self.assertTrue(all(row["exact"] for row in residual_trials(values, bits)))

    def test_packaged_bitplanes(self):
        rng = np.random.default_rng(44)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for bits in (8, 9, 10):
                values = rng.integers(0, 1 << bits, (59, 3), dtype=np.uint16)
                paths = []
                for bit in range(bits):
                    name = f"plane_{bit}"
                    raw = np.packbits(((values.reshape(-1) >> bit) & 1).astype(np.uint8), bitorder="little")
                    (root / name).write_bytes(zlib.compress(raw.tobytes()))
                    paths.append(name)
                np.testing.assert_array_equal(unpack_bitplanes(root, paths, len(values), bits), values)


if __name__ == "__main__":
    unittest.main()
