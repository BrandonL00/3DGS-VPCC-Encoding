import unittest

import numpy as np

from lossless_geometry import decode_positions, encode_positions, morton


class LosslessGeometryTest(unittest.TestCase):
    def test_round_trips_boundaries_and_partial_octree_levels(self):
        rng = np.random.default_rng(42)
        for bits in (1, 8, 9, 10, 16):
            values = rng.integers(0, 1 << bits, (47, 3), dtype=np.uint16)
            values = np.unique(np.vstack([values, [0, 0, 0], [(1 << bits) - 1] * 3]), axis=0).astype(np.uint16)
            values = values[np.argsort(morton(values, bits), kind="stable")]
            for layout in ("xyz_planar", "xyz_byteplanes", "xyz_moddelta", "morton_delta_varint", "octree"):
                for codec in ("deflate", "lzma"):
                    with self.subTest(bits=bits, layout=layout, codec=codec):
                        blob = encode_positions(values, bits, layout, codec)
                        np.testing.assert_array_equal(decode_positions(blob), values)

    def test_duplicate_positions_and_single_point(self):
        for values in (np.zeros((1, 3), np.uint16), np.ones((5, 3), np.uint16)):
            for layout in ("xyz_planar", "xyz_byteplanes", "xyz_moddelta", "morton_delta_varint"):
                np.testing.assert_array_equal(decode_positions(encode_positions(values, 16, layout, "deflate")), values)
        with self.assertRaises(ValueError):
            encode_positions(np.ones((2, 3), np.uint16), 8, "octree", "deflate")


if __name__ == "__main__":
    unittest.main()
