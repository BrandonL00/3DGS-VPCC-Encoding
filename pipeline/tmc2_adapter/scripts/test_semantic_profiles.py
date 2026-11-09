import unittest
from types import SimpleNamespace

from semantic_profiles import codebook_size, configure, install_layout_transform


class SemanticProfilesTest(unittest.TestCase):
    def test_budget_edges(self):
        self.assertEqual(codebook_size(64000, "legacy"), 63504)
        self.assertEqual(codebook_size(128000, "legacy"), 65536)
        self.assertEqual(codebook_size(128000, "kiss"), 14400)
        self.assertEqual(codebook_size(256000, "kiss"), 30976)
        with self.assertRaises(ValueError):
            codebook_size(64000, "legacy", 65536)

    def test_profile_copy_and_explicit_settings(self):
        from ffsplat.coding.scene_encoder import EncodingParams
        base = EncodingParams.from_template_yaml("SOG-web")
        revised, report = configure(base, 128000, policy="kiss", levels=100, layout="plas", tiled=True)
        self.assertEqual(report["clusters"], 14400)
        self.assertFalse(any("vpcc_sh_layout" in t for op in base.ops for t in op["transforms"]))
        self.assertTrue(any("vpcc_sh_layout" in t for op in revised.ops for t in op["transforms"]))
        # SH layout must never change the scene's position sorting recipe.
        spatial_before = [op for op in base.ops if "sorted_indices" in op.get("input_fields", [])
                          or any("sort" in t for t in op["transforms"])]
        spatial_after = [op for op in revised.ops if "sorted_indices" in op.get("input_fields", [])
                         or any("sort" in t for t in op["transforms"])]
        self.assertEqual(spatial_before, spatial_after)

    def test_layout_and_tile_inverse(self):
        import torch
        from ffsplat.models.fields import Field
        from ffsplat.models.operations import Operation
        from ffsplat.models.transformations import transformation_map
        install_layout_transform()
        centers = torch.arange(16 * 45, dtype=torch.float32).reshape(16, 45).flip(0)
        labels = torch.tensor([0, 8, 15, 2, 8])
        proxy = SimpleNamespace(input_fields={"f_rest_centroids": Field(centers, None),
                                              "f_rest_labels": Field(labels, None)})
        fields, _ = transformation_map["vpcc_sh_layout"].apply({"method": "lexicographic"}, proxy)
        self.assertTrue(torch.equal(fields["f_rest_centroids"].data[fields["f_rest_labels"].data], centers[labels]))
        tiled, inverse = transformation_map["vpcc_sh_tile"].apply({"side": 4}, SimpleNamespace(input_fields=fields))
        fields.update(tiled)
        for update in reversed(inverse):
            for transform in reversed(update["transforms"]):
                op = Operation.from_json(update["input_fields"], transform, fields)
                decoded, _ = op.apply(verbose=False, decoding_params_hashable="")
                fields.update(decoded)
        self.assertTrue(torch.equal(fields["f_rest_centroids"].data[fields["f_rest_labels"].data], centers[labels]))


if __name__ == "__main__":
    unittest.main()
