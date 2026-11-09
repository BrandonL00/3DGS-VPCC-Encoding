"""Local SH policies composed with ffsplat's public operation/profile interface.

No upstream files or decoder patches are required. Layout is an encoder-only
permutation of centroids and inverse permutation of assignments. Quantization,
image coding, metadata and decoding continue to belong to ffsplat.
"""
from __future__ import annotations

import math
from copy import deepcopy


def codebook_size(count: int, policy: str, explicit: int | None = None) -> int:
    if count < 1:
        raise ValueError("empty Gaussian scene")
    if explicit is not None:
        if not 1 <= explicit <= min(count, 65536) or math.isqrt(explicit) ** 2 != explicit:
            raise ValueError("SH clusters must be a perfect square <= min(scene count, 65536)")
        return explicit
    if policy == "legacy":
        return min(256, math.isqrt(count)) ** 2
    if policy == "kiss":
        side = max(8, min(256, math.isqrt(count // 8)))
        side = min(math.isqrt(count), side - side % 8)
        return max(1, side) ** 2
    raise ValueError(f"unknown SH budget policy: {policy}")


def configure(parameters, count: int, *, policy="legacy", clusters=None,
              levels=32, layout="none", tiled=False):
    if not 2 <= levels <= 256:
        raise ValueError("SH centroid levels must be between 2 and 256")
    if layout not in {"none", "lexicographic", "plas"}:
        raise ValueError("unsupported codebook layout")
    parameters = deepcopy(parameters)
    size = codebook_size(count, policy, clusters)
    side = math.isqrt(size)
    found_cluster = found_quantizer = False
    centroid_op = None
    for operation in parameters.ops:
        for transform in operation["transforms"]:
            if "cluster" in transform and operation["input_fields"] == ["f_rest"]:
                transform["cluster"]["num_clusters"] = size
                found_cluster = True
            if "simple_quantize" in transform:
                if operation["input_fields"] == ["f_rest_labels"]:
                    transform["simple_quantize"]["max"] = size - 1
                if operation["input_fields"] == ["f_rest_centroids"]:
                    transform["simple_quantize"]["max"] = levels - 1
                    centroid_op = operation
                    found_quantizer = True
    if not found_cluster or not found_quantizer:
        raise ValueError("ffsplat profile lacks the expected SH cluster/quantizer interface")
    # Quantize before reordering, preserving exactly the existing centroid values.
    if layout != "none":
        parameters.ops.remove(centroid_op)
        index = next(i for i, op in enumerate(parameters.ops)
                     if any("cluster" in t for t in op["transforms"])) + 1
        parameters.ops[index:index] = [centroid_op, {
            "input_fields": ["f_rest_centroids", "f_rest_labels"],
            "transforms": [{"vpcc_sh_layout": {"method": layout}}],
        }]
    if tiled:
        # Channel-major 45D centroids -> a 3x5 tiled RGB image, as in KISS.
        index = next(i for i, op in enumerate(parameters.ops)
                     if op["input_fields"] == ["f_rest_centroids"]
                     and any("write_file" in t for t in op["transforms"]))
        parameters.ops.insert(index, {
            "input_fields": ["f_rest_centroids"],
            "transforms": [{"vpcc_sh_tile": {"side": side}}],
        })
    return parameters, {"policy": policy, "clusters": size, "centroid_levels": levels,
                        "layout": layout, "tiled": tiled, "input_gaussians": count}


def install_layout_transform():
    import numpy as np
    import torch
    from types import SimpleNamespace
    from ffsplat.models.fields import Field
    from ffsplat.models.transformations import Sort, transformation_map

    class CodebookLayout:
        @staticmethod
        def apply(params, parentOp, verbose=False, **kwargs):
            centers = parentOp.input_fields["f_rest_centroids"].data
            labels = parentOp.input_fields["f_rest_labels"].data
            if params["method"] == "lexicographic":
                array = centers.detach().cpu().numpy()
                permutation = torch.as_tensor(np.lexsort(array.T[::-1]), device=centers.device)
            else:
                proxy = SimpleNamespace(input_fields={"f_rest_centroids": parentOp.input_fields["f_rest_centroids"]})
                result, _ = Sort.apply({"method": "plas", "to_field": "order", "prune_by": None,
                    "scaling_fn": "none", "shuffle": False, "improvement_break": 1e-4,
                    "weights": {"f_rest_centroids": 1.0}}, proxy, verbose=verbose)
                permutation = result["order"].data.reshape(-1)
            inverse = torch.argsort(permutation)
            reordered = centers[permutation]
            remapped = inverse[labels.to(torch.int64)]
            # Exact check: layout must never change a reconstructed coefficient.
            for start in range(0, labels.numel(), 65536):
                original_ids = labels.reshape(-1)[start:start + 65536].to(torch.int64)
                new_ids = remapped.reshape(-1)[start:start + 65536]
                if not torch.equal(reordered[new_ids], centers[original_ids]):
                    raise RuntimeError("SH layout changed decoded coefficients")
            return {"f_rest_centroids": Field(reordered, parentOp),
                    "f_rest_labels": Field(remapped, parentOp)}, []

    transformation_map["vpcc_sh_layout"] = CodebookLayout

    class CodebookTile:
        @staticmethod
        def apply(params, parentOp, verbose=False, **kwargs):
            side = params["side"]
            centers = parentOp.input_fields["f_rest_centroids"].data
            tiled = centers.reshape(side, side, 3, 3, 5).permute(3, 0, 4, 1, 2).reshape(3 * side, 5 * side, 3)
            # ffsplat's Permute encoder records the same permutation for decode;
            # supply the actual inverse here using standard decoder operations.
            inverse_ops = [{"input_fields": ["f_rest_centroids"], "transforms": [
                # SceneEncoder reverses both operations and transform lists.
                {"reshape": {"shape": [side * side, 45]}},
                {"permute": {"dims": [1, 3, 4, 0, 2]}},
                {"reshape": {"shape": [3, side, 5, side, 3]}},
            ]}]
            return {"f_rest_centroids": Field(tiled, parentOp)}, inverse_ops

    transformation_map["vpcc_sh_tile"] = CodebookTile
