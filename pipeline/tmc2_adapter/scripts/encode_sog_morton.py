#!/usr/bin/env python3
"""Encode a matched SOG-web container with Morton, not PLAS, ordering.

It starts with ffsplat's SOG-web profile and substitutes deterministic full-
16-bit Morton ordering after the same opacity trimming to a square grid.
Optional SH budget, precision and codebook-layout settings are composed with
that profile; the E19 runner supplies its selected settings explicitly.

Run with the ffsplat environment, for example:
  /path/to/ffsplat/.venv/bin/python encode_sog_morton.py input.ply output_dir
"""

from __future__ import annotations

import argparse
import math
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ffsplat.coding.scene_decoder import decode_gaussians
from ffsplat.coding.scene_encoder import DecodingParams, EncodingParams, SceneEncoder
from ffsplat.models.fields import Field
from ffsplat.models.transformations import Sort
from semantic_profiles import configure, install_layout_transform

ORDER_OUTPUT: Path | None = None
ORDER_ONLY = False


def morton3_16(values: torch.Tensor) -> torch.Tensor:
    """Interleave a `(N, 3)` uint16 tensor into deterministic uint64 keys."""
    code = torch.zeros(values.shape[0], dtype=torch.int64, device=values.device)
    values = values.to(torch.int64)
    for bit in range(16):
        code |= ((values[:, 0] >> bit) & 1) << (3 * bit)
        code |= ((values[:, 1] >> bit) & 1) << (3 * bit + 1)
        code |= ((values[:, 2] >> bit) & 1) << (3 * bit + 2)
    return code


def morton_sort_apply(
    params: dict[str, Any], parent_op: Any, verbose: bool = False, **_: Any
) -> tuple[dict[str, Field], list[dict[str, Any]]]:
    """The SOG Sort extension used only when profile method is `morton16`."""
    if params.get("method") != "morton16":
        return ORIGINAL_SORT_APPLY(params, parent_op, verbose=verbose)

    fields = parent_op.input_fields
    prune_field = fields[params["prune_by"]].data.squeeze()
    primitive_count = int(prune_field.shape[0])
    side = math.isqrt(primitive_count)
    retained = side * side
    if retained != primitive_count:
        if verbose:
            print(f"Removing {primitive_count - retained}/{primitive_count} primitives to fit the grid.")
        _, source_indices = torch.topk(prune_field, k=retained)
        source_indices = torch.sort(source_indices).values
    else:
        source_indices = torch.arange(primitive_count, device=prune_field.device)

    low = fields["means_bytes_0"].data[source_indices].to(torch.int64)
    high = fields["means_bytes_1"].data[source_indices].to(torch.int64)
    quantized_means = low | (high << 8)
    order = torch.argsort(morton3_16(quantized_means), stable=True)
    sorted_indices = source_indices[order].reshape(side, side)
    if ORDER_OUTPUT is not None:
        ORDER_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        np.save(ORDER_OUTPUT, sorted_indices.detach().cpu().numpy())
        if ORDER_ONLY:
            raise SystemExit(f"Wrote exact Morton source permutation: {ORDER_OUTPUT}")
    return {params["to_field"]: Field(sorted_indices, parent_op)}, []


ORIGINAL_SORT_APPLY = Sort.apply


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--sh-policy", choices=["legacy", "kiss"], default="legacy")
    parser.add_argument("--sh-clusters", type=int)
    parser.add_argument("--sh-levels", type=int,
                        help="Centroid precision; defaults to 32 for legacy or 100 for kiss.")
    parser.add_argument("--sh-layout", choices=["none", "lexicographic", "plas"], default="none")
    parser.add_argument("--sh-tiled", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scene-order", choices=["morton", "plas"], default="morton",
                        help="Spatial layout; plas is a diagnostic SOG control, not the hybrid's production order.")
    parser.add_argument("--save-order", type=Path, help="Write source indices at the Morton sort stage, then exit")
    parser.add_argument("--order-output", type=Path, help="Write source indices at the Morton sort stage and continue encoding")
    args = parser.parse_args()
    if args.sh_levels is None:
        args.sh_levels = 100 if args.sh_policy == "kiss" else 32
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    if args.scene_order != "morton" and (args.save_order or args.order_output):
        parser.error("source permutation export currently requires Morton scene order")

    parameters = EncodingParams.from_template_yaml("SOG-web")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    install_layout_transform()
    for operation in parameters.ops:
        for transform in operation["transforms"]:
            if "sort" in transform:
                transform["sort"]["method"] = "morton16" if args.scene_order == "morton" else "plas"

    if args.save_order and args.order_output:
        raise SystemExit("choose only one of --save-order or --order-output")
    global ORDER_OUTPUT, ORDER_ONLY
    ORDER_OUTPUT = args.save_order or args.order_output
    ORDER_ONLY = args.save_order is not None
    Sort.apply = staticmethod(morton_sort_apply)
    gaussians = decode_gaussians(args.input, "3DGS-INRIA.ply", verbose=args.verbose).to(args.device)
    count = int(gaussians.to_field_dict()["means"].data.shape[0])
    parameters, sh_report = configure(parameters, count, policy=args.sh_policy,
        clusters=args.sh_clusters, levels=args.sh_levels, layout=args.sh_layout, tiled=args.sh_tiled)
    encoder = SceneEncoder(
        encoding_params=parameters,
        output_path=args.output,
        fields=gaussians.to_field_dict(),
        decoding_params=DecodingParams.default_ffsplat_packer(
            profile=f"SOG-web-{args.scene_order}-{args.sh_policy}-{args.sh_layout}-tile{int(args.sh_tiled)}",
            profile_version=parameters.profile_version,
            scene=parameters.scene,
        ),
    )
    encoder.encode(verbose=args.verbose)
    (args.output / "sh_encoding_report.json").write_text(json.dumps({**sh_report, "seed": args.seed, "scene_order": args.scene_order}, indent=2) + "\n")
    print(f"Successfully wrote matched {args.scene_order} SOG container: {args.output}")


if __name__ == "__main__":
    main()
