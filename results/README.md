# Saved research results

These six CSVs are exports of the completed October 2026 E19 study, not new
measurements of this release build. `metric_evidence` / `manifest_evidence`
columns reference files in the larger research archive, which is not bundled.

- `fixed_cohort_graph_tables.csv`: equal-scene aggregates for graphing.
- `preservation_per_scene.csv`: four uncompacted SH settings, nine scenes.
- `compaction_per_scene.csv`: C18.3 + E19, 55 applicable budget/scene pairs.
- `bicycle_SH_sweep.csv`: SH-only quality/size sweep.
- `native_comparison_per_scene.csv`: native ffsplat, matched SOG and E19 controls.
- `codec_byte_breakdown.csv`: per-category encoded payload sizes.

For curves, X=payload_MB (individual scene) or mean_MB (aggregates), Y=PLY or
photo metric. MB means bytes/1,000,000. Choose a fixed scene cohort and keep
preservation and compaction as separate series. Outdoor five supports all
seven budgets; all nine supports budgets through 1.024M only.

PLY metrics compare renders with the ORIGINAL full source PLY. Photo metrics
compare with held-out photographs. Protocol: 1600px long side, LLFF holdout-8,
16–39 views depending on scene. Aggregates are arithmetic per-scene means.
Native ffsplat is itself a compressed SOG baseline; it is not the original PLY.
Matched/native differ in several settings; do not attribute all gains to V-PCC.
KISS-inspired C18.3 is a local recipe, not official KISS-GS code.

Curves do not include repeated-seed uncertainty estimates. Reused measurements
are not additional independent samples. Later November compaction-only work
is outside this selected October export.
