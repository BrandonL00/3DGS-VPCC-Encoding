# E19 package

The package is a directory with `manifest.json` as its entry point. The
manifest identifies the format mode, coordinate transform, record count,
anchor precision, associations, residual layout and payload accounting.

- `stream.bin`: genuine TMC2 V-PCC coded coarse geometry.
- Association stream: Morton-ordered records mapped to reconstructed anchors.
- `low_plane_*.deflate`: complementary fine-coordinate bitplanes, losslessly
  compressed after coordinate quantization.
- `semantic_sog/`: image-coded Gaussian attributes, SH codebook and labels,
  and ffsplat decoding metadata. Position images are replaced by V-PCC geometry.
- `v19_encoder_run.json`: encoder settings and input lineage.
- `restored_0000.ply`: validation decode produced by encode; it is not part of
  the encoded payload and may be omitted when distributing a package.

The reader currently supports historical package modes and experimental
geometry controls for compatibility; the release launcher encodes only the
selected V-PCC 8+8 bitplane baseline. The decoder requires no original PLY.
Quantization and SH clustering introduce distortion. SOG grid fitting can
trim a small number of records. Byte accounting excludes reconstructed PLYs,
logs and top-level reporting files; use manifest payload bytes rather than
directory size to compare rate.
