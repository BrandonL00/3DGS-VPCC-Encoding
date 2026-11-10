# 3DGS + V-PCC interactive viewer

Static PlayCanvas viewer for combined compaction and compression across nine
Mip-NeRF 360 scenes and five budgets:64k,128k,256k,512k,1.024M. A second experiment
offers compressed-only Bicycle, Bonsai, and Garden at the maximum-quality preset.
E19 and C18.3 remain internal provenance identifiers rather than public branding.
The browser downloads compressed V-PCC/AVIF/residual streams and reconstructs
Gaussian data inside a disposable Web Worker. No server decoder or CUDA is
required. No predecoded PLYs or native pixel arrays are hosted in this gallery.

TMC2+HM runs in WebAssembly with its internal video decoder. A portable AVIF
WebAssembly decoder restores the semantic images and is checked against native
pixel hashes; no browser ImageDecoder API is required. The original quantizers,
associations and SH codebooks are reconstructed without another quantization.
The complete PLY can be downloaded after decoding. Small floating-point
rounding differences from native opacity/scale reconstruction remain.

Full-package decoding passed native-reference comparisons for the original
45 compacted packages in Node. Main-thread loader/real compute-worker interaction
is also tested in Node. User confirmed the original browser viewer works.
See validation.json and the maintainer handoff for the larger-package test status.

Desktop browsers are the target. Full-scene examples require several GB of
system memory plus GPU memory. The worker is terminated before uploading decoded
data to the GPU. Start with Bonsai for a smaller compressed-only example.

Serve this branch root using `python3 -m http.server 8765`, then open
http://localhost:8765/. A file:// URL will not work. The viewer keeps its camera
on budget changes and supports mouse, touch and keyboard interaction.

GitHub Pages: experimental gallery under `compressed-preview/` on `gh-pages`.
The root redirects to this gallery; codec main remains independent.

Files were prepared by tools/wasm/stage_collection.py and stage_preservation.py, preserving the actual
compressed streams and exporting a JSON interpretation of their YAML metadata.
artifacts.json identifies results and populations. decoder/TMC2-COPYING and
decoder/vendor/avif/LICENSE preserve native/image-decoder notices. PlayCanvas
2.23.3 and its MIT license are under vendor/.
Source-render quality metrics refer to the paper benchmark, not browser output.

Artifact PLYs derive from the Mip-NeRF 360 scenes and pretrained inputs
used in our experiments. Dataset/model attribution and upstream terms remain
applicable; the project's MIT license does not relicense those data.
