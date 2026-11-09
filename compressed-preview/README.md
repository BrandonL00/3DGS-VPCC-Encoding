# E19 experimental compressed-package viewer

Static PlayCanvas viewer for the actual combined C18.3 + E19 packages across
nine Mip-NeRF 360 scenes and five budgets:64k,128k,256k,512k,1.024M.
The browser downloads compressed V-PCC/AVIF/residual streams and reconstructs
Gaussian data inside a disposable Web Worker. No server decoder or CUDA is
required. No predecoded PLYs or native pixel arrays are hosted in this gallery.

TMC2+HM runs in WebAssembly with its internal video decoder. A portable AVIF
WebAssembly decoder restores the semantic images and is checked against native
pixel hashes; no browser ImageDecoder API is required. The original quantizers,
associations and SH codebooks are reconstructed without another quantization.
The complete PLY can be downloaded after decoding. Small floating-point
rounding differences from native opacity/scale reconstruction remain.

**Experimental:** full-package decoding passed native-reference comparisons
for45/45 packages in Node. Main-thread loader/real compute-worker interaction
is also tested in Node. Browser UI/WebGL validation is pending because local
browser automation was disabled. The working PLY gallery remains separate.

The geometry decoder allocates roughly930MB of linear memory; desktop browsers
are the initial target. Rendering and decoded Gaussian buffers need additional
memory. The worker is terminated before uploading decoded data to the GPU.

Serve this branch root using `python3 -m http.server 8765`, then open
http://localhost:8765/. A file:// URL will not work. The viewer keeps its camera
on budget changes and supports mouse, touch and keyboard interaction.

GitHub Pages: experimental gallery under `compressed-preview/` on `gh-pages`.
The root gallery and codec main branch are independent.

Files were prepared by tools/wasm/stage_collection.py, preserving the actual
compressed streams and exporting a JSON interpretation of their YAML metadata.
artifacts.json identifies results and populations. decoder/TMC2-COPYING and
decoder/vendor/avif/LICENSE preserve native/image-decoder notices. PlayCanvas
2.23.3 and its MIT license are under vendor/.
Source-render quality metrics refer to the paper benchmark, not browser output.

Artifact PLYs derive from the Mip-NeRF 360 Bicycle scene and pretrained input
used in our experiments. Dataset/model attribution and upstream terms remain
applicable; the project's MIT license does not relicense those data.
