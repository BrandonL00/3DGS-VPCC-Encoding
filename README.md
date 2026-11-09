# E19 paper companion

Static PlayCanvas viewer for the actual decoded combined C18.3 + E19 Bicycle
results (64k,256k,1.024M). Engine pinned to 2.23.3 and vendored with its MIT
license. No CDN or server-side rendering dependency.

PLYs exceeding GitHub's regular-Git file limit are stored as 48 MiB byte chunks.
All chunks are downloaded and SHA-256 checked, then reconstructed into the
unchanged PLY using a Blob. These are not tiles or lower-resolution proxies.
Nothing about the archived PLY attributes is converted or requantized.

Serve this branch root using `python3 -m http.server 8765`, then open
http://localhost:8765/. A file:// URL will not work. The viewer keeps its camera
on budget changes and supports mouse, touch and keyboard interaction.

GitHub Pages: source branch `gh-pages`, directory `/ (root)`. `.nojekyll` keeps
the static asset layout intact. Main codec branch is independent.

Files were prepared by the research workspace's tools/pages/prepare_pages.py,
which verifies all PLY and chunk hashes against the saved benchmark records.
The manifest identifies each archived result and actual retained population.
Source-render quality metrics refer to the paper benchmark, not browser output.

Artifact PLYs derive from the Mip-NeRF 360 Bicycle scene and pretrained input
used in our experiments. Dataset/model attribution and upstream terms remain
applicable; the project's MIT license does not relicense those data.
