#!/usr/bin/env bash
set -euo pipefail

tmc2_commit="d47e6ce879206b8a7881ecabce9e1ad2e8254a38"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd "${script_dir}/../.." && pwd)"
tmc2_root="${TMC2_ROOT:-${HOME}/tools/mpeg-pcc-tmc2}"

mkdir -p "$(dirname "${tmc2_root}")"
if [[ ! -d "${tmc2_root}/.git" ]]; then
  git clone https://github.com/MPEGGroup/mpeg-pcc-tmc2.git "${tmc2_root}"
fi

git -C "${tmc2_root}" checkout --detach "${tmc2_commit}"

for patch in "${script_dir}"/patches/*.patch; do
  if git -C "${tmc2_root}" apply --ignore-space-change --ignore-whitespace --reverse --check "${patch}" >/dev/null 2>&1; then
    echo "Already applied: $(basename "${patch}")"
  else
    git -C "${tmc2_root}" apply --ignore-space-change --ignore-whitespace --check "${patch}"
    git -C "${tmc2_root}" apply --ignore-space-change --ignore-whitespace "${patch}"
    echo "Applied: $(basename "${patch}")"
  fi
done

cmake \
  -S "${tmc2_root}" \
  -B "${tmc2_root}/build" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_JMAPP_VIDEO_CODEC=OFF \
  -DUSE_HMAPP_VIDEO_CODEC=ON \
  -DUSE_HMLIB_VIDEO_CODEC=ON \
  -DENABLE_TBB=OFF \
  -DUSE_HDRTOOLS=OFF

hm_library_root="${tmc2_root}/dependencies/HM-16.20+SCM-8.8"
hm_root="${tmc2_root}/dependencies/HM-16.20+SCM-8.8-app"
if [[ ! -d "${hm_root}/.git" ]]; then
  # TMC2 patches its embedded HM copy into the pcc_hm namespace. Keep a clean,
  # separately built checkout for the HM application backend.
  git clone "${hm_library_root}" "${hm_root}"
fi
git -C "${hm_root}" checkout --detach cae35dba3dcf8e1b6121a70b397000af36301045
for patch in "${script_dir}"/patches-hm/*.patch; do
  if git -C "${hm_root}" apply --ignore-space-change --ignore-whitespace --reverse --check "${patch}" >/dev/null 2>&1; then
    echo "Already applied to HM: $(basename "${patch}")"
  else
    git -C "${hm_root}" apply --ignore-space-change --ignore-whitespace --check "${patch}"
    git -C "${hm_root}" apply --ignore-space-change --ignore-whitespace "${patch}"
    echo "Applied to HM: $(basename "${patch}")"
  fi
done

make -C "${hm_root}/build/linux" -j8
cmake --build "${tmc2_root}/build" --parallel 8

"${tmc2_root}/bin/PccAppEncoder" --help >/dev/null 2>&1 || true
"${tmc2_root}/bin/PccAppDecoder" --help >/dev/null 2>&1 || true

echo "TMC2 root: ${tmc2_root}"
echo "TMC2 commit: $(git -C "${tmc2_root}" rev-parse HEAD)"
sha256sum "${tmc2_root}/bin/PccAppEncoder" "${tmc2_root}/bin/PccAppDecoder"
sha256sum "${hm_root}/bin/TAppEncoderStatic" "${hm_root}/bin/TAppDecoderStatic"
