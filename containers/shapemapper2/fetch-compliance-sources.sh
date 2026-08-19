#!/usr/bin/env bash
set -euo pipefail

destination="${1:-compliance-sources}"
mkdir -p "${destination}"

if command -v sha256sum >/dev/null 2>&1; then
  checksum_command=(sha256sum)
elif command -v shasum >/dev/null 2>&1; then
  checksum_command=(shasum -a 256)
else
  echo "sha256sum or shasum is required" >&2
  exit 2
fi

fetch() {
  local name="$1"
  local expected="$2"
  local url="$3"
  local target="${destination}/${name}"
  local partial="${target}.partial"
  local actual

  if [[ -f "${target}" ]]; then
    actual="$("${checksum_command[@]}" "${target}" | awk '{print $1}')"
    if [[ "${actual}" == "${expected}" ]]; then
      echo "verified existing ${name}"
      return
    fi
  fi

  curl --fail --location --retry 3 --output "${partial}" "${url}"
  actual="$("${checksum_command[@]}" "${partial}" | awk '{print $1}')"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "checksum mismatch for ${name}: ${actual}" >&2
    exit 1
  fi
  mv "${partial}" "${target}"
  echo "downloaded and verified ${name}"
}

fetch bowtie2-v2.3.4.3-source.tar.gz \
  f10fc386277677329f4b9d1cb6951e6b8e6c125f07438476bd3653c79ad00b07 \
  https://codeload.github.com/BenLangmead/bowtie2/tar.gz/refs/tags/v2.3.4.3
fetch STAR-2.5.2a-source.tar.gz \
  2a372d9bcab1dac8d35cbbed3f0ab58291e4fbe99d6c1842b094ba7449d55476 \
  https://codeload.github.com/alexdobin/STAR/tar.gz/refs/tags/2.5.2a
fetch ghostscript-9.25-source.tar.gz \
  baafa64740b090bff50b220a6df3be95c46069b7e30f4b4effed28316e5b2389 \
  https://github.com/ArtifexSoftware/ghostpdl-downloads/releases/download/gs925/ghostscript-9.25.tar.gz
fetch BBMap_37.78-source.tar.gz \
  f2da19f64d2bfb7db4c0392212668b425c96a27c77bd9d88d8f0aea90a193509 \
  https://downloads.sourceforge.net/project/bbmap/BBMap_37.78.tar.gz
fetch graphviz-7.1.0-source.tar.gz \
  8b28a283644a8442e6925b15d95055228d25172c7c30681810625616cbb23913 \
  https://gitlab.com/graphviz/graphviz/-/archive/7.1.0/graphviz-7.1.0.tar.gz
fetch openjdk-jdk8u112-b16-upstream-baseline.tar.gz \
  62447a6609b46055b80ad6c9b22a99f60d240bc63bfb5274d05bc811f5dfeca7 \
  https://codeload.github.com/openjdk/jdk8u/tar.gz/64ad20c00f850c3ed4c796c72ade483460a7897f

fetch bbmap-37.78-0.tar.bz2 \
  dd8dcf31f2fdffd22a79583d4ba56f7da21beb20282118fb8750402f4d4e23a2 \
  https://conda.anaconda.org/bioconda/linux-64/bbmap-37.78-0.tar.bz2
fetch star-2.5.2a-0.tar.bz2 \
  7af6e4aa48ade269e4e9435da02ce01dae45c8072af8a0045fde131d5f6c3da2 \
  https://conda.anaconda.org/bioconda/linux-64/star-2.5.2a-0.tar.bz2
fetch graphviz-7.1.0-h2e5815a_0.conda \
  cecaa9e6dce7f2df042768d9a794f0126565a30384fcd59879e107d760bed7f1 \
  https://conda.anaconda.org/conda-forge/linux-64/graphviz-7.1.0-h2e5815a_0.conda
fetch openjdk-8.0.112-zulu8.19.0.1_3.tar.bz2 \
  14e90e90593065802e36147c801ed89c12924b695059aecc179a756eaf5386da \
  https://conda.anaconda.org/conda-forge/linux-64/openjdk-8.0.112-zulu8.19.0.1_3.tar.bz2
fetch zulu8.19.0.1-jdk8.0.112-linux_x64.tar.gz \
  9ddcfa6d5a549af216cc4efbde972a6d017e72cf1633313921f6a4106c348bc5 \
  https://cdn.azul.com/zulu/bin/zulu8.19.0.1-ca-jdk8.0.112-linux_x64.tar.gz

echo "Source collection complete. The OpenJDK archive is an upstream baseline,"
echo "not yet confirmed as complete corresponding source for the Azul build."
