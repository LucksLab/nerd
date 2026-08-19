#!/usr/bin/env bash
set -euo pipefail

image_ref="${1:-}"
if [[ -z "${image_ref}" || "${image_ref}" != *@sha256:* ]]; then
  echo "usage: $0 ghcr.io/edr-choi/nerd-shapemapper2@sha256:<digest> [cache-directory]" >&2
  exit 2
fi

if command -v apptainer >/dev/null 2>&1; then
  runtime=apptainer
elif command -v singularity >/dev/null 2>&1; then
  runtime=singularity
else
  echo "Apptainer or Singularity is required" >&2
  exit 2
fi

cache_root="${2:-${SCRATCH:-${TMPDIR:-/tmp}}/nerd-shapemapper2-cache}"
mkdir -p "${cache_root}/oci" "${cache_root}/work"
export APPTAINER_CACHEDIR="${cache_root}/oci"
export SINGULARITY_CACHEDIR="${cache_root}/oci"

digest="${image_ref##*@sha256:}"
sif="${cache_root}/shapemapper2-${digest}.sif"
work_dir="$(mktemp -d "${cache_root}/work/validation.XXXXXX")"
trap 'rm -rf "${work_dir}"' EXIT

if [[ ! -s "${sif}" ]]; then
  "${runtime}" pull "${sif}" "docker://${image_ref}"
fi

"${runtime}" exec "${sif}" /opt/shapemapper2/shapemapper --version \
  | grep -F 'ShapeMapper v2.3'

"${runtime}" exec "${sif}" tar -C /opt/shapemapper2 -cf - example_data \
  | tar -xf - -C "${work_dir}"

"${runtime}" exec \
  --bind "${work_dir}:/work" \
  --pwd /work \
  "${sif}" \
  /opt/shapemapper2/shapemapper \
  --name example-results \
  --target example_data/TPP.fa \
  --amplicon \
  --overwrite \
  --min-depth 1000 \
  --modified --folder example_data/TPPplus \
  --untreated --folder example_data/TPPminus \
  --denatured --folder example_data/TPPdenat

test -s "${work_dir}/shapemapper_out/example-results_TPP_profile.txt"
test -s "${work_dir}/shapemapper_out/example-results_TPP.shape"

sha256sum "${sif}"
echo "SIF validation passed with ${runtime}: ${sif}"
