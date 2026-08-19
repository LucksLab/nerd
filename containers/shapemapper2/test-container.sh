#!/usr/bin/env bash
set -euo pipefail

image="${1:-nerd-shapemapper2:2.3-public-candidate}"
container_engine="${CONTAINER_ENGINE:-docker}"
work_dir="$(mktemp -d "${TMPDIR:-/tmp}/nerd-shapemapper2-test.XXXXXX")"
host_user="$(id -u):$(id -g)"

cleanup() {
  status="$?"
  if [[ "${status}" -ne 0 ]]; then
    echo "ShapeMapper smoke-test diagnostics from ${work_dir}:" >&2
    find "${work_dir}" -type f \( -name '*.stderr' -o -name '*_log.txt' \) -print -exec tail -n 200 {} \; >&2 || true
  fi
  rm -rf "${work_dir}"
  exit "${status}"
}
trap cleanup EXIT

"${container_engine}" image inspect "${image}" --format '{{.Architecture}}' | grep -qx amd64
"${container_engine}" run --rm --platform linux/amd64 "${image}" --version | grep -F 'ShapeMapper v2.3'

"${container_engine}" run --rm --platform linux/amd64 --entrypoint tar "${image}" \
  -C /opt/shapemapper2 -cf - example_data | tar -xf - -C "${work_dir}"

"${container_engine}" run --rm \
  --platform linux/amd64 \
  --user "${host_user}" \
  --entrypoint /opt/shapemapper2/shapemapper \
  --mount "type=bind,src=${work_dir},dst=/work" \
  --workdir /work \
  "${image}" \
  --name example-results \
  --target example_data/TPP.fa \
  --amplicon \
  --overwrite \
  --output-temp \
  --verbose \
  --min-depth 1000 \
  --modified --folder example_data/TPPplus \
  --untreated --folder example_data/TPPminus \
  --denatured --folder example_data/TPPdenat

test -s "${work_dir}/shapemapper_out/example-results_TPP_profile.txt"
test -s "${work_dir}/shapemapper_out/example-results_TPP.shape"

echo "ShapeMapper2 v2.3 container smoke test passed"
