#!/usr/bin/env bash
set -euo pipefail

image="${1:-nerd-shapemapper2:2.3-public-candidate}"
container_engine="${CONTAINER_ENGINE:-docker}"
log_file="$(mktemp "${TMPDIR:-/tmp}/nerd-shapemapper2-upstream.XXXXXX")"
trap 'rm -f "${log_file}"' EXIT

set +e
"${container_engine}" run --rm \
  --platform linux/amd64 \
  --entrypoint bash \
  "${image}" \
  /opt/shapemapper2/internals/test/run_all_tests.sh 2>&1 | tee "${log_file}"
test_status="${PIPESTATUS[0]}"
set -e

if [[ "${test_status}" -eq 0 ]] \
  && grep -Fxq 'All tests passed' "${log_file}" \
  && grep -Fxq 'SUCCESS' "${log_file}"; then
  echo "All ShapeMapper2 upstream tests passed"
  exit 0
fi

# Upstream documents that one environment-dependent module-failure injection
# test may be ignored. Accept only that exact aggregate result; normal pipeline,
# unit, variant-correction, and ROC failures still fail this validation.
expected_summaries=(
  $'0 / 98\tc++ unit test(s) failed.'
  $'0 / 32\tend-to-end success test(s) failed.'
  $'1 / 63\tmodule failure detection test(s) failed.'
  $'0 / 13\tsequence variant correction test(s) failed.'
  $'0 / 2\tarea under ROC curve test(s) failed.'
  $'1 / 208\ttotal test(s) failed.'
)

if [[ "${test_status}" -eq 1 ]]; then
  for summary in "${expected_summaries[@]}"; do
    grep -Fxq "${summary}" "${log_file}" || exit 1
  done
  grep -Fxq 'FAILURE' "${log_file}" || exit 1
  echo "Accepted the single environment-dependent failure allowed by upstream"
  exit 0
fi

echo "ShapeMapper2 upstream validation failed with status ${test_status}" >&2
exit "${test_status}"
