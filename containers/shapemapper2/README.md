# ShapeMapper2 v2.3 container

This directory contains two Linux-amd64 container recipes based on upstream
ShapeMapper2 v2.3. Neither recipe forks ShapeMapper, and neither image is
currently published to GHCR.

- `Dockerfile` is the smaller public candidate. It downloads the upstream v2.3
  source tag directly, verifies its SHA-256, applies the documented
  `patches/python311-open-mode.patch`, compiles ShapeMapper, and uses Debian
  packages for runtime dependencies. The patch replaces Python's removed
  `rU` file mode with equivalent `r` mode; it changes no scientific logic.
- `Dockerfile.reference` is the private reference recipe. It wraps the official
  upstream release archive, including its historical Miniconda environment.
  This image remains useful as a behavioral comparison but is not the public
  distribution candidate.

Intended image name:

```text
ghcr.io/edr-choi/nerd-shapemapper2:2.3-r0
```

Build and test locally:

```bash
docker build --platform linux/amd64 \
  --tag nerd-shapemapper2:2.3-public-candidate \
  containers/shapemapper2

containers/shapemapper2/test-container.sh nerd-shapemapper2:2.3-public-candidate
containers/shapemapper2/test-upstream.sh nerd-shapemapper2:2.3-public-candidate
```

The test wrappers use Docker by default. To run the same checks with Podman,
set `CONTAINER_ENGINE=podman`.

Build the private reference image only when a comparison is needed:

```bash
docker build --platform linux/amd64 \
  --file containers/shapemapper2/Dockerfile.reference \
  --tag nerd-shapemapper2:2.3-reference \
  containers/shapemapper2
```

Both recipes pin the amd64 Debian base manifest. The public candidate pins the
upstream source-tag archive and carries one small, auditable compatibility
patch rather than a ShapeMapper fork; the private reference recipe pins the
official release asset and does not patch it.

## Test levels

- The smoke test runs `shapemapper --version` and the upstream example using a
  host-mounted writable work directory.
- The full upstream suite can be run separately inside the image:

  ```bash
  docker run --rm --entrypoint bash nerd-shapemapper2:2.3-public-candidate \
    /opt/shapemapper2/internals/test/run_all_tests.sh
  ```

  The `test-upstream.sh` wrapper accepts either all 208 tests passing or the
  exact upstream-documented result in which one of 63 environment-dependent
  module-failure injection tests fails. It does not accept failures in normal
  end-to-end, unit, variant-correction, or ROC tests.

The full suite writes beneath `/opt/shapemapper2`, so this command is intended
for Docker validation. Singularity/Apptainer validation should copy the test
tree to a writable directory or use a writable temporary overlay.

### Private reference result

On 2026-08-18, the image built and the upstream example completed successfully.
After adding the redistribution records, the local amd64 image size was
2,053,836,415 bytes (about 1.91 GiB), before conversion to SIF. The example
also passed while running as the host UID/GID and writing all results through
a mounted work directory.

The complete upstream suite reported all 208 tests passing on the final beta
wrapper validation:

- 98 of 98 C++ unit tests
- 32 of 32 normal end-to-end tests
- 63 of 63 injected component-failure tests
- 13 of 13 sequence-variant correction tests
- 2 of 2 ROC tests

An earlier run exhibited the upstream-documented environment-sensitive
BowtieAligner failure-injection variation. The wrapper permits one such
failure-injection miss while rejecting any normal execution failure.

### Smaller public candidate result

On 2026-08-18, the source-based candidate compiled successfully from the
checksum-pinned upstream v2.3 source archive. Its local amd64 image size was
508,942,308 bytes (about 485 MiB), approximately 75% smaller than the private
reference image. A second low-memory build also compiled successfully with the
recipe's two-job compiler limit and installed the full Debian runtime. Runtime
and built-in test validation is pending because the host ran out of disk space
during final layer assembly, leaving both local container stores unavailable.
This is an infrastructure failure, not a ShapeMapper compile or test failure.
A subsequent GitHub Actions smoke test exposed Python 3.11's removal of the
legacy `rU` open mode. The documented compatibility patch fixes all 30 such
uses across 18 upstream Python files; final runtime validation is pending the
CI rerun.

## Publication gate

Do not create a release tag or change GHCR visibility until the public
candidate's built-in tests, image-size report, Debian package inventory, SBOM,
and source-availability record have been presented to and approved by the
project owner. Release tags must never be overwritten; NERD should consume an
image by OCI digest.

The GitHub Actions workflow uses only built-in Git and Docker commands because
LucksLab's organization policy disallows third-party actions. Pull requests
build and test without registry access. A manual run can, after all tests pass,
push a uniquely tagged private development image using the repository secret
`EDR_CHOI_GHCR_TOKEN`. It verifies that the GHCR package remains private and
prints the immutable digest for NERD testing. This private development image is
not a public release; generating and retaining a fresh SPDX SBOM remains part
of the public-release gate.

## Licensing status

ShapeMapper itself is MIT-licensed; its license is preserved in the upstream
tree and copied into `LICENSES/ShapeMapper-MIT.txt`. The smaller candidate
avoids the private reference image's unresolved historical Miniconda and Azul
Zulu source mapping by installing maintained Debian packages instead. Debian's
package-specific notices remain under `/usr/share/doc/<package>/copyright`.

The private reference image carries the historical records in
`THIRD_PARTY_NOTICES.md`, `source-manifest.tsv`, and the Conda inventories. The
smaller candidate instead carries `PUBLIC_THIRD_PARTY_NOTICES.md`; its exact
Debian binary/source package inventory and new SBOM must be generated from the
tested final image before approval.

## SIF validation

After an OCI image exists by digest, follow `QUEST.md` and run
`validate-sif-on-quest.sh`. The script supports either Singularity or Apptainer,
refuses mutable tags, caches by OCI digest, and runs the upstream example. An
actual Quest validation is pending registry reachability and Quest access.
