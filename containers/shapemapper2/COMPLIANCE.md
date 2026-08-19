# ShapeMapper2 container compliance checklist

This is a working redistribution checklist, not legal advice. Public GHCR
publication remains blocked until the public-candidate checklist is complete
and the project owner gives explicit approval.

## Public candidate (`Dockerfile`)

The public candidate compiles the checksum-pinned upstream v2.3 source tag,
after applying one documented Python 3.11 file-mode compatibility patch, and
uses Debian 12 packages. It excludes the official release archive's Miniconda
environment and old prebuilt third-party executables. That removes the exact
Azul Zulu and historical Conda source gaps described later in this document.

- [x] Pin the upstream ShapeMapper v2.3 source archive by SHA-256.
- [x] Preserve the ShapeMapper MIT notice in the source tree and image.
- [x] Apply only the documented `python311-open-mode.patch` (`rU` to `r` in
  30 file-open calls across 18 files); do not change scientific logic or
  vendor/fork the upstream repository.
- [x] Use distribution packages for Bowtie2, STAR, BBMap/BBMerge, Graphviz,
  Ghostscript, pv, OpenJDK, Python, Boost, zlib, and transitive dependencies.
- [x] Preserve Debian package copyright records under
  `/usr/share/doc/<package>/copyright`.
- [x] Add `PUBLIC_THIRD_PARTY_NOTICES.md` as an index to those records.
- [ ] Run the upstream example and all built-in tests against the final
  candidate image.
- [ ] Export the exact installed binary-to-source Debian package inventory
  using `dpkg-query`, and retain it with the release record.
- [ ] Generate a fresh SPDX SBOM from the final candidate; the existing
  `sbom.spdx.json` describes only the private reference image.
- [ ] Retrieve and retain the exact Debian corresponding-source packages needed
  for redistributed copyleft components. Do this from the tested final package
  inventory, not from the older Conda list.
- [ ] Confirm that every directly used executable is found at the expected
  command name, especially Debian's BBMerge launcher.
- [ ] Record the final image digest, CI run, package inventory, SBOM, and source
  bundle checksums together.
- [x] Obtain explicit owner approval before pushing a uniquely tagged private
  development image for NERD integration testing.
- [ ] Obtain explicit owner approval before creating a release tag or changing
  GHCR visibility to public.

The first local source build produced a 508,942,308-byte amd64 image. Runtime
inspection was interrupted by host disk exhaustion immediately after the
build. A second build compiled with the final two-job limit and installed all
runtime packages, but storage failed during final layer assembly. The size is
therefore provisional evidence rather than a release result.

## Private reference image (`Dockerfile.reference`)

The remaining sections document the official-release wrapper. They explain why
it is retained privately as a behavioral reference rather than published.

## Evidence already captured

- [x] Preserve the upstream ShapeMapper MIT license in the image and in
  `LICENSES/ShapeMapper-MIT.txt`.
- [x] Pin and checksum the exact upstream v2.3 release archive.
- [x] Generate an SPDX JSON SBOM from the locally built image.
- [x] Record the prominent bundled tools and their versions.
- [x] Extract and preserve the upstream archive's exact license texts for
  Bowtie2, Ghostscript, Miniconda, and pv.
- [x] Extract and preserve the Conda environment's exact license texts for
  BBMap, Graphviz, and OpenJDK.
- [x] Export all 218 embedded Conda package records, including exact package
  URLs and available checksums, to `conda-packages.tsv`.
- [x] Isolate 64 Conda records with copyleft-related metadata into
  `copyleft-conda-packages.tsv` for source/notice follow-up.
- [x] Download and checksum exact source releases for Bowtie2, STAR,
  Ghostscript, BBMap, and Graphviz.
- [x] Preserve the exact Conda package archives containing the historical
  build recipes for STAR, BBMap, Graphviz, and Azul Zulu OpenJDK in the source
  retrieval plan.

The current SBOM contains 168 package records. Thirty-six records have
`NOASSERTION` for at least one SPDX license field, but the Conda metadata
resolves most of those scanner omissions. Four Conda records themselves lack
a license value: `_libgcc_mutex`, `mysql-common`, `mysql-libs`, and the old
`readline` 6.2 package. The SBOM remains discovery evidence rather than a
complete redistribution determination.

## Historical release-wrapper review

- [ ] ShapeMapper2: retain the MIT copyright and permission notice.
- [x] Bowtie2 2.3.4.3: retain its bundled GPLv3 license and map its exact
  versioned source archive and build scripts.
- [x] STAR 2.5.2a: retain GPLv3 and map the exact source plus the Conda recipe
  that selected the bundled static executable.
- [ ] Ghostscript 9.25: conservatively treat the executable as AGPL and ship
  the official complete 9.25 source. Keep documenting that ShapeMapper's
  adjacent binary-archive `COPYING` file unexpectedly contains GPLv3 instead.
- [x] BBMap/BBMerge 37.78: retain the UC/LBNL notice. The license is permissive,
  the exact source was verified, and the installed package includes Java source.
- [x] pv 1.6.20: retain Artistic License 2.0. The bundled tarball is a source
  archive; the previous GPL classification was incorrect.
- [x] Graphviz 7.1.0: retain EPL-1.0 and map the exact source, Conda package,
  recipe, and patches.
- [ ] OpenJDK 8.0.112: the exact Zulu binary input, license, exception, third-
  party notices, source-offer identifier, Conda package, and recipe are known.
  The upstream OpenJDK 8u112-b16 source is only a baseline; obtain Azul's
  complete source for build `Zulu 8.19.0.1 d0cf8daf3adb` or replace this JDK
  with a source-traceable runtime before public distribution.
- [ ] Miniconda and Conda packages: retain Miniconda terms; resolve the four
  empty license records and map corresponding source for every remaining
  copyleft package, not only the prominent tools. The focused list currently
  contains 64 records, including large build/GUI stacks that ShapeMapper may
  not need at runtime.
- [ ] Debian base packages: retain required copyright/license material and
  document the Debian source retrieval path for the exact package versions.
- [x] Assemble a draft distributable `THIRD_PARTY_NOTICES.md`, license set,
  source manifest, and checksum-verifying source fetcher.
- [ ] Retain the verified source collection as immutable release artifacts;
  do not rely solely on upstream URLs.
- [ ] Optionally have the final notices/source bundle reviewed before changing
  GHCR visibility to public.

## Publication records for the public candidate

- [ ] Save the successful CI run URL and source commit.
- [ ] Save the pushed OCI digest and generated provenance attestation.
- [ ] Attach the SPDX SBOM to the immutable release record.
- [ ] Validate the digest-pinned image after conversion to SIF on Quest.
- [ ] Record the SIF SHA-256, runtime/module version, host, date, and test log.
- [ ] Obtain explicit owner approval before making the GHCR package public.
- [ ] Verify anonymous pull only after approval and visibility change.

Do not describe the container as compliance-complete while any review item is
unchecked.
