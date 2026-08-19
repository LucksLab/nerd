# Third-party notices for the source-built public candidate

This image compiles the checksum-pinned ShapeMapper2 v2.3 source tag after
applying `patches/python311-open-mode.patch`. That patch replaces Python's
removed `rU` file mode with equivalent `r` mode in 30 file-open calls and
changes no scientific logic. The image installs runtime dependencies as
Debian 12 (bookworm) packages. It does not contain the official ShapeMapper
release archive's bundled Miniconda environment or historical prebuilt
executables.

ShapeMapper2 is Copyright (c) 2018 Steven Busan and is distributed under the
MIT License. The complete notice is installed beside this file and remains in
the upstream source tree at `/opt/shapemapper2/LICENSE`.

The image includes Debian packages for BBMap/BBMerge, Bowtie2, STAR, Graphviz,
Ghostscript, pv, Python and its scientific libraries, Boost, zlib, and their
runtime dependencies. Their package-specific copyright and license notices
are retained under `/usr/share/doc/<package>/copyright`. The corresponding
Debian source packages can be located by the exact binary-package versions
reported by `dpkg-query -W` and retrieved from Debian's source archives.

This notice is an index, not a replacement for the package-specific notices.
The CI-generated SPDX SBOM and exact `dpkg-query` inventory are release
records for a published image and must be associated with its immutable OCI
digest.

The private reference image built with `Dockerfile.reference` has a different
dependency set and separate notices in `THIRD_PARTY_NOTICES.md`.
