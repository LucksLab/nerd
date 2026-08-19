# Third-party notices for NERD ShapeMapper2 v2.3

This image packages the unmodified ShapeMapper2 v2.3 release together with the
third-party programs and libraries that release supplies. Each component
remains under its own license. Inclusion in one container does not replace
those licenses with the license of NERD or ShapeMapper.

The complete machine-generated inventories are `sbom.spdx.json` and
`conda-packages.tsv`. `copyleft-conda-packages.tsv` is a focused review list
of 64 records whose metadata mentions GPL, LGPL, AGPL, EPL, MPL, Artistic, or
CDDL terms. Exact binary and source checksums for prominent components are in
`source-manifest.tsv`.

## Prominent components

| Component | Version | License and retained notice |
| --- | --- | --- |
| ShapeMapper2 | v2.3; runtime 2.3.0 | MIT; `LICENSES/ShapeMapper-MIT.txt` |
| Bowtie2 | 2.3.4.3 | GPLv3; `LICENSES/GPL-3.0.txt` |
| STAR | 2.5.2a | GPLv3; `LICENSES/GPL-3.0.txt` |
| Ghostscript | 9.25 | Official source is AGPLv3; `LICENSES/AGPL-3.0.txt` |
| BBMap/BBMerge | 37.78 | UC/LBNL permissive license; `LICENSES/BBMap-UC-LBL.txt` |
| pv | 1.6.20 | Artistic License 2.0; `LICENSES/Artistic-2.0.txt` |
| Graphviz | 7.1.0 | Eclipse Public License 1.0; `LICENSES/Graphviz-EPL-1.0.txt` |
| Azul Zulu OpenJDK | 8.0.112 / Zulu 8.19.0.1 | GPLv2 with Classpath Exception and accompanying exceptions/notices; see `LICENSES/OpenJDK-*` |
| Miniconda/Conda | Conda 4.3.21 | BSD-3-Clause plus individual package licenses; `LICENSES/Miniconda.txt` |
| Debian base packages | pinned bookworm-slim image | Mixed licenses; package copyright files remain under `/usr/share/doc` |

Ghostscript requires special explanation. The official Ghostscript 9.25
source archive contains the GNU Affero GPL v3 text, while the prebuilt binary
archive embedded by ShapeMapper contains a GNU GPL v3 `COPYING` file. Both are
retained (`AGPL-3.0.txt` and
`Ghostscript-bundled-COPYING-GPL-3.0.txt`), and the redistribution package
conservatively identifies Ghostscript 9.25 as AGPL-covered.

The Zulu distribution's historical source-offer text identifies build code
`Zulu 8.19.0.1 d0cf8daf3adb`. Its three-year offer is not relied on for NERD's
redistribution. Until complete corresponding source for that exact build is
obtained or the runtime is replaced with a source-traceable OpenJDK build, the
container is not ready for public binary distribution.

## Source availability

For a public release, the files fetched and checksum-verified by
`fetch-compliance-sources.sh` should be retained as immutable release
artifacts beside the image. Upstream links alone are discovery locations, not
a promise that NERD will keep required source available.

No project or institution named by a third-party license endorses NERD or this
container. All components are supplied without additional warranty.
