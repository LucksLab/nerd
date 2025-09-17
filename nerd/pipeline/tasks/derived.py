"""
Derived sample materialization strategies used by MutCountTask.

Each materializer returns the remote paths to use for R1/R2 and a list of
shell commands to produce them inside the remote working directory, plus any
extra stage-out patterns to collect.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional


class DerivedMaterializer:
    def prepare(
        self,
        *,
        sample_name: str,
        parent_r1_remote: Path,
        parent_r2_remote: Path,
        sample_dir: Path,
        target_fa_remote: Path,
        plugin,
        plugin_opts: Dict,
        params: Optional[Dict] = None,
    ) -> Tuple[Path, Path, List[str], List[str]]:
        raise NotImplementedError


class SubsampleMaterializer(DerivedMaterializer):
    """Generic materializer using a cmd_template with placeholders."""

    def __init__(self, cmd_template: str):
        self.cmd_template = cmd_template

    def prepare(
        self,
        *,
        sample_name: str,
        parent_r1_remote: Path,
        parent_r2_remote: Path,
        sample_dir: Path,
        target_fa_remote: Path,
        plugin,
        plugin_opts: Dict,
        params: Optional[Dict] = None,
    ) -> Tuple[Path, Path, List[str], List[str]]:
        out_r1 = sample_dir / "derived_R1.fastq"
        out_r2 = sample_dir / "derived_R2.fastq"
        mapping: Dict[str, str] = {
            "R1": str(parent_r1_remote),
            "R2": str(parent_r2_remote),
            "OUT_R1": str(out_r1),
            "OUT_R2": str(out_r2),
        }
        for k, v in (params or {}).items():
            mapping[str(k)] = str(v)
        try:
            cmd = str(self.cmd_template).format(**mapping)
        except Exception:
            cmd = str(self.cmd_template)
        sep = "################################################################################"
        count_parent_r1 = (
            f"orig_lines=$( ( [[ '{parent_r1_remote}' == *.gz ]] && zcat '{parent_r1_remote}' || cat '{parent_r1_remote}' ) | wc -l ); orig_rec=$((orig_lines/4))"
        )
        count_parent_r2 = (
            f"orig2_lines=$( ( [[ '{parent_r2_remote}' == *.gz ]] && zcat '{parent_r2_remote}' || cat '{parent_r2_remote}' ) | wc -l ); orig2_rec=$((orig2_lines/4))"
        )
        count_derived = (
            f"der1_rec=$(( $(wc -l < {out_r1}) / 4 )); der2_rec=$(( $(wc -l < {out_r2}) / 4 ))"
        )
        commands = [
            f"echo '{sep}'",
            f"echo '# 2 - Derive records via subsample and summarize'",
            f"echo '{sep}'",
            cmd,
            count_parent_r1,
            count_parent_r2,
            count_derived,
            f"echo '[derive:{sample_name}] Original records (R1,R2): ' $orig_rec ' ' $orig2_rec",
            f"echo '[derive:{sample_name}] Derived  records (R1,R2): ' $der1_rec ' ' $der2_rec",
            f"frac=$(awk -v der=\"$der1_rec\" -v orig=\"$orig_rec\" 'BEGIN{{if(orig>0) printf \"%.2f\", der/orig*100; else printf \"0.00\"}}'); echo '[derive:{sample_name}] Fraction kept (R1):' $frac ' %'",
            f"echo '{sep}'",
            f"echo '# 3 - Verify staged FASTQ (to be used)'",
            f"echo '{sep}'",
            f"ls -lh {out_r1} {out_r2} || true",
            f"echo '{sep}'",
            f"echo '# 4 - Verify created FASTA'",
            f"echo '{sep}'",
            f"head -n 2 {target_fa_remote} || true",
        ]
        return out_r1, out_r2, commands, []


class FilterSingleHitMaterializer(DerivedMaterializer):
    """
    Materializer that scans parent reads to produce parsed mutation files, extracts
    read IDs with < N mutations, and filters parent FASTQs via seqtk subseq.
    """

    def __init__(self, max_mutations: int = 1):
        self.max_mut = int(max_mutations)

    def prepare(
        self,
        *,
        sample_name: str,
        parent_r1_remote: Path,
        parent_r2_remote: Path,
        sample_dir: Path,
        target_fa_remote: Path,
        plugin,
        plugin_opts: Dict,
        params: Optional[Dict] = None,
    ) -> Tuple[Path, Path, List[str], List[str]]:
        # 1) Run a minimal ShapeMapper scan to get parsed mutations
        parent_out = sample_dir / "parent_scan"
        scan_cmd = plugin.command(
            sample_name=f"{sample_name}__parent",
            r1_path=parent_r1_remote,
            r2_path=parent_r2_remote,
            fasta_path=target_fa_remote,
            out_dir=parent_out,
            options={
                **plugin_opts,
                "output_parsed_mutations": True,
                "per_read_histograms": False,
            },
        )
        # 2) Extract <N mutation reads into a list; prefer *_parsed.mut or *_parsed.mutga
        lst = sample_dir / "reads_singlehit.lst"
        # Sum 1s in field 9 (0/1 string), output field 2 (read ID) when sum < threshold
        awk = (
            "awk -F'\t' '{ s=0; for(i=1;i<=length($9); i++) if(substr($9,i,1)==\"1\") s++; "
            f"if (s<{self.max_mut + 1}) print $2 }}'"
        )
        parse_cmd = (
            # Enable nullglob so unmatched globs expand to nothing (not the literal pattern)
            "shopt -s nullglob; "
            # Pick the first matching parsed mutations file (mut or mutga)
            f"MUT=; for f in {parent_out}/*_parsed.mut {parent_out}/*_parsed.mutga; do MUT=\"$f\"; break; done; "
            'if [ -z "$MUT" ]; then echo "No parsed mutations file found in '"'"' + str(parent_out) + '"'"'" >&2; exit 1; fi; '
            # Extract read ids with < max mutations
            f"{awk} \"$MUT\" > {lst}; "
            # Log how many ids were selected for easier debugging
            f'echo "singlehit IDs: $(wc -l < {lst}) from $(basename \"$MUT\")"'
        )
        # 3) Filter with seqtk
        out_r1 = sample_dir / "derived_R1.fastq"
        out_r2 = sample_dir / "derived_R2.fastq"
        filter_cmd = (
            f"seqtk subseq {parent_r1_remote} {lst} > {out_r1}\n"
            f"seqtk subseq {parent_r2_remote} {lst} > {out_r2}"
        )

        # Summaries and nice headings
        sep = "################################################################################"
        count_parent_r1 = (
            f"orig_lines=$( ( [[ '{parent_r1_remote}' == *.gz ]] && zcat '{parent_r1_remote}' || cat '{parent_r1_remote}' ) | wc -l ); orig_rec=$((orig_lines/4))"
        )
        count_parent_r2 = (
            f"orig2_lines=$( ( [[ '{parent_r2_remote}' == *.gz ]] && zcat '{parent_r2_remote}' || cat '{parent_r2_remote}' ) | wc -l ); orig2_rec=$((orig2_lines/4))"
        )
        count_derived = (
            f"der1_rec=$(( $(wc -l < {out_r1}) / 4 )); der2_rec=$(( $(wc -l < {out_r2}) / 4 ))"
        )

        commands = [
            f"echo '{sep}'",
            f"echo '# 1 - Run parent scan'",
            f"echo '{sep}'",
            scan_cmd,
            f"echo '{sep}'",
            f"echo '# 2 - Derive single-hit reads and summarize'",
            f"echo '{sep}'",
            parse_cmd,
            f"echo '[derive:{sample_name}] First 5 single-hit IDs:'",
            f"head -n 5 {lst} || true",
            filter_cmd,
            count_parent_r1,
            count_parent_r2,
            count_derived,
            f"echo '[derive:{sample_name}] Original records (R1,R2): ' $orig_rec ' ' $orig2_rec",
            f"echo '[derive:{sample_name}] Derived  records (R1,R2): ' $der1_rec ' ' $der2_rec",
            f"frac=$(awk -v der=\"$der1_rec\" -v orig=\"$orig_rec\" 'BEGIN{{if(orig>0) printf \"%.2f\", der/orig*100; else printf \"0.00\"}}'); echo '[derive:{sample_name}] Fraction kept (R1):' $frac ' %'",
            f"echo '{sep}'",
            f"echo '# 3 - Verify staged FASTQ (to be used)'",
            f"echo '{sep}'",
            f"ls -lh {out_r1} {out_r2} || true",
            f"echo '{sep}'",
            f"echo '# 4 - Verify created FASTA'",
            f"echo '{sep}'",
            f"head -n 2 {target_fa_remote} || true",
        ]
        patterns = [str(parent_out / "*_parsed.mut*"), str(lst)]
        return out_r1, out_r2, commands, patterns
