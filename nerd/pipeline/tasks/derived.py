"""
Derived sample materialization strategies used by MutCountTask.

Each materializer returns the remote paths to use for R1/R2 and a list of
shell commands to produce them inside the remote working directory, plus any
extra stage-out patterns to collect.
"""

from __future__ import annotations

from pathlib import Path
import shlex
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
        q = lambda value: shlex.quote(str(value))
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
            "orig_lines=$( ( [[ %s == *.gz ]] && zcat %s || cat %s ) | wc -l ); orig_rec=$((orig_lines/4))"
            % (q(parent_r1_remote), q(parent_r1_remote), q(parent_r1_remote))
        )
        count_parent_r2 = (
            "orig2_lines=$( ( [[ %s == *.gz ]] && zcat %s || cat %s ) | wc -l ); orig2_rec=$((orig2_lines/4))"
            % (q(parent_r2_remote), q(parent_r2_remote), q(parent_r2_remote))
        )
        count_derived = (
            "der1_rec=$(( $(wc -l < %s) / 4 )); der2_rec=$(( $(wc -l < %s) / 4 ))"
            % (q(out_r1), q(out_r2))
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
            "ls -lh %s %s || true" % (q(out_r1), q(out_r2)),
            f"echo '{sep}'",
            f"echo '# 4 - Verify created FASTA'",
            f"echo '{sep}'",
            "head -n 2 %s || true" % q(target_fa_remote),
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
        q = lambda value: shlex.quote(str(value))
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
            "MUT=; for f in %s/*_parsed.mut %s/*_parsed.mutga; do MUT=\"$f\"; break; done; "
            % (q(parent_out), q(parent_out)) +
            'if [ -z "$MUT" ]; then echo "No parsed mutations file found in '"'"' + str(parent_out) + '"'"'" >&2; exit 1; fi; '
            # Extract read ids with < max mutations
            "%s \"$MUT\" > %s; " % (awk, q(lst)) +
            # Log how many ids were selected for easier debugging
            'echo "singlehit IDs: $(wc -l < %s) from $(basename \"$MUT\")"' % q(lst)
        )
        # 3) Filter with seqtk
        out_r1 = sample_dir / "derived_R1.fastq"
        out_r2 = sample_dir / "derived_R2.fastq"
        filter_cmd = (
            "seqtk subseq %s %s > %s\n" % (q(parent_r1_remote), q(lst), q(out_r1)) +
            "seqtk subseq %s %s > %s" % (q(parent_r2_remote), q(lst), q(out_r2))
        )

        # Summaries and nice headings
        sep = "################################################################################"
        count_parent_r1 = (
            "orig_lines=$( ( [[ %s == *.gz ]] && zcat %s || cat %s ) | wc -l ); orig_rec=$((orig_lines/4))"
            % (q(parent_r1_remote), q(parent_r1_remote), q(parent_r1_remote))
        )
        count_parent_r2 = (
            "orig2_lines=$( ( [[ %s == *.gz ]] && zcat %s || cat %s ) | wc -l ); orig2_rec=$((orig2_lines/4))"
            % (q(parent_r2_remote), q(parent_r2_remote), q(parent_r2_remote))
        )
        count_derived = (
            "der1_rec=$(( $(wc -l < %s) / 4 )); der2_rec=$(( $(wc -l < %s) / 4 ))"
            % (q(out_r1), q(out_r2))
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
            "head -n 5 %s || true" % q(lst),
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
            "ls -lh %s %s || true" % (q(out_r1), q(out_r2)),
            f"echo '{sep}'",
            f"echo '# 4 - Verify created FASTA'",
            f"echo '{sep}'",
            "head -n 2 %s || true" % q(target_fa_remote),
        ]
        patterns = [str(parent_out / "*_parsed.mut*"), str(lst)]
        return out_r1, out_r2, commands, patterns
