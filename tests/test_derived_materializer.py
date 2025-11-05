from pathlib import Path
import shutil
import subprocess
import sys
import pytest

from nerd.pipeline.tasks.derived import FilterSingleHitMaterializer


class DummyPlugin:
    def command(self, *, sample_name, r1_path, r2_path, fasta_path, out_dir, options=None):
        # Just return a sentinel so we can assert it shows up in commands[1]
        return (
            f"SHAPEMAPPER_CMD --name {sample_name} --target {fasta_path} "
            f"--out {out_dir} --R1 {r1_path} --R2 {r2_path}"
        )


def test_filter_singlehit_prepare_returns_expected_commands_and_patterns(tmp_path):
    plugin = DummyPlugin()
    mat = FilterSingleHitMaterializer(max_mutations=1)

    sample_name = "fourU_WT_65c_rep1_tp1__singlehit"
    sample_dir = tmp_path / "samples" / sample_name
    parent_r1 = sample_dir / "parent_R1.fastq.gz"
    parent_r2 = sample_dir / "parent_R2.fastq.gz"
    target_fa = sample_dir / "target.fa"

    # Inputs can be arbitrary Paths for this unit test (we don't execute commands)
    plugin_opts = {"amplicon": True, "dms_mode": False, "output_N7": False}

    out_r1, out_r2, commands, patterns = mat.prepare(
        sample_name=sample_name,
        parent_r1_remote=parent_r1,
        parent_r2_remote=parent_r2,
        sample_dir=sample_dir,
        target_fa_remote=target_fa,
        plugin=plugin,
        plugin_opts=plugin_opts,
        params={"max_mutations": 1},
    )

    # Derived outputs
    assert out_r1 == sample_dir / "derived_R1.fastq"
    assert out_r2 == sample_dir / "derived_R2.fastq"

    # Commands contain the expected stages regardless of logging wrappers
    assert any("Run parent scan" in cmd for cmd in commands)
    assert any(cmd.startswith("SHAPEMAPPER_CMD") for cmd in commands)
    awk_cmds = [cmd for cmd in commands if "awk -F" in cmd]
    assert awk_cmds, "parsed.mut filtering command missing"
    assert any("reads_singlehit.lst" in cmd for cmd in awk_cmds)
    seqtk_cmds = [cmd for cmd in commands if "seqtk subseq" in cmd]
    assert seqtk_cmds, "seqtk subseq command missing"
    assert all(str(parent_r1) in cmd or str(parent_r2) in cmd for cmd in seqtk_cmds)

    # Stage-out patterns include parsed.mut* and the list file
    assert str(sample_dir / "parent_scan/*_parsed.mut*") in patterns
    assert str(sample_dir / "reads_singlehit.lst") in patterns


def _read_ids_lt(mut_file: Path, max_mut: int = 1) -> list:
    ids = []
    with mut_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 3:
                continue
            read_id = cols[1]
            # Try candidate columns for the 0/1 mask; prefer the longest valid mask
            candidates = []
            for idx in (8, 9, 10, 11):
                if idx < len(cols):
                    s = cols[idx]
                    if s and all(c in "01" for c in s):
                        candidates.append(s)
            if not candidates:
                continue
            mask = max(candidates, key=len)
            muts = mask.count("1")
            if muts <= max_mut:
                ids.append(read_id)
    return ids


def test_parsed_mut_singlehit_extraction_nonempty(tmp_path):
    # Use a repo-relative parsed.mut test file that exists in this repo
    mut_file = Path(
        "tests/data/samples/fourU_WT_65c_rep1_tp1__subsample-n10000-s42/"
        "fourU_WT_65c_rep1_tp1__subsample-n10000-s42_Modified_"
        "fourU_WT_65c_rep1_tp1__subsample-n10000-s42_parsed.mut"
    )
    assert mut_file.is_file(), f"missing test parsed.mut at {mut_file}"

    ids = _read_ids_lt(mut_file, max_mut=1)
    assert isinstance(ids, list)
    assert len(ids) > 0, "No single-hit reads found; check mask column or file contents"

    # Optionally write out and inspect
    lst = tmp_path / "reads_singlehit.lst"
    lst.write_text("\n".join(ids) + "\n", encoding="utf-8")
    assert lst.stat().st_size > 0
    # Show where the file is and sample content when running with -s
    preview = "\n".join(ids[:5])
    print(f"[singlehit] wrote: {lst} (n={len(ids)})\n{preview}")


@pytest.mark.skipif(shutil.which("awk") is None, reason="awk not available")
def test_awk_on_parsed_mut_which_column_yields_ids(tmp_path):
    mut_file = Path(
        "tests/data/samples/fourU_WT_65c_rep1_tp1__subsample-n10000-s42/"
        "fourU_WT_65c_rep1_tp1__subsample-n10000-s42_Modified_"
        "fourU_WT_65c_rep1_tp1__subsample-n10000-s42_parsed.mut"
    )
    assert mut_file.is_file(), f"missing test parsed.mut at {mut_file}"
    lst9 = tmp_path / "reads9.lst"
    lst10 = tmp_path / "reads10.lst"
    awk9 = r"""awk -F'	' '{ s=0; for(i=1;i<=length($9); i++) if(substr($9,i,1)=="1") s++; if (s<2) print $2 }'"""
    awk10 = r"""awk -F'	' '{ s=0; for(i=1;i<=length($10); i++) if(substr($10,i,1)=="1") s++; if (s<2) print $2 }'"""
    subprocess.run(["bash", "-lc", f"{awk9} '{mut_file}' > '{lst9}'"], check=True, text=True)
    subprocess.run(["bash", "-lc", f"{awk10} '{mut_file}' > '{lst10}'"], check=True, text=True)
    n9 = len(lst9.read_text().splitlines()) if lst9.exists() else 0
    n10 = len(lst10.read_text().splitlines()) if lst10.exists() else 0
    assert n9 > 0 or n10 > 0, "Neither $9 nor $10 yielded reads; check file format"
    # Print paths, counts, and small previews when running with -s
    head9 = "\n".join(lst9.read_text().splitlines()[:5]) if lst9.exists() else ""
    head10 = "\n".join(lst10.read_text().splitlines()[:5]) if lst10.exists() else ""
    print(f"[awk] $9 -> {lst9} (n={n9})\n{head9}")
    print(f"[awk] $10 -> {lst10} (n={n10})\n{head10}")
