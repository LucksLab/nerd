"""Fillers: the composable ways a sample sheet gets populated.

Each filler writes a defined set of columns with a defined provenance, and
respects the precedence rules in model.py -- so re-running one never
destroys work of higher provenance (in practice: never destroys manual
edits).

  fastq_scan   -> sample_name, fq_dir, r1_file, r2_file   (origin: fastq)
  pattern_fill -> whatever the pattern's tokens map to     (origin: pattern)
  batch_fill   -> one column across selected rows          (origin: batch)
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .model import BATCH, FASTQ, PATTERN, SAMPLE_COLUMNS, Row, Sheet
from .pattern import CompiledPattern, TokenRegistry, compile_pattern, parse_name

FASTQ_SUFFIXES = (".fastq.gz", ".fq.gz", ".fastq", ".fq")

# Read marker inside an Illumina-style filename. The R1/R2 must be its own
# delimited chunk -- bounded by . or _ on the left and by . or _ or
# end-of-stem on the right -- so a construct named "R1x" in the stem is not
# mistaken for a read marker.
_READ_MARKER = re.compile(r"[._](R[12])(?=[._]|$)", re.IGNORECASE)

# Illumina sample-index suffix: everything from _S<n> onward is machine
# bookkeeping, not part of the biological sample name. Verified against the
# real sheet: 001-...-30_S1_L001_R1_001.fastq.gz -> 001-...-30
_INDEX_SUFFIX = re.compile(r"_S\d+(?=[._]|$)")


def _strip_fastq_ext(name: str) -> str:
    lowered = name.lower()
    for suffix in FASTQ_SUFFIXES:
        if lowered.endswith(suffix):
            return name[: -len(suffix)]
    return name


def read_number(filename: str) -> Optional[int]:
    """Return 1 or 2 if the filename carries an R1/R2 marker."""
    match = _READ_MARKER.search(filename)
    if not match:
        return None
    return int(match.group(1)[1])


def pair_key(filename: str) -> str:
    """Collapse a fastq filename to a key shared by its R1/R2 partner."""
    stem = _strip_fastq_ext(filename)
    return _READ_MARKER.sub("", stem, count=1)


def sample_name_from_fastq(filename: str) -> str:
    """Derive the biological sample name from a fastq filename."""
    stem = pair_key(os.path.basename(filename))
    split = _INDEX_SUFFIX.split(stem, maxsplit=1)
    candidate = split[0] if split and split[0] else stem
    return candidate.rstrip("._-")


def parse_listing(text: str) -> List[str]:
    """Accept a pasted directory listing and return fastq basenames.

    Tolerates `ls -l` output, full paths, and blank lines, so a user can
    paste the result of running `ls` on Quest without cleaning it up.
    """
    names: List[str] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        candidate = line.split()[-1] if " " in line else line
        candidate = os.path.basename(candidate)
        if candidate.lower().endswith(FASTQ_SUFFIXES):
            names.append(candidate)
    return names


def pair_fastqs(filenames: Sequence[str]) -> Tuple[List[Dict[str, str]], List[str]]:
    """Group filenames into R1/R2 pairs.

    Returns (pairs, unpaired). Each pair is
    {sample_name, r1_file, r2_file}; order follows first appearance so the
    sheet comes out in the same order as the listing.
    """
    groups: Dict[str, Dict[str, str]] = {}
    order: List[str] = []
    unpaired: List[str] = []

    for filename in filenames:
        base = os.path.basename(filename)
        read = read_number(base)
        if read is None:
            unpaired.append(base)
            continue
        key = pair_key(base)
        if key not in groups:
            groups[key] = {}
            order.append(key)
        groups[key]["r%d_file" % read] = base

    pairs: List[Dict[str, str]] = []
    for key in order:
        entry = groups[key]
        r1, r2 = entry.get("r1_file", ""), entry.get("r2_file", "")
        if not r1 or not r2:
            unpaired.extend(v for v in (r1, r2) if v)
            continue
        pairs.append({
            "sample_name": sample_name_from_fastq(r1),
            "r1_file": r1,
            "r2_file": r2,
        })
    return pairs, unpaired


def scan_directory(directory: str) -> List[str]:
    path = Path(directory).expanduser()
    if not path.is_dir():
        raise FileNotFoundError("Not a directory: %s" % directory)
    return sorted(
        p.name for p in path.iterdir()
        if p.is_file() and p.name.lower().endswith(FASTQ_SUFFIXES)
    )


def fastq_scan(
    sheet: Sheet,
    filenames: Sequence[str],
    fq_dir: str,
    fq_source: str = "local",
    replace: bool = True,
) -> Dict[str, Any]:
    """Populate the sheet from a fastq listing.

    `replace=True` rebuilds the row set (the normal case: you are starting
    from a sequencing run). `replace=False` appends, for a second run.
    """
    pairs, unpaired = pair_fastqs(filenames)
    if replace:
        sheet.clear()

    for pair in pairs:
        sheet.add_row(
            {
                "sample_name": pair["sample_name"],
                "fq_source": fq_source,
                "fq_dir": fq_dir,
                "r1_file": pair["r1_file"],
                "r2_file": pair["r2_file"],
            },
            origin=FASTQ,
        )

    return {
        "added": len(pairs),
        "unpaired": unpaired,
        "total_files": len(filenames),
    }


def pattern_fill(
    sheet: Sheet,
    pattern: str,
    registry: TokenRegistry,
    uids: Optional[Iterable[int]] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Parse each row's sample_name and write the tokens' target columns.

    Cells edited by hand (or set by batch fill) are left alone unless
    `force` is set, so iterating on a pattern is non-destructive.
    """
    compiled = compile_pattern(pattern, registry)
    targets = set(uids) if uids is not None else None

    matched = unmatched = 0
    skipped_cells = 0
    columns_written: set = set()

    for row in sheet.rows:
        if targets is not None and row.uid not in targets:
            continue
        name = str(row.get("sample_name") or "")
        if not name:
            row.unmatched = True
            unmatched += 1
            continue
        parsed = parse_name(name, compiled, registry)
        if not parsed["matched"]:
            row.unmatched = True
            unmatched += 1
            continue
        row.unmatched = False
        matched += 1
        # Tokens with no target column (notably [tp_num]) are kept as row
        # context for the reaction-group layer to use.
        for token_name, raw_value in parsed["raw"].items():
            spec = registry.get(token_name)
            if spec is not None and not spec.maps_to:
                row.context[token_name] = spec.apply(raw_value)
        for column, value in parsed["values"].items():
            if column not in SAMPLE_COLUMNS:
                continue
            if row.set(column, value, PATTERN, force=force):
                columns_written.add(column)
            else:
                skipped_cells += 1

    return {
        "matched": matched,
        "unmatched": unmatched,
        "columns_written": sorted(columns_written),
        "protected_cells": skipped_cells,
        "warnings": compiled.warnings,
    }


def batch_fill(
    sheet: Sheet,
    column: str,
    value: Any,
    uids: Optional[Iterable[int]] = None,
    force: bool = True,
) -> Dict[str, Any]:
    """Set one column across selected rows (all rows if uids is None).

    This is the workhorse for columns that are constant across a run --
    probe, RT, done_by, probe_concentration, sequencing_run_name -- which
    in real sheets have only a handful of distinct values across
    thousands of rows.
    """
    if column not in SAMPLE_COLUMNS:
        raise ValueError("Unknown column: %s" % column)
    targets = set(uids) if uids is not None else None
    written = 0
    for row in sheet.rows:
        if targets is not None and row.uid not in targets:
            continue
        if row.set(column, value, BATCH, force=force):
            written += 1
    return {"written": written, "column": column}
