"""
Base interface for mutation-count plugins.

Plugins are responsible for:
- Emitting a concrete shell command to run the tool for a given sample
- Generating any auxiliary inputs required by the tool (e.g., target FASTA)
- Locating and parsing the primary output relevant to mutation rates

This interface is intentionally light to avoid coupling to Task internals.
The Task constructs paths and collects DB/config context, then calls into a
plugin to get the command and to process outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence
import abc


class MutCountPlugin(abc.ABC):
    """Abstract base class for mutation counting plugins."""

    def __init__(self, bin_path: Optional[str] = None, version: Optional[str] = None):
        self.bin_path = bin_path or self.default_binary()
        self.version = version

    @staticmethod
    def default_binary() -> str:
        """Return the default CLI executable name for the tool."""
        return ""

    # ---- Inputs/helpers ----
    @staticmethod
    def nts_to_fasta(nt_rows: Sequence[Dict], fasta_path: Path, header: str = "target") -> Path:
        """
        Write a FASTA file for a construct from `nt_rows`.

        - base_region: 1 = target (uppercase), 0/2 = primers (lowercase)
        - base: expected to be a single character in A/C/G/T/U
        """
        seq_chars: List[str] = []
        for nt in nt_rows:
            base = str(nt.get("base", "")).strip()
            region = str(nt.get("base_region", "")).strip()
            if not base:
                raise ValueError(f"Invalid nt row (missing base): {nt}")
            if region not in {"0", "1", "2"}:
                raise ValueError(f"Invalid base_region in nt row: {nt}")
            # Normalize to DNA alphabet for external tools (use T, not U)
            b = base.upper()
            if b == "U":
                b = "T"
            if region in {"0", "2"}:
                b = b.lower()
            seq_chars.append(b)
        sequence = "".join(seq_chars) + "\n"
        fasta = f">{header}\n{sequence}"
        fasta_path.parent.mkdir(parents=True, exist_ok=True)
        fasta_path.write_text(fasta)
        return fasta_path

    # ---- Execution ----
    @abc.abstractmethod
    def command(
        self,
        *,
        sample_name: str,
        r1_path: Path,
        r2_path: Path,
        fasta_path: Path,
        out_dir: Path,
        options: Optional[Dict] = None,
    ) -> str:
        """
        Return a shell command that executes the tool for the given sample.
        Implementations should not create directories; the Task ensures `out_dir` exists.
        """
        raise NotImplementedError

    # ---- Outputs ----
    @abc.abstractmethod
    def find_profile(self, out_dir: Path) -> Optional[Path]:
        """Locate the primary mutation-rate table produced by the tool, if present."""
        raise NotImplementedError

    def parse_profile(self, profile_path: Path) -> List[Dict]:
        """
        Parse the mutation-rate table. Default is a generic CSV/TSV reader that
        returns a list of row dicts. Plugins may override for richer parsing.
        """
        import csv

        with profile_path.open("r", newline="", encoding="utf-8") as f:
            # Try to sniff delimiter; fallback to comma, then tab
            sample = f.read(8192)
            f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
            except Exception:
                dialect = csv.excel
                if "\t" in sample and "," not in sample:
                    dialect = csv.excel_tab
            reader = csv.DictReader(f, dialect=dialect)
            return [dict(row) for row in reader]
