"""
spats plugin for mutation counting. TO FIX

Example command:
  spats_tool run config

Flags mapping (from YAML options → CLI):

"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
import glob

from .base import MutCountPlugin


class SpatsPlugin(MutCountPlugin):
    @staticmethod
    def default_binary() -> str:
        return "spats_tool"

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
        opts = options or {}

        bin_path = self.bin_path or "spats_tool"
        name = sample_name
        target = str(fasta_path)
        out = str(out_dir)

        parts = [
            bin_path,
            "--name", name,
            "--target", target,
            "--out", out,
            "--modified",
            "--R1", str(r1_path),
            "--R2", str(r2_path),
        ]

        if opts.get("amplicon", False):
            parts.append("--amplicon")
        if opts.get("dms_mode", False):
            parts.append("--dms")
        if opts.get("output_N7", False):
            parts += ["--N7", "--bypass_filters"]
        # Enable additional outputs by default unless explicitly disabled
        if opts.get("per_read_histograms", False):
            parts.append("--per-read-histograms")
        if opts.get("output_parsed_mutations", False):
            parts.append("--output-parsed-mutations")

        return " ".join(parts)

    def find_profile(self, out_dir: Path) -> Optional[Path]:
        # Files end with _profile.txt (or _profile.txtga when N7 is enabled)
        patterns = [
            str(out_dir / "*_profile.txt"),
            str(out_dir / "*_profile.txtga"),
            str(out_dir / "**/*_profile.txt"),
            str(out_dir / "**/*_profile.txtga"),
        ]
        for pat in patterns:
            matches = sorted(glob.glob(pat, recursive=True))
            if matches:
                return Path(matches[0])
        return None

    def find_parsed_mut(self, out_dir: Path) -> Optional[Path]:
        # Files end with _parsed.mut or _parsed.mutga depending on N7
        patterns = [
            str(out_dir / "*_parsed.mut"),
            str(out_dir / "*_parsed.mutga"),
            str(out_dir / "**/*_parsed.mut"),
            str(out_dir / "**/*_parsed.mutga"),
        ]
        for pat in patterns:
            matches = sorted(glob.glob(pat, recursive=True))
            if matches:
                return Path(matches[0])
        return None
