"""
ShapeMapper plugin for mutation counting.

Example command:
  shapemapper --name example2 --target TPP.fa --out TPP_shapemap \
              --amplicon --modified --R1 <r1> --R2 <r2> \
              --dms --N7 --bypass_filters

Flags mapping (from YAML options → CLI):
- output_N7: true  →  --N7 --bypass_filters
- dms_mode: true   →  --dms
- amplicon: true   →  --amplicon
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
import glob

from .base import MutCountPlugin


class ShapeMapperPlugin(MutCountPlugin):
    @staticmethod
    def default_binary() -> str:
        return "shapemapper"

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

        bin_path = self.bin_path or "shapemapper"
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
