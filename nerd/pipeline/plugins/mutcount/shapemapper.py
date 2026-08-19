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
import shlex

from .base import MutCountPlugin
from nerd.containers import render_container_exec, shapemapper_container_spec, RuntimeInfo


class ShapeMapperPlugin(MutCountPlugin):
    def __init__(self, bin_path: Optional[str] = None, version: Optional[str] = None,
                 container_execution: Optional[Dict] = None):
        super().__init__(bin_path=bin_path, version=version)
        self.container_execution = container_execution

    def container_spec(self, tool_cfg: Optional[Dict] = None):
        return shapemapper_container_spec(tool_cfg)

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

        if self.container_execution:
            info = self.container_execution
            runtime = RuntimeInfo(**info["runtime"])
            bind_paths = [str(info["workdir"])]
            for path in (r1_path, r2_path, fasta_path, out_dir):
                if path.is_absolute():
                    bind_paths.append(str(path if path.is_dir() else path.parent))
            parts[0] = str(info.get("executable") or "shapemapper")
            return render_container_exec(
                runtime, str(info["sif_path"]), parts, bind_paths, str(info["workdir"])
            )
        return shlex.join(str(item) for item in parts)

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
