# nerd/pipeline/tasks/mut_count.py
"""
Task for mutation counting using pluggable tools (e.g., shapemapper).
Parent samples only (no derived materialization yet).
"""

from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List

from .base import Task, TaskContext
from nerd.utils.logging import get_logger
from nerd.pipeline.runners.local import LocalRunner
from nerd.pipeline.tasks.derived import (
    DerivedMaterializer,
    SubsampleMaterializer,
    FilterSingleHitMaterializer,
)

log = get_logger(__name__)


class MutCountTask(Task):
    """
    A task that invokes a mutation counting tool via a runner and imports
    resulting data. This initial version handles only parent samples specified
    by name in the config.
    """
    name = "mut_count"
    scope_kind = "sample"  # per-sample operations; aggregated under one task

    def __init__(self) -> None:
        super().__init__()
        self._stage_in: List[Dict[str, str]] = []  # list of {src, dst} relative to remote workdir
        self._stage_out_extra: List[str] = []

    def prepare(self, cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if self.name not in cfg:
            raise ValueError(f"Configuration must contain a '{self.name}' section.")

        mc = cfg[self.name]
        # Minimal required fields
        plugin = mc.get("plugin")
        if not plugin:
            raise ValueError("mut_count.plugin is required (e.g., 'shapemapper').")

        # Accept either parent sample names under 'samples' or derived child_names under 'derived_samples'
        samples = list(mc.get("samples", []) or [])
        derived = list(mc.get("derived_samples", []) or [])
        if not samples and not derived:
            raise ValueError("mut_count requires 'samples' and/or 'derived_samples' listing sample names or child_names.")
        merged = samples + derived
        mc = {**mc, "samples": merged}

        # Resolve parent samples from DB now so command() can run everything; no derived support yet
        from nerd.db import api as db_api  # local import
        # We cannot access ctx here, so defer fq roots to command() using run_dir; only fetch DB rows now.
        return mc, {}

    def command(self, ctx: TaskContext, inputs: Dict[str, Any], params: Dict[str, Any]) -> Optional[str]:
        """
        Build a composite shell command that iterates over parent samples,
        creates per-sample output dirs, and invokes the tool (or placeholder).
        """
        from nerd.db import api as db_api
        from nerd.pipeline.plugins.mutcount import load_mutcount_plugin

        plugin_name = str(inputs.get("plugin")).strip().lower()
        tool_cfg = inputs.get("tool", {}) or {}
        threads = inputs.get("params", {}).get("n_proc") or inputs.get("threads") or 1

        # Plugin and options
        plugin = load_mutcount_plugin(
            plugin_name,
            bin_path=tool_cfg.get("bin"),
            version=tool_cfg.get("version"),
        )
        options = {
            "amplicon": bool(inputs.get("amplicon", True)),
            "dms_mode": bool(inputs.get("dms_mode", False)),
            "output_N7": bool(inputs.get("output_N7", False)),
            "output_parsed_mutations": bool(inputs.get("output_parsed_mutations", False)),
            "per_read_histograms": bool(inputs.get("per_read_histograms", False)),
        }
        dry_run = bool(inputs.get("dry_run", False))

        sample_names: List[str] = inputs.get("samples", [])
        if not sample_names:
            return None

        label_dir = Path(ctx.output_dir) / ctx.label

        def _nt_rows_for_construct(cid: int) -> List[Dict[str, Any]]:
            rows = ctx.db.execute(
                "SELECT site, base, base_region FROM nucleotides WHERE construct_id = ? ORDER BY site",
                (cid,),
            ).fetchall()
            return [dict(r) for r in rows]

        def _sample_row_by_name(name: str):
            rows = ctx.db.execute(
                "SELECT * FROM sequencing_samples WHERE sample_name = ?",
                (name,),
            ).fetchall()
            if not rows:
                return None
            if len(rows) > 1:
                raise ValueError(f"Sample name '{name}' is ambiguous across sequencing runs.")
            return rows[0]

        def _derived_by_child(child_name: str):
            row = ctx.db.execute(
                "SELECT * FROM derived_samples WHERE child_name = ?",
                (child_name,),
            ).fetchone()
            return row

        def _construct_id_for_sample_id(sid: int) -> Optional[int]:
            row = ctx.db.execute(
                "SELECT construct_id FROM probing_reactions WHERE s_id = ? ORDER BY rowid DESC LIMIT 1",
                (sid,),
            ).fetchone()
            return int(row[0]) if row and row[0] is not None else None

        cmds: List[str] = []
        self._stage_in = []
        self._stage_out_extra = [
            "artifacts/*/*_profile.txt",
            "artifacts/*/*_profile.txtga",
        ]
        for name in sample_names:
            parent_srow = _sample_row_by_name(name)
            is_derived = False
            derived_row = None
            if parent_srow is None:
                # Try derived by child_name
                derived_row = _derived_by_child(name)
                if derived_row is None:
                    raise ValueError(f"Unknown sample or derived child_name: {name}")
                is_derived = True
                parent_id = int(derived_row["parent_sample_id"]) if "parent_sample_id" in derived_row.keys() else int(derived_row[1])
                parent_srow = ctx.db.execute("SELECT * FROM sequencing_samples WHERE id = ?", (parent_id,)).fetchone()
                if parent_srow is None:
                    raise ValueError(f"Derived sample '{name}' refers to missing parent sample id={parent_id}")

            sid = int(parent_srow["id"]) if "id" in parent_srow.keys() else int(parent_srow[0])
            # Resolve parent R1/R2 local paths
            fq_dir = Path(str(parent_srow["fq_dir"]))
            if not fq_dir.is_absolute():
                fq_dir = label_dir / fq_dir
            r1 = fq_dir / str(parent_srow["r1_file"])
            r2 = fq_dir / str(parent_srow["r2_file"])

            cid = _construct_id_for_sample_id(sid)
            if cid is None:
                raise ValueError(f"Could not resolve construct for sample '{name}' (parent s_id={sid})")
            nt_rows = _nt_rows_for_construct(cid)

            # Build FASTA content locally to embed via heredoc on remote
            def _fasta_text(nt_rows: List[Dict[str, Any]], header: str = "target") -> str:
                seq_chars: List[str] = []
                for nt in nt_rows:
                    base = str(nt.get("base", "")).strip()
                    region = str(nt.get("base_region", "")).strip()
                    # Normalize to DNA alphabet for external tools (use T, not U)
                    b = base.upper()
                    if b == "U":
                        b = "T"
                    if region in {"0", "2"}:
                        b = b.lower()
                    seq_chars.append(b)
                return f">{header}\n{''.join(seq_chars)}\n"

            sample_dir = Path("artifacts") / name
            target_fa = sample_dir / "target.fa"

            # 1) ensure sample dir on remote
            cmds.append(f"mkdir -p {sample_dir}")
            # 2) write FASTA via heredoc
            fasta_text = _fasta_text(nt_rows, header=name)
            # Protect EOF and content; use single-quoted EOF to avoid shell interpolation
            heredoc = (
                f"cat > {target_fa} << 'EOF'\n" + fasta_text + "EOF\n"
            )
            cmds.append(heredoc)
            # 3) stage-in parent R1/R2 to remote sample dir
            out_dir = sample_dir  # write outputs under the sample directory
            remote_r1 = sample_dir / r1.name
            remote_r2 = sample_dir / r2.name
            self._stage_in.append({"src": str(r1), "dst": str(remote_r1)})
            self._stage_in.append({"src": str(r2), "dst": str(remote_r2)})

            # If derived, materialize child FASTQs on remote first
            use_r1 = remote_r1
            use_r2 = remote_r2
            if is_derived and derived_row is not None:
                import json as _json
                params_json = derived_row["params_json"] if "params_json" in derived_row.keys() else derived_row[6]
                try:
                    params = _json.loads(params_json) if params_json else {}
                except Exception:
                    params = {}
                kind = (derived_row["kind"] if "kind" in derived_row.keys() else derived_row[3]) or "subsample"
                # Select materializer
                if str(kind).lower() == "filter_singlehit":
                    max_mut = int(params.get("max_mutations", 1))
                    materializer: DerivedMaterializer = FilterSingleHitMaterializer(max_mutations=max_mut)
                else:
                    cmd_template = derived_row["cmd_template"] if "cmd_template" in derived_row.keys() else derived_row[5]
                    materializer = SubsampleMaterializer(cmd_template)

                # Minimal plugin opts propagated
                plugin_opts = {
                    "amplicon": bool(inputs.get("amplicon", True)),
                    "dms_mode": bool(inputs.get("dms_mode", False)),
                    "output_N7": bool(inputs.get("output_N7", False)),
                }
                out_r1, out_r2, prep_cmds, patterns = materializer.prepare(
                    sample_name=name,
                    parent_r1_remote=remote_r1,
                    parent_r2_remote=remote_r2,
                    sample_dir=sample_dir,
                    target_fa_remote=target_fa,
                    plugin=plugin,
                    plugin_opts=plugin_opts,
                    params=params,
                )
                if patterns:
                    self._stage_out_extra.extend([str(p) for p in patterns])
                cmds.extend(prep_cmds)
                use_r1 = out_r1
                use_r2 = out_r2

            shapecmd = plugin.command(
                sample_name=name,
                r1_path=Path(str(use_r1)),
                r2_path=Path(str(use_r2)),
                fasta_path=Path(str(target_fa)),
                out_dir=Path(str(out_dir)),
                options=options,
            )
            sep = "################################################################################"
            if not is_derived:
                # Add verification steps for non-derived cases (derived already logs these)
                cmds.append(f"echo '{sep}'")
                cmds.append("echo '# 3 - Verify staged FASTQ'")
                cmds.append(f"echo '{sep}'")
                cmds.append(f"ls -lh {use_r1} {use_r2} || true")
                cmds.append(f"echo '{sep}'")
                cmds.append("echo '# 4 - Verify created FASTA'")
                cmds.append(f"echo '{sep}'")
                cmds.append(f"head -n 2 {target_fa} || true")

            # Step 5: show/create shapemapper command
            cmds.append(f"echo '{sep}'")
            cmds.append("echo '# 5 - Create shapemapper' ")
            cmds.append(f"echo '{sep}'")
            cmds.append(f"echo 'Command: {shapecmd}'")

            # Step 6: run shapemapper (or stage in dry-run)
            cmds.append(f"echo '{sep}'")
            cmds.append("echo '# 6 - Run shapemapper'")
            cmds.append(f"echo '{sep}'")
            if dry_run:
                cmds.append(f"echo '[Step 1] Verifying staged FASTQs for {name}'")
                cmds.append(f"ls -lh {remote_r1} {remote_r2} || true")
                cmds.append(f"echo '[Step 2] FASTA created for {name} at {target_fa}'")
                cmds.append(f"head -n 2 {target_fa} || true")
                run_script = sample_dir / "run_shapemapper.sh"
                cmds.append(f"echo '[Step 3] Creating shapemapper script for {name}: {run_script}'")
                mk_script = (
                    f"cat > {run_script} << 'EOSH'\n#!/usr/bin/env bash\nset -euo pipefail\n{shapecmd}\nEOSH\n"
                )
                cmds.append(mk_script)
                cmds.append(f"chmod +x {run_script}")
                cmds.append(f"echo '[Step 4] Would run shapemapper via {run_script} (skipped in dry_run)'")
            else:
                cmds.append(shapecmd)

        # Use newlines between commands; 'set -e' in the job script ensures abort on failure.
        return "\n".join(cmds)

    def stage_in_pairs(self) -> List[Dict[str, str]]:
        """Return stage-in file mappings prepared during command() build."""
        return list(self._stage_in or [])

    def stage_out_patterns(self) -> Optional[List[str]]:
        return list(dict.fromkeys(self._stage_out_extra)) if self._stage_out_extra else []

    def _resolve_fastqs(self, ctx: TaskContext, row) -> Tuple[Path, Path]:
        label_dir = Path(ctx.output_dir) / ctx.label
        fq_dir = Path(row["fq_dir"]) if isinstance(row["fq_dir"], str) else Path(str(row["fq_dir"]))
        if not fq_dir.is_absolute():
            fq_dir = label_dir / fq_dir
        r1 = fq_dir / row["r1_file"]
        r2 = fq_dir / row["r2_file"]
        return r1, r2

    def scope_id(self, ctx: Optional[TaskContext], inputs: Any) -> Optional[int]:
        """Return parent sequencing_samples.id if exactly one sample/child is requested; else None."""
        try:
            if ctx is None or inputs is None:
                return None
            sample_names: List[str] = inputs.get("samples", []) if isinstance(inputs, dict) else []
            if len(sample_names) != 1:
                return None
            name = sample_names[0]
            row = ctx.db.execute(
                "SELECT id FROM sequencing_samples WHERE sample_name = ?",
                (name,),
            ).fetchall()
            if len(row) == 1:
                return int(row[0][0])
            # Try derived child name → parent id
            d = ctx.db.execute(
                "SELECT parent_sample_id FROM derived_samples WHERE child_name = ?",
                (name,),
            ).fetchone()
            if d is not None:
                return int(d[0])
            return None
        except Exception:
            return None

    def task_tool(self, inputs: Any) -> Optional[str]:
        try:
            if isinstance(inputs, dict):
                val = inputs.get("plugin")
                if val is not None:
                    s = str(val).strip()
                    return s or None
        except Exception:
            pass
        return None

    def task_tool_version(self, inputs: Any) -> Optional[str]:
        try:
            if isinstance(inputs, dict):
                tool = inputs.get("tool") or {}
                ver = tool.get("version")
                if ver not in (None, ""):
                    return str(ver)
        except Exception:
            pass
        return None

    def consume_outputs(self, ctx: TaskContext, inputs: Dict[str, Any], params: Dict[str, Any], run_dir: Path):
        """
        Execute mutation counting for the selected parent samples.
        """
        from nerd.pipeline.plugins.mutcount import load_mutcount_plugin

        plugin_name = inputs.get("plugin")
        tool_cfg = inputs.get("tool", {}) or {}
        plugin = load_mutcount_plugin(plugin_name, bin_path=tool_cfg.get("bin"), version=tool_cfg.get("version"))

        sample_names: List[str] = inputs.get("samples", [])
        if not sample_names:
            log.warning("mut_count.consume_outputs called with no samples.")
            return

        artifacts_dir = run_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        found = 0
        for name in sample_names:
            sdir = artifacts_dir / name
            prof = plugin.find_profile(sdir)
            if prof and prof.exists():
                found += 1
                log.info("Found ShapeMapper profile for %s at %s", name, prof)
            else:
                # Fallback: sometimes stage-out placed the profile at run_dir root.
                fallback_found = False
                try:
                    from glob import glob as _glob
                    pats = [
                        str(artifacts_dir / name / f"{name}*_profile.txt"),
                        str(artifacts_dir / name / f"{name}*_profile.txtga"),
                        str(run_dir / f"{name}*_profile.txt"),
                        str(run_dir / f"{name}*_profile.txtga"),
                    ]
                    for p in pats:
                        m = sorted(_glob(p))
                        if m:
                            prof = Path(m[0])
                            log.info("Fallback found profile for %s at %s", name, prof)
                            found += 1
                            fallback_found = True
                            break
                except Exception:
                    pass
                if not fallback_found:
                    log.warning("No ShapeMapper profile found for %s under %s", name, sdir)
                    try:
                        # Help debug by listing available txt files
                        txts = sorted([str(p.relative_to(run_dir)) for p in sdir.rglob("*.txt")])
                        if txts:
                            log.info("Found txt files under %s: %s", sdir, ", ".join(txts[:10]) + (" ..." if len(txts) > 10 else ""))
                        else:
                            log.info("No txt files present under %s", sdir)
                    except Exception:
                        pass

        log.info("mut_count completed. Profiles found for %d/%d samples.", found, len(sample_names))
