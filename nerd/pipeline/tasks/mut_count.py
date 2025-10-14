# nerd/pipeline/tasks/mut_count.py
"""
Task for mutation counting using pluggable tools (e.g., shapemapper).
Parent samples only (no derived materialization yet).
"""

from pathlib import Path
from datetime import datetime
import math
import re
import shutil
from typing import Any, Dict, Tuple, Optional, List, Set

from .base import Task, TaskContext, TaskScope, TaskScopeMember
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

    def resolve_scope(self, ctx: Optional[TaskContext], inputs: Any) -> TaskScope:
        if ctx is None or inputs is None:
            return super().resolve_scope(ctx, inputs)

        try:
            resolved_samples, rg_map = self._resolve_sample_names(ctx, inputs)
        except Exception:
            return super().resolve_scope(ctx, inputs)

        members: List[TaskScopeMember] = []
        sample_ids: List[int] = []
        member_key_seen: Set[tuple[Any, Any, Any]] = set()

        def _add_member(member: TaskScopeMember) -> None:
            key = (member.kind, member.ref_id, member.label)
            if key in member_key_seen:
                return
            member_key_seen.add(key)
            members.append(member)

        # Attach reaction-group members (if any were resolved)
        for rg_id, rg_label in rg_map.items():
            _add_member(TaskScopeMember(kind="rg", ref_id=rg_id, label=rg_label))

        # Attach sample/derived members
        for name in resolved_samples:
            if not name:
                continue
            row = ctx.db.execute(
                """
                SELECT ss.id AS sample_id,
                       ss.sample_name AS sample_name,
                       pr.rg_id AS rg_id,
                       rg.rg_label AS rg_label
                FROM sequencing_samples ss
                LEFT JOIN probe_reactions pr ON pr.s_id = ss.id
                LEFT JOIN probe_reaction_groups rg ON rg.rg_id = pr.rg_id
                WHERE ss.sample_name = ?
                LIMIT 1
                """,
                (name,),
            ).fetchone()
            if row:
                sample_id = int(row["sample_id"])
                sample_ids.append(sample_id)
                _add_member(TaskScopeMember(kind="sample", ref_id=sample_id, label=row["sample_name"]))
                if row["rg_id"] is not None and row["rg_id"] not in rg_map:
                    _add_member(TaskScopeMember(kind="rg", ref_id=int(row["rg_id"]), label=row["rg_label"]))
                continue

            drow = ctx.db.execute(
                """
                SELECT id, parent_sample_id, child_name
                FROM sequencing_derived_samples
                WHERE child_name = ?
                """,
                (name,),
            ).fetchone()
            if drow:
                parent_id = drow["parent_sample_id"]
                extra: Dict[str, Any] = {}
                if parent_id is not None:
                    extra["parent_sample_id"] = int(parent_id)
                _add_member(
                    TaskScopeMember(
                        kind="derived_sample",
                        ref_id=int(drow["id"]),
                        label=drow["child_name"],
                        extra=extra or None,
                    )
                )
                if parent_id is not None and parent_id not in sample_ids:
                    prow = ctx.db.execute(
                        "SELECT id, sample_name FROM sequencing_samples WHERE id = ?",
                        (parent_id,),
                    ).fetchone()
                    if prow:
                        pid = int(prow["id"])
                        sample_ids.append(pid)
                        _add_member(TaskScopeMember(kind="sample", ref_id=pid, label=prow["sample_name"]))
                continue

            _add_member(TaskScopeMember(kind="unknown_sample", label=name))

        # Deduplicate sample_ids preserving order
        unique_sample_ids: List[int] = []
        seen_ids: Set[int] = set()
        for sid in sample_ids:
            if sid not in seen_ids:
                unique_sample_ids.append(sid)
                seen_ids.add(sid)

        combined_rg_map = dict(rg_map)
        if not combined_rg_map:
            # Harvest rg information gleaned from samples (if any)
            for member in members:
                if member.kind == "rg" and member.ref_id is not None:
                    combined_rg_map.setdefault(int(member.ref_id), member.label or None)

        if len(combined_rg_map) == 1:
            rg_id, rg_label = next(iter(combined_rg_map.items()))
            return TaskScope(
                kind="rg",
                scope_id=int(rg_id),
                label=rg_label,
                members=members,
            )

        if len(unique_sample_ids) == 1:
            sid = unique_sample_ids[0]
            sample_label = next(
                (m.label for m in members if m.kind == "sample" and m.ref_id == sid),
                None,
            )
            return TaskScope(
                kind="sample",
                scope_id=sid,
                label=sample_label,
                members=members,
            )

        scope_kind = "sample_batch" if unique_sample_ids else self.scope_kind or "global"
        scope = TaskScope(
            kind=scope_kind,
            scope_id=None,
            members=members,
        )
        if scope_kind == "sample_batch":
            # Provide a display label mentioning the first few samples
            if members:
                names = [
                    m.label for m in members
                    if m.kind in {"sample", "derived_sample"} and m.label
                ]
                if names:
                    scope.label = ", ".join(names[:3]) + ("…" if len(names) > 3 else "")
        return scope

    def _resolve_sample_names(
        self,
        ctx: TaskContext,
        inputs: Dict[str, Any],
    ) -> Tuple[List[str], Dict[int, Optional[str]]]:
        cache = inputs.get("_scope_resolution_cache")
        if isinstance(cache, dict) and "samples" in cache:
            samples = list(cache.get("samples") or [])
            rg_map = dict(cache.get("reaction_groups") or {})
            inputs["samples"] = samples
            return samples, rg_map

        configured_samples: List[str] = []
        for raw_sample in inputs.get("samples") or []:
            if raw_sample is None:
                continue
            text_val = str(raw_sample).strip()
            if not text_val:
                continue
            configured_samples.append(text_val)

        reaction_samples: List[str] = []
        rg_map: Dict[int, Optional[str]] = {}
        reaction_groups_cfg = inputs.get("reaction_group")
        if reaction_groups_cfg:
            if isinstance(reaction_groups_cfg, (list, tuple, set)):
                rg_values = list(reaction_groups_cfg)
            else:
                rg_values = [reaction_groups_cfg]
            for rg_value in rg_values:
                info = self._fetch_reaction_group_info(ctx, rg_value)
                if info is None:
                    continue
                rg_id, rg_label, samples = info
                if rg_id is not None:
                    rg_map.setdefault(rg_id, rg_label)
                reaction_samples.extend(samples)
                log.info("Resolved reaction_group %s to %d sample(s).", rg_value, len(samples))

        combined = configured_samples + reaction_samples
        seen: Set[str] = set()
        sample_names: List[str] = []
        for name in combined:
            if not name or name in seen:
                continue
            seen.add(name)
            sample_names.append(name)

        inputs["_scope_resolution_cache"] = {
            "samples": list(sample_names),
            "reaction_groups": dict(rg_map),
        }
        inputs["samples"] = sample_names
        inputs["_resolved_reaction_groups"] = dict(rg_map)
        return sample_names, rg_map

    def _fetch_reaction_group_info(
        self,
        ctx: TaskContext,
        rg_value: Any,
    ) -> Optional[Tuple[int, Optional[str], List[str]]]:
        if rg_value in (None, ""):
            return None
        rg_id: Optional[int] = None
        rg_label: Optional[str] = None
        if isinstance(rg_value, int):
            row = ctx.db.execute(
                "SELECT rg_id, rg_label FROM probe_reaction_groups WHERE rg_id = ?",
                (rg_value,),
            ).fetchone()
            if not row:
                raise ValueError(f"Reaction group id '{rg_value}' not found in database.")
            rg_id = int(row["rg_id"])
            rg_label = row["rg_label"]
        else:
            text = str(rg_value).strip()
            if not text:
                return None
            row = ctx.db.execute(
                "SELECT rg_id, rg_label FROM probe_reaction_groups WHERE rg_label = ?",
                (text,),
            ).fetchone()
            if not row and text.isdigit():
                row = ctx.db.execute(
                    "SELECT rg_id, rg_label FROM probe_reaction_groups WHERE rg_id = ?",
                    (int(text),),
                ).fetchone()
            if not row:
                raise ValueError(f"Reaction group '{rg_value}' not found in database.")
            rg_id = int(row["rg_id"])
            rg_label = row["rg_label"]

        rows = ctx.db.execute(
            """
            SELECT ss.sample_name
            FROM probe_reactions pr
            JOIN sequencing_samples ss ON ss.id = pr.s_id
            WHERE pr.rg_id = ?
            ORDER BY pr.reaction_time, pr.replicate, ss.sample_name
            """,
            (rg_id,),
        ).fetchall()
        samples = [
            row["sample_name"] if hasattr(row, "keys") else row[0]
            for row in rows
        ]
        if not samples:
            raise ValueError(f"Reaction group '{rg_value}' has no associated samples.")
        return rg_id, rg_label, samples

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
        rg_cfg = mc.get("reaction_group")
        if isinstance(rg_cfg, (list, tuple, set)):
            reaction_groups = [str(item).strip() for item in rg_cfg if str(item).strip()]
        elif rg_cfg not in (None, ""):
            reaction_groups = [str(rg_cfg).strip()]
        else:
            reaction_groups = []
        if not samples and not derived and not reaction_groups:
            raise ValueError("mut_count requires 'samples', 'derived_samples', and/or 'reaction_group' to list inputs.")
        merged = samples + derived
        mc = {**mc, "samples": merged}
        if reaction_groups:
            mc["reaction_group"] = reaction_groups if len(reaction_groups) > 1 else reaction_groups[0]

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
        param_opts = inputs.get("params") or {}
        if not isinstance(param_opts, dict):
            param_opts = {}

        def _opt_bool(key: str, default: bool) -> bool:
            if key in param_opts:
                return bool(param_opts.get(key))
            return bool(inputs.get(key, default))

        options = {
            "amplicon": _opt_bool("amplicon", True),
            "dms_mode": _opt_bool("dms_mode", False),
            "output_N7": _opt_bool("output_N7", False),
            "output_parsed_mutations": _opt_bool("output_parsed_mutations", False),
            "per_read_histograms": _opt_bool("per_read_histograms", False),
        }
        dry_run = bool(param_opts.get("dry_run", inputs.get("dry_run", False)))

        sample_names, _ = self._resolve_sample_names(ctx, inputs)
        if not sample_names:
            return None

        label_dir = Path(ctx.output_dir) / ctx.label

        def _nt_rows_for_construct(cid: int) -> List[Dict[str, Any]]:
            rows = ctx.db.execute(
                "SELECT site, base, base_region FROM meta_nucleotides WHERE construct_id = ? ORDER BY site",
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
                "SELECT * FROM sequencing_derived_samples WHERE child_name = ?",
                (child_name,),
            ).fetchone()
            return row

        def _construct_id_for_sample_id(sid: int) -> Optional[int]:
            row = ctx.db.execute(
                "SELECT construct_id FROM probe_reactions WHERE s_id = ? ORDER BY rowid DESC LIMIT 1",
                (sid,),
            ).fetchone()
            return int(row[0]) if row and row[0] is not None else None

        cmds: List[str] = []
        self._stage_in = []
        self._stage_out_extra = [
            "artifacts/*/*_profile.txt",
            "artifacts/*/*_profile.txtga",
            "artifacts/*/per_read_histogram.txt",
            "artifacts/*/per_read_histogram.txtga",
        ]
        successes = 0
        for name in sample_names:
            start_cmd_idx = len(cmds)
            start_stage_in_idx = len(self._stage_in)
            start_stage_out_idx = len(self._stage_out_extra)
            try:
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
                backend = str(ctx.backend or "").lower()
                needs_stage = backend not in {"local"}
                if needs_stage:
                    self._stage_in.append({"src": str(r1), "dst": str(remote_r1)})
                    self._stage_in.append({"src": str(r2), "dst": str(remote_r2)})

                # If derived, materialize child FASTQs on remote first
                parent_r1_for_use = remote_r1 if needs_stage else r1
                parent_r2_for_use = remote_r2 if needs_stage else r2
                use_r1 = parent_r1_for_use
                use_r2 = parent_r2_for_use
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
                        parent_r1_remote=parent_r1_for_use,
                        parent_r2_remote=parent_r2_for_use,
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
                wrapped_shapecmd = f"{shapecmd} || echo 'shapemapper failed for {name}'"
                sep = "################################################################################"
                if not is_derived:
                    # Add verification steps for non-derived cases (derived already logs these)
                    verify_label = "Verify staged FASTQ" if needs_stage else "Verify FASTQ inputs"
                    cmds.append(f"echo '{sep}'")
                    cmds.append(f"echo '# 3 - {verify_label}'")
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
                    dry_label = "staged FASTQs" if needs_stage else "FASTQ inputs"
                    cmds.append(f"echo '[Step 1] Verifying {dry_label} for {name}'")
                    cmds.append(f"ls -lh {use_r1} {use_r2} || true")
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
                    cmds.append(wrapped_shapecmd)
            except Exception:
                cmds[:] = cmds[:start_cmd_idx]
                self._stage_in[:] = self._stage_in[:start_stage_in_idx]
                self._stage_out_extra[:] = self._stage_out_extra[:start_stage_out_idx]
                log.exception("Failed to prepare mut_count inputs for sample %s; skipping.", name)
                continue

            successes += 1

        if successes == 0:
            raise RuntimeError("No samples qualified for mut_count task.")

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
            if isinstance(inputs, dict):
                sample_names, _ = self._resolve_sample_names(ctx, inputs)
            else:
                sample_names = []
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
                "SELECT parent_sample_id FROM sequencing_derived_samples WHERE child_name = ?",
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

    def consume_outputs(self, ctx: TaskContext, inputs: Dict[str, Any], params: Dict[str, Any], run_dir: Path, task_id: Optional[int] = None):
        """
        Execute mutation counting for the selected parent samples and ingest the results.
        """
        from nerd.pipeline.plugins.mutcount import load_mutcount_plugin
        from nerd.db import api as db_api

        plugin_name = inputs.get("plugin")
        tool_cfg = inputs.get("tool", {}) or {}
        plugin = load_mutcount_plugin(plugin_name, bin_path=tool_cfg.get("bin"), version=tool_cfg.get("version"))

        param_opts = inputs.get("params") or {}
        if not isinstance(param_opts, dict):
            param_opts = {}
        per_read_hist_enabled = bool(param_opts.get("per_read_histograms", inputs.get("per_read_histograms", False)))

        sample_names: List[str] = inputs.get("samples", [])
        if not sample_names:
            log.warning("mut_count.consume_outputs called with no samples.")
            return

        artifacts_dir = run_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        def _resolve_profile_path(sample_dir: Path, sample_name: str, ga: bool = False) -> Optional[Path]:
            suffix = "_profile.txtga" if ga else "_profile.txt"
            for root in (sample_dir, run_dir):
                if not root.exists():
                    continue
                try:
                    candidates = sorted(root.rglob(f"*{suffix}"))
                except Exception:
                    candidates = []
                for candidate in candidates:
                    posix = candidate.as_posix()
                    if "shapemapper_temp" in posix:
                        continue
                    if ga:
                        if not candidate.name.endswith("_profile.txtga"):
                            continue
                    else:
                        if candidate.name.endswith(".txtga"):
                            continue
                    if sample_name not in candidate.name and root is run_dir:
                        continue
                    return candidate
            return None

        found_profiles = 0
        ingested_runs = 0

        for name in sample_names:
            sample_dir = artifacts_dir / name
            profile_path = _resolve_profile_path(sample_dir, name, ga=False)
            if profile_path:
                found_profiles += 1
                log.info("Found ShapeMapper profile for %s at %s", name, profile_path)
            else:
                log.warning("No ShapeMapper profile found for %s under %s", name, sample_dir)
                try:
                    txts = sorted([str(p.relative_to(run_dir)) for p in sample_dir.rglob("*.txt")])
                    if txts:
                        log.info(
                            "Found txt files under %s: %s",
                            sample_dir,
                            ", ".join(txts[:10]) + (" ..." if len(txts) > 10 else ""),
                        )
                    else:
                        log.info("No txt files present under %s", sample_dir)
                except Exception:
                    pass
                continue

            ga_profile_path = _resolve_profile_path(sample_dir, name, ga=True)

            allowed_paths: Set[Path] = set()
            try:
                resolved_profile = profile_path.resolve()
                allowed_paths.add(resolved_profile)
            except Exception:
                pass
            if ga_profile_path:
                try:
                    allowed_paths.add(ga_profile_path.resolve())
                except Exception:
                    pass

            sample_row = ctx.db.execute(
                "SELECT id FROM sequencing_samples WHERE sample_name = ?",
                (name,),
            ).fetchone()
            if sample_row is None:
                log.error("Sample %s not found in sequencing_samples; skipping ingestion.", name)
                continue
            s_id = int(sample_row["id"] if hasattr(sample_row, "keys") else sample_row[0])

            probing_row = ctx.db.execute(
                "SELECT id, construct_id, treated FROM probe_reactions WHERE s_id = ?",
                (s_id,),
            ).fetchone()
            if probing_row is None:
                log.error("No probe_reactions entry linked to sample %s (id=%s); skipping.", name, s_id)
                continue
            if hasattr(probing_row, "keys"):
                rxn_id = int(probing_row["id"])
                construct_id = int(probing_row["construct_id"])
                treated_flag = int(probing_row["treated"])
            else:
                rxn_id = int(probing_row[0])
                construct_id = int(probing_row[1])
                treated_flag = int(probing_row[2])

            nt_rows = ctx.db.execute(
                "SELECT id, site, base FROM meta_nucleotides WHERE construct_id = ? ORDER BY site",
                (construct_id,),
            ).fetchall()
            if not nt_rows:
                log.error("No meta_nucleotides found for construct %s (sample %s); skipping.", construct_id, name)
                continue

            nt_map: Dict[int, int] = {}
            db_sequence_parts: List[str] = []
            for row in nt_rows:
                if hasattr(row, "keys"):
                    nt_id = int(row["id"])
                    site = int(row["site"])
                    base = str(row["base"])
                else:
                    nt_id = int(row[0])
                    site = int(row[1])
                    base = str(row[2])
                nt_map[site] = nt_id
                db_sequence_parts.append(base)

            log_path = self._find_shapemapper_log(run_dir, sample_dir, name)
            meta = self._parse_shapemapper_log(log_path) if log_path else {}
            histograms = meta.get("histograms") or {}
            run_datetime = meta.get("run_datetime") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            software_version = meta.get("software_version") or tool_cfg.get("version") or "unknown"
            run_args = meta.get("run_args") or ""
            untreated_arg = meta.get("untreated")
            use_untreated_calc = bool(untreated_arg) and treated_flag == 0

            value_column = "Untreated_rate" if use_untreated_calc else "Modified_rate"
            depth_column = "Untreated_read_depth" if use_untreated_calc else "Modified_read_depth"

            run_data = {
                "software_name": str(plugin_name or "unknown"),
                "software_version": str(software_version),
                "run_args": run_args,
                "run_datetime": run_datetime,
                "output_dir": str(sample_dir.resolve()),
                "s_id": s_id,
            }

            run_id = db_api.insert_probe_fmod_run(ctx.db, run_data)
            if run_id is None:
                existing = ctx.db.execute(
                    "SELECT id FROM probe_fmod_runs WHERE software_name = ? AND software_version = ? AND run_args = ? AND run_datetime = ? AND output_dir = ? AND s_id = ?",
                    (
                        run_data["software_name"],
                        run_data["software_version"],
                        run_data["run_args"],
                        run_data["run_datetime"],
                        run_data["output_dir"],
                        run_data["s_id"],
                    ),
                ).fetchone()
                if existing is None:
                    log.error("Unable to insert or locate probe_fmod_run for %s; skipping fmod ingestion.", name)
                    continue
                run_id = int(existing["id"] if hasattr(existing, "keys") else existing[0])
            else:
                log.info("Inserted probe_fmod_run %s for sample %s", run_id, name)

            existing_valtypes_rows = ctx.db.execute(
                "SELECT DISTINCT valtype FROM probe_fmod_values WHERE fmod_run_id = ?",
                (run_id,),
            ).fetchall()
            existing_valtypes = {
                (row["valtype"] if hasattr(row, "keys") else row[0]) for row in existing_valtypes_rows
            }

            profiles_to_ingest = [(profile_path, "modrate")]
            if ga_profile_path:
                profiles_to_ingest.append((ga_profile_path, "GAmodrate"))

            inserted_for_sample = False
            db_sequence = "".join(db_sequence_parts).upper().replace("T", "U")

            for path, valtype in profiles_to_ingest:
                if valtype in existing_valtypes:
                    log.info("probe_fmod_values for %s (%s) already present; skipping.", name, valtype)
                    continue

                parsed_rows = plugin.parse_profile(path)
                if not parsed_rows:
                    log.warning("Profile %s for %s is empty; skipping %s ingestion.", path, name, valtype)
                    continue

                profile_sequence = "".join((row.get("Sequence") or "") for row in parsed_rows).upper().replace("T", "U")
                if profile_sequence != db_sequence:
                    log.warning(
                        "Sequence mismatch for %s when ingesting %s (profile vs construct). Continuing with available nucleotides.",
                        name,
                        valtype,
                    )

                records: List[Dict[str, Any]] = []
                for row in parsed_rows:
                    try:
                        site = int(float(row.get("Nucleotide", 0)))
                    except (TypeError, ValueError):
                        continue
                    nt_id = nt_map.get(site)
                    if nt_id is None:
                        continue
                    fmod_value = self._safe_float(row.get(value_column))
                    depth_value = self._safe_float(row.get(depth_column))
                    read_depth = int(round(depth_value)) if depth_value is not None else 0
                    records.append(
                        {
                            "nt_id": nt_id,
                            "fmod_run_id": run_id,
                            "fmod_val": fmod_value,
                            "valtype": valtype,
                            "read_depth": read_depth,
                            "rxn_id": rxn_id,
                        }
                    )

                if not records:
                    log.warning("No probe_fmod_values generated for %s (%s); skipping.", name, valtype)
                    continue

                db_api.bulk_insert_fmod_vals(ctx.db, records)
                log.info("Inserted %d probe_fmod_values records for %s (%s).", len(records), name, valtype)
                existing_valtypes.add(valtype)
                inserted_for_sample = True

            if per_read_hist_enabled and histograms:
                mod_hist = histograms.get("modrate")
                if mod_hist:
                    hist_path = sample_dir / "per_read_histogram.txt"
                    self._write_histogram_file(hist_path, mod_hist)
                    try:
                        allowed_paths.add(hist_path.resolve())
                    except Exception:
                        pass
                    log.info("Wrote per-read histogram for %s (%s) to %s", name, "Modified", hist_path)
                ga_hist = histograms.get("GAmodrate")
                if ga_hist:
                    hist_ga_path = sample_dir / "per_read_histogram.txtga"
                    self._write_histogram_file(hist_ga_path, ga_hist)
                    try:
                        allowed_paths.add(hist_ga_path.resolve())
                    except Exception:
                        pass
                    log.info("Wrote per-read histogram for %s (%s) to %s", name, "GA", hist_ga_path)

            if log_path and log_path.exists():
                try:
                    log_path.unlink()
                    log.debug("Removed shapemapper log %s", log_path)
                except Exception as exc:
                    log.debug("Failed to remove shapemapper log %s: %s", log_path, exc)

            if sample_dir.exists():
                for item in list(sample_dir.iterdir()):
                    try:
                        resolved_item = item.resolve()
                    except FileNotFoundError:
                        continue
                    if resolved_item in allowed_paths:
                        continue
                    if item.is_dir():
                        shutil.rmtree(item, ignore_errors=True)
                    else:
                        try:
                            item.unlink()
                        except FileNotFoundError:
                            pass

            if inserted_for_sample:
                ingested_runs += 1

        log.info(
            "mut_count completed. Profiles found for %d/%d samples; ingested %d run(s).",
            found_profiles,
            len(sample_names),
            ingested_runs,
        )

    def _find_shapemapper_log(self, run_dir: Path, sample_dir: Path, sample_name: str) -> Optional[Path]:
        candidates = [
            run_dir / f"{sample_name}_shapemapper_log.txt",
            sample_dir / f"{sample_name}_shapemapper_log.txt",
        ]
        for cand in candidates:
            if cand.exists():
                return cand
        matches = sorted(run_dir.glob(f"*{sample_name}*shapemapper_log*.txt"))
        if matches:
            return matches[0]
        matches = sorted(sample_dir.glob("*shapemapper_log*.txt"))
        if matches:
            return matches[0]
        return None

    def _parse_shapemapper_log(self, log_path: Path) -> Dict[str, Any]:
        if not log_path or not log_path.exists():
            return {}
        try:
            lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            return {}

        indices = [i for i, line in enumerate(lines) if line.startswith("Started ShapeMapper")]
        if not indices:
            return {}
        block = lines[indices[-1]:]
        header = block[0].strip()
        run_datetime = ""
        software_version = ""
        if " at " in header:
            left, right = header.split(" at ", 1)
            run_datetime = right.strip()
            parts = left.split()
            if len(parts) >= 3:
                software_version = parts[2]
        args_line = next((line for line in block if line.strip().startswith("args:")), "")
        args_text = ""
        if args_line:
            args_text = args_line.split("args:", 1)[1].strip()
        arg_map: Dict[str, Optional[str]] = {}
        if args_text:
            for chunk in args_text.split(" --"):
                chunk = chunk.strip()
                if not chunk:
                    continue
                if chunk.startswith("--"):
                    chunk = chunk[2:]
                if " " in chunk:
                    flag, value = chunk.split(" ", 1)
                    arg_map[flag] = value.strip()
                else:
                    arg_map[chunk] = None
        histograms = self._extract_mutation_histograms(lines)
        return {
            "run_datetime": run_datetime,
            "software_version": software_version,
            "run_args": args_text,
            "untreated": arg_map.get("untreated"),
            "denatured": arg_map.get("denatured"),
            "r1": arg_map.get("R1"),
            "histograms": histograms,
        }

    def _extract_mutation_histograms(self, lines: List[str]) -> Dict[str, List[Tuple[int, float]]]:
        hist_data: Dict[str, List[Tuple[int, float]]] = {}
        current: Optional[str] = None
        collecting = False
        header_skip = 0
        buffer: List[str] = []

        def _finalize() -> None:
            nonlocal buffer, current
            if current and buffer:
                parsed = self._parse_histogram_records(buffer)
                if parsed:
                    hist_data[current] = parsed
            buffer = []

        for raw_line in lines:
            line = raw_line.rstrip("\n")
            if "MutationCounterGA_Modified" in line:
                if collecting:
                    _finalize()
                current = "GAmodrate"
                collecting = False
                header_skip = 0
                continue
            if "MutationCounter_Modified" in line and "MutationCounterGA_Modified" not in line:
                if collecting:
                    _finalize()
                current = "modrate"
                collecting = False
                header_skip = 0
                continue
            if current is None:
                continue
            stripped = line.strip()
            if stripped.startswith("| Mutations per read"):
                if collecting:
                    _finalize()
                collecting = True
                header_skip = 0
                buffer = []
                continue
            if not collecting:
                continue
            if stripped.startswith("| --------------------"):
                if header_skip < 2:
                    header_skip += 1
                    continue
                _finalize()
                collecting = False
                header_skip = 0
                continue
            if header_skip < 2:
                header_skip += 1
                continue
            buffer.append(line)

        if collecting and buffer:
            _finalize()

        return hist_data

    @staticmethod
    def _parse_histogram_records(lines: List[str]) -> List[Tuple[int, float]]:
        data: List[Tuple[int, float]] = []
        for line in lines:
            if "|" not in line:
                continue
            payload = line.split("|", 1)[1].strip()
            if not payload:
                continue
            parts = re.split(r"\s+", payload)
            if len(parts) < 2:
                continue
            try:
                bin_left = int(parts[0])
                frequency = float(parts[1])
            except ValueError:
                continue
            data.append((bin_left, frequency))
        return data

    @staticmethod
    def _write_histogram_file(path: Path, data: List[Tuple[int, float]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            handle.write("bin_left\tfrequency\n")
            for bin_left, frequency in data:
                handle.write(f"{bin_left}\t{frequency:.6f}\n")

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, str):
            val = value.strip()
            if not val or val.lower() == "nan":
                return None
        else:
            val = value
        try:
            num = float(val)
        except (TypeError, ValueError):
            return None
        if math.isnan(num):
            return None
        return num
