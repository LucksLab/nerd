# nerd/pipeline/tasks/create.py
"""
Task for creating and ingesting initial data from a YAML configuration file.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .base import Task, TaskContext
from nerd.utils.logging import get_logger
from nerd.db import api as db_api

log = get_logger(__name__)

class CreateTask(Task):
    """
    A task for initial data ingestion from a structured YAML file.
    This task reads construct, buffer, sequencing run, and sample information
    and populates the corresponding tables in the database.
    """
    name = "create"
    scope_kind = "global"  # This task operates at a global level, not on a specific sample/rg

    _HEADER_MAPS: Dict[str, Dict[str, str]] = {
        "construct": {
            "family": "family",
            "name": "name",
            "version": "version",
            "sequence": "sequence",
            "disp_name": "disp_name",
            "display_name": "disp_name",
            "nt_info": "nt_info",
        },
        "buffer": {
            "name": "name",
            "buffer_name": "name",
            "ph": "pH",
            "p_h": "pH",
            "pH": "pH",
            "composition": "composition",
            "disp_name": "disp_name",
            "display_name": "disp_name",
        },
        "sequencing_run": {
            "run_name": "run_name",
            "name": "run_name",
            "date": "date",
            "sequencer": "sequencer",
            "run_manager": "run_manager",
            "operator": "run_manager",
        },
        "samples": {
            "sample_name": "sample_name",
            "fq_dir": "fq_dir",
            "fastq_dir": "fq_dir",
            "r1": "r1_file",
            "r1_file": "r1_file",
            "r2": "r2_file",
            "r2_file": "r2_file",
            "reaction_group": "reaction_group",
            "rxn_group": "reaction_group",
            "temperature": "temperature",
            "replicate": "replicate",
            "reaction_time": "reaction_time",
            "probe": "probe",
            "probe_concentration": "probe_concentration",
            "probe_conc": "probe_concentration",
            "rt": "RT",
            "done_by": "done_by",
            "treated": "treated",
            "buffer": "buffer",
            "buffer_disp": "buffer",
            "construct": "construct",
            "construct_disp": "construct",
            "sequencing_run_name": "sequencing_run_name",
            "sequencing_run": "sequencing_run_name",
        },
        "derived_samples": {
            "child_name": "child_name",
            "parent_sample": "parent_sample",
            "kind": "kind",
            "tool": "tool",
            "cmd_template": "cmd_template",
            "params": "params",
        },
    }

    _REQUIRED_COLUMNS: Dict[str, List[str]] = {
        "construct": ["family", "name", "version", "sequence", "disp_name"],
        "buffer": ["name", "pH", "composition", "disp_name"],
        "sequencing_run": ["run_name", "date", "sequencer", "run_manager"],
        "samples": [
            "sample_name",
            "fq_dir",
            "r1_file",
            "r2_file",
            "reaction_group",
            "temperature",
            "replicate",
            "reaction_time",
            "probe",
            "probe_concentration",
            "RT",
            "treated",
            "buffer",
            "construct",
            "done_by",
        ],
    }

    def prepare(self, cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Validates the presence of the 'create' section in the config
        and checks for the existence of specified file paths.
        """
        if self.name not in cfg:
            raise ValueError(f"Configuration must contain a '{self.name}' section.")
        
        create_cfg = dict(cfg[self.name])
        run_cfg = cfg.get("run", {})

        # --- Resolve sheet inputs if CSV paths were provided ---
        output_dir = Path(run_cfg.get("output_dir", ".")).resolve()
        label = run_cfg.get("label")
        if not label:
            raise ValueError("Configuration must contain a 'run.label'.")
        label_dir = output_dir / label
        search_roots = self._sheet_search_roots(label_dir, output_dir)

        create_cfg = self._inflate_sheet_inputs(create_cfg, search_roots)

        # --- Validate paths ---
        # 1. Check fq_dir for each sample
        for sample in create_cfg.get("samples", []):
            fq_dir_str = sample.get("fq_dir")
            if fq_dir_str:
                fq_dir = Path(fq_dir_str)
                if not fq_dir.is_absolute():
                    # Default root is the label directory
                    fq_dir = label_dir / fq_dir

                if not fq_dir.is_dir():
                    raise FileNotFoundError(f"FastQ directory not found: {fq_dir}")

                # 2. Check R1 and R2 files within the fq_dir
                r1_file = fq_dir / sample.get("r1_file", "")
                r2_file = fq_dir / sample.get("r2_file", "")
                if not r1_file.is_file():
                    raise FileNotFoundError(f"R1 file not found: {r1_file}")
                if not r2_file.is_file():
                    raise FileNotFoundError(f"R2 file not found: {r2_file}")

        # 3. Check nt_info csv if provided
        construct_cfg = create_cfg.get("construct") or {}
        if isinstance(construct_cfg, list):
            construct_iter: Iterable[Dict[str, Any]] = construct_cfg
        else:
            construct_iter = [construct_cfg]
        for item in construct_iter:
            if not item:
                continue
            nt_info_path_str = item.get("nt_info")
            if not nt_info_path_str:
                continue
            # Assumes nt_info is always within a 'configs' directory 
            # that is a sibling of the label directory.
            configs_dir = label_dir / "configs"
            nt_info_path = configs_dir / nt_info_path_str
            if not nt_info_path.is_file():
                raise FileNotFoundError(f"nt_info file not found at assumed path: {nt_info_path}")

        # 4. Ensure sequencing_run_name populated on samples if sequencing_run provided
        seqrun_section = create_cfg.get("sequencing_run")
        run_name = None
        if isinstance(seqrun_section, list):
            if seqrun_section:
                run_name = seqrun_section[0].get("run_name")
        elif isinstance(seqrun_section, dict):
            run_name = seqrun_section.get("run_name")
        if run_name:
            samples_block = create_cfg.get("samples") or []
            if isinstance(samples_block, dict):
                samples_block = [samples_block]
                create_cfg["samples"] = samples_block
            for sample in samples_block:
                if isinstance(sample, dict):
                    sample.setdefault("sequencing_run_name", run_name)

        log.debug("'create' task prepared with inputs from config.")
        return create_cfg, {}  # Return the 'create' block as inputs

    def _sheet_search_roots(self, label_dir: Path, output_dir: Path) -> List[Path]:
        roots: List[Path] = []
        for candidate in (
            label_dir / "configs",
            label_dir,
            output_dir,
            Path.cwd(),
        ):
            try:
                resolved = candidate.resolve()
            except Exception:
                continue
            if resolved not in roots:
                roots.append(resolved)
        return roots

    def _inflate_sheet_inputs(self, cfg_block: Dict[str, Any], search_roots: List[Path]) -> Dict[str, Any]:
        data = dict(cfg_block or {})

        data["construct"] = self._maybe_load_sheet(
            section="construct",
            value=data.get("construct"),
            search_roots=search_roots,
            allow_multiple=False,
        )
        data["buffer"] = self._maybe_load_sheet(
            section="buffer",
            value=data.get("buffer"),
            search_roots=search_roots,
            allow_multiple=False,
        )
        data["sequencing_run"] = self._maybe_load_sheet(
            section="sequencing_run",
            value=data.get("sequencing_run"),
            search_roots=search_roots,
            allow_multiple=False,
        )
        data["samples"] = self._maybe_load_sheet(
            section="samples",
            value=data.get("samples"),
            search_roots=search_roots,
            allow_multiple=True,
        ) or []
        data["derived_samples"] = self._maybe_load_sheet(
            section="derived_samples",
            value=data.get("derived_samples"),
            search_roots=search_roots,
            allow_multiple=True,
        ) or []

        return data

    def _maybe_load_sheet(
        self,
        *,
        section: str,
        value: Any,
        search_roots: List[Path],
        allow_multiple: bool,
    ) -> Any:
        if isinstance(value, str):
            rows = self._load_sheet_rows(section, value, search_roots)
            if not rows:
                raise ValueError(f"Sheet '{value}' for '{section}' is empty.")
            if not allow_multiple and len(rows) > 1:
                log.warning(
                    "Sheet '%s' for '%s' contains %d rows; only the first will be used.",
                    value,
                    section,
                    len(rows),
                )
                return rows[0]
            return rows
        return value

    def _load_sheet_rows(self, section: str, path_str: str, search_roots: List[Path]) -> List[Dict[str, Any]]:
        path = Path(path_str)
        candidates: List[Path] = []
        if path.is_absolute():
            candidates.append(path)
        else:
            for root in search_roots:
                candidates.append((root / path).resolve())
            candidates.append((Path.cwd() / path).resolve())
        resolved: Optional[Path] = None
        for candidate in candidates:
            if candidate.is_file():
                resolved = candidate
                break
        if resolved is None:
            raise FileNotFoundError(f"Sheet file for '{section}' not found: {path_str}")

        rows: List[Dict[str, Any]] = []
        with resolved.open("r", newline="", encoding="utf-8-sig") as fh:
            sample = fh.read(2048)
            fh.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample)
            except csv.Error:
                dialect = csv.excel_tab if "\t" in sample and "," not in sample else csv.excel
            reader = csv.DictReader(fh, dialect=dialect)
            for raw_row in reader:
                normalized = self._normalize_row(section, raw_row)
                if not any(val not in (None, "", "NaN") for val in normalized.values()):
                    continue
                rows.append(normalized)

        required = self._REQUIRED_COLUMNS.get(section)
        if required:
            missing = [
                req
                for req in required
                if any(req not in row or row[req] in (None, "") for row in rows)
            ]
            if missing:
                raise ValueError(
                    f"Sheet '{path_str}' for '{section}' is missing required columns or values: {missing}"
                )

        return rows

    def _normalize_row(self, section: str, row: Dict[str, Any]) -> Dict[str, Any]:
        header_map = self._HEADER_MAPS.get(section, {})
        normalized: Dict[str, Any] = {}
        for raw_key, raw_value in row.items():
            if raw_key is None:
                continue
            key_clean = raw_key.strip()
            if not key_clean:
                continue
            target_key = header_map.get(key_clean.lower(), header_map.get(key_clean, key_clean))
            value = raw_value
            if isinstance(value, str):
                value = value.strip()
                if value == "":
                    value = None
            normalized[target_key] = value
        return normalized

    def command(self, ctx, inputs: Dict[str, Any], params: Dict[str, Any]) -> Optional[str]:
        """
        This is a pure-Python task, so no shell command is executed.
        """
        return None

    def consume_outputs(self, ctx: TaskContext, inputs: Dict[str, Any], params: Dict[str, Any], run_dir: Path):
        """
        Parses the 'create' config and writes the data to the database.
        """
        log.info("Consuming 'create' inputs and writing to the database.")

        try:
            # The 'inputs' are the contents of the 'create:' block from the YAML
            construct_entries_raw = inputs.get("construct")
            buffer_entries_raw = inputs.get("buffer")
            sequencing_run_data = inputs.get("sequencing_run")
            samples_data = inputs.get("samples", []) or []
            derived_data = inputs.get("derived_samples", []) or []
            # Ensure seqrun_id is defined for type-checkers; may remain None
            seqrun_id: Optional[int] = None
            default_construct_id: Optional[int] = None
            default_buffer_id: Optional[int] = None
            construct_cache: Dict[str, Optional[int]] = {}
            buffer_cache: Dict[str, Optional[int]] = {}

            if isinstance(samples_data, dict):
                samples_data = [samples_data]
            if isinstance(derived_data, dict):
                derived_data = [derived_data]

            label_dir = Path(ctx.output_dir) / ctx.label

            with ctx.db: # Use the connection as a context manager for transactions
                construct_entries: List[Dict[str, Any]] = []
                if isinstance(construct_entries_raw, list):
                    construct_entries = [c for c in construct_entries_raw if c]
                elif isinstance(construct_entries_raw, dict) and construct_entries_raw:
                    construct_entries = [construct_entries_raw]

                if construct_entries:
                    log.info("Inserting %d construct definition(s).", len(construct_entries))
                    for entry in construct_entries:
                        payload = dict(entry)
                        nt_info_rel = payload.get("nt_info")
                        if nt_info_rel:
                            configs_dir = label_dir / "configs"
                            nt_info_path = Path(nt_info_rel)
                            if not nt_info_path.is_absolute():
                                nt_info_path = configs_dir / nt_info_path
                            nt_rows: List[Dict[str, Any]] = db_api.prep_nt_rows(nt_info_path)
                            payload = {**payload, "nt_rows": nt_rows}

                        cid = db_api.upsert_construct(ctx.db, payload)
                        log.debug("Construct upserted with id=%s", cid)
                        if cid is not None:
                            if default_construct_id is None:
                                default_construct_id = cid
                            disp = payload.get("disp_name")
                            if disp not in (None, ""):
                                construct_cache[str(disp).lower()] = cid

                buffer_entries: List[Dict[str, Any]] = []
                if isinstance(buffer_entries_raw, list):
                    buffer_entries = [b for b in buffer_entries_raw if b]
                elif isinstance(buffer_entries_raw, dict) and buffer_entries_raw:
                    buffer_entries = [buffer_entries_raw]

                if buffer_entries:
                    log.info("Inserting %d buffer definition(s).", len(buffer_entries))
                    for entry in buffer_entries:
                        payload = dict(entry)
                        bid = db_api.upsert_buffer(ctx.db, payload)
                        log.debug("Buffer upserted with id=%s", bid)
                        if bid is not None:
                            if default_buffer_id is None:
                                default_buffer_id = bid
                            buf_name = payload.get("name")
                            buf_disp = payload.get("disp_name")
                            if buf_name not in (None, ""):
                                buffer_cache[str(buf_name).lower()] = bid
                            if buf_disp not in (None, ""):
                                buffer_cache[str(buf_disp).lower()] = bid

                seqrun_payload: Optional[Dict[str, Any]] = None
                if isinstance(sequencing_run_data, list):
                    if sequencing_run_data:
                        if len(sequencing_run_data) > 1:
                            log.warning(
                                "Sequencing run sheet contained %d rows; only the first will be used.",
                                len(sequencing_run_data),
                            )
                        seqrun_payload = sequencing_run_data[0]
                elif isinstance(sequencing_run_data, dict):
                    seqrun_payload = sequencing_run_data

                if seqrun_payload:
                    log.info("Inserting sequencing run: %s", seqrun_payload.get('run_name'))
                    seqrun_id = db_api.upsert_sequencing_run(ctx.db, seqrun_payload)
                    log.debug("Sequencing run upserted with id=%s", seqrun_id)

                if samples_data:
                    log.info("Inserting %d samples.", len(samples_data))

                    run_name_for_samples = (seqrun_payload or {}).get("run_name")
                    if run_name_for_samples:
                        for sample in samples_data:
                            if isinstance(sample, dict):
                                sample.setdefault("sequencing_run_name", run_name_for_samples)

                    if seqrun_id is None:
                        # Try to resolve based on sequencing_run_name in samples
                        run_names = {
                            str(sample.get("sequencing_run_name")).strip()
                            for sample in samples_data
                            if isinstance(sample, dict) and sample.get("sequencing_run_name") not in (None, "")
                        }
                        run_names = {name for name in run_names if name}
                        if not run_names:
                            raise ValueError(
                                "No sequencing run provided and samples lack sequencing_run_name; cannot resolve sequencing run."
                            )
                        if len(run_names) > 1:
                            raise ValueError(
                                f"Multiple sequencing_run_name values found with no sequencing_run block: {sorted(run_names)}"
                            )
                        run_name_only = next(iter(run_names))
                        seqrun_id = db_api.find_sequencing_run_id_by_name(ctx.db, run_name_only)
                        if seqrun_id is None:
                            raise ValueError(
                                f"Sequencing run '{run_name_only}' not found. Provide a sequencing_run section or create it first."
                            )
                        log.debug("Resolved existing sequencing run '%s' to id=%s", run_name_only, seqrun_id)

                    if seqrun_id is None:
                        raise ValueError("'samples' provided but sequencing_run could not be resolved.")

                    if 'seqrun_id' in samples_data[0]:
                        # Support pre-populated seqrun_id, but prefer the resolved one above
                        pass

                    inserted = db_api.bulk_upsert_samples(ctx.db, seqrun_id, samples_data)
                    log.debug("Upserted %d samples for seqrun_id=%s", inserted, seqrun_id)

                # --- Optional: upsert derived_samples metadata (no FASTQs stored) ---
                if derived_data:
                    log.info("Inserting %d derived sample definitions.", len(derived_data))
                    for d in derived_data:
                        child_name = d.get("child_name")
                        parent_sample = d.get("parent_sample")
                        kind = d.get("kind")
                        tool = d.get("tool")
                        cmd_template = d.get("cmd_template")
                        params = d.get("params", {})

                        if not all([child_name, parent_sample, kind, tool, cmd_template]):
                            raise ValueError("derived_samples entries must include child_name, parent_sample, kind, tool, cmd_template")

                        # Resolve parent sample id; prefer current seqrun if provided, else global by name
                        parent_id = None
                        if 'seqrun_id' in locals() and seqrun_id is not None:
                            parent_id = db_api.get_sample_id(ctx.db, seqrun_id, parent_sample)
                        if parent_id is None:
                            parent_id = db_api.get_sample_id_by_name(ctx.db, parent_sample)
                        if parent_id is None:
                            raise ValueError(f"Could not resolve parent sample '{parent_sample}' in sequencing_run for derived sample '{child_name}'")

                        ds_id = db_api.upsert_derived_sample(
                            ctx.db, parent_id, child_name, kind, tool, cmd_template, params
                        )
                        log.debug("Upserted derived_sample id=%s for child '%s' (parent s_id=%s)", ds_id, child_name, parent_id)
                # Only link probing_reactions when actual parent samples are provided
                if samples_data:
                    # Ensure seqrun_id is available (created above)
                    if seqrun_id is None:
                        raise ValueError("'samples' provided but sequencing_run is missing or invalid.")

                    # Map user reaction_group labels to numeric rg_ids for this import
                    rg_labels = [s.get("reaction_group") for s in samples_data]
                    if any(lbl is None for lbl in rg_labels):
                        raise ValueError("Each sample must have a 'reaction_group' label in the YAML.")
                    unique_labels = []
                    seen = set()
                    for lbl in rg_labels:
                        if lbl not in seen:
                            unique_labels.append(lbl)
                            seen.add(lbl)

                    # Resolve each label to an rg_id, reusing existing ids if present
                    max_rg = db_api.get_max_reaction_group_id(ctx.db)
                    next_rg = max_rg
                    label_to_rgid = {}
                    for lbl in unique_labels:
                        existing = db_api.find_rg_id_by_label(ctx.db, lbl)
                        if existing is not None:
                            label_to_rgid[lbl] = existing
                        else:
                            next_rg += 1
                            label_to_rgid[lbl] = next_rg

                    # Ensure a reaction_groups row exists for each rg_id/label
                    for lbl, rgid in label_to_rgid.items():
                        db_api.upsert_reaction_group(ctx.db, rgid, lbl)

                    # For each sample, insert probing_reaction and pair in reaction_groups
                    for s in samples_data:
                        sample_name = s.get("sample_name")
                        fq_dir = s.get("fq_dir")
                        # seqrun_id is guaranteed non-None from the guard above
                        assert seqrun_id is not None
                        s_id = db_api.get_sample_id(ctx.db, seqrun_id, sample_name, fq_dir)
                        if s_id is None:
                            raise ValueError(f"Could not resolve sequencing sample id for name='{sample_name}', fq_dir='{fq_dir}'")

                        # Resolve construct for this sample
                        construct_ref = s.get("construct")
                        local_construct_id = default_construct_id
                        if construct_ref not in (None, ""):
                            key = str(construct_ref).lower()
                            if key not in construct_cache:
                                construct_cache[key] = db_api.get_construct_id_by_disp_name(ctx.db, construct_ref)
                            local_construct_id = construct_cache.get(key)

                        if local_construct_id is None:
                            ref_msg = construct_ref or "<unspecified>"
                            raise ValueError(
                                f"Sample '{sample_name}' references construct '{ref_msg}', but no matching construct was found. "
                                "Create or import that construct (by disp_name) before adding samples."
                            )

                        # Resolve buffer for this sample
                        buffer_ref = s.get("buffer")
                        local_buffer_id = default_buffer_id
                        if buffer_ref not in (None, ""):
                            key = str(buffer_ref).lower()
                            if key not in buffer_cache:
                                buffer_cache[key] = db_api.get_buffer_id_by_name_or_disp(ctx.db, buffer_ref)
                            local_buffer_id = buffer_cache.get(key)

                        if local_buffer_id is None:
                            ref_msg = buffer_ref or "<unspecified>"
                            raise ValueError(
                                f"Sample '{sample_name}' references buffer '{ref_msg}', but no matching buffer was found. "
                                "Create or import that buffer (by disp_name or name) before adding samples."
                            )

                        reaction = {
                            "temperature": int(s.get("temperature")),
                            "replicate": int(s.get("replicate")),
                            "reaction_time": int(s.get("reaction_time")),
                            "probe_concentration": float(s.get("probe_concentration")),
                            "probe": str(s.get("probe")),
                            "RT": str(s.get("RT")),
                            "done_by": str(s.get("done_by", "NO_DONEBY")),
                            "treated": int(s.get("treated", 1)),
                            "buffer_id": int(local_buffer_id),
                            "construct_id": int(local_construct_id),
                            "rg_id": int(label_to_rgid[s.get("reaction_group")]),
                            "s_id": int(s_id),
                        }

                        rxn_id = db_api.insert_probing_reaction(ctx.db, reaction)
                        if rxn_id is None:
                            raise RuntimeError(f"Failed to insert probing_reaction for sample '{sample_name}'")

                        log.debug("Inserted reaction id=%s into group rg_id=%s (label=%s)", rxn_id, reaction["rg_id"], s.get("reaction_group"))
            
            log.info("'create' task consumption completed successfully.")

        except Exception as e:
            log.exception("An error occurred during the 'create' task consumption: %s", e)
            # Re-raise the exception to ensure the task is marked as failed
            raise

        # Pure-Python ingestion; no external command required.
        return None
