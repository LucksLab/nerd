# nerd/pipeline/tasks/create.py
"""
Task for creating and ingesting initial data from a YAML configuration file.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List

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

    def prepare(self, cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Validates the presence of the 'create' section in the config
        and checks for the existence of specified file paths.
        """
        if self.name not in cfg:
            raise ValueError(f"Configuration must contain a '{self.name}' section.")
        
        create_cfg = cfg[self.name]
        run_cfg = cfg.get("run", {})
        
        # --- Validate paths ---
        output_dir = Path(run_cfg.get("output_dir", ".")).resolve()
        label_dir = output_dir / run_cfg.get("label", "")
        
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
        construct_cfg = create_cfg.get("construct", {})
        nt_info_path_str = construct_cfg.get("nt_info")
        if nt_info_path_str:
            # Assumes nt_info is always within a 'configs' directory 
            # that is a sibling of the label directory.
            configs_dir = label_dir / "configs"
            nt_info_path = configs_dir / nt_info_path_str
            
            if not nt_info_path.is_file():
                raise FileNotFoundError(f"nt_info file not found at assumed path: {nt_info_path}")

        log.debug("'create' task prepared with inputs from config.")
        return create_cfg, {}  # Return the 'create' block as inputs

    def command(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Optional[str]:
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
            construct_data = inputs.get("construct")
            buffer_data = inputs.get("buffer")
            sequencing_run_data = inputs.get("sequencing_run")
            samples_data = inputs.get("samples", [])
            derived_data = inputs.get("derived_samples", [])

            with ctx.db: # Use the connection as a context manager for transactions
                if construct_data:
                    log.info("Inserting construct: %s", construct_data.get('disp_name'))
                    # Attach nt_rows if nt_info CSV provided
                    nt_info_rel = construct_data.get("nt_info")
                    if nt_info_rel:
                        label_dir = Path(ctx.output_dir) / ctx.label
                        configs_dir = label_dir / "configs"
                        nt_info_path = (configs_dir / nt_info_rel)
                        nt_rows: List[Dict[str, Any]] = db_api.prep_nt_rows(nt_info_path)
                        # pass parsed rows along to db upsert
                        construct_data = {**construct_data, "nt_rows": nt_rows}

                    construct_id = db_api.upsert_construct(ctx.db, construct_data)
                    log.debug("Construct upserted with id=%s", construct_id)

                if buffer_data:
                    log.info("Inserting buffer: %s", buffer_data.get('disp_name'))
                    buffer_id = db_api.upsert_buffer(ctx.db, buffer_data)
                    log.debug("Buffer upserted with id=%s", buffer_id)

                if sequencing_run_data:
                    log.info("Inserting sequencing run: %s", sequencing_run_data.get('run_name'))
                    seqrun_id = db_api.upsert_sequencing_run(ctx.db, sequencing_run_data)
                    log.debug("Sequencing run upserted with id=%s", seqrun_id)

                if samples_data:
                    log.info("Inserting %d samples.", len(samples_data))
                    if not sequencing_run_data:
                        raise ValueError("'samples' provided without 'sequencing_run' block to associate seqrun_id.")
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
                        if 'seqrun_id' in locals():
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
                    if construct_data is None or buffer_data is None:
                        raise ValueError("'samples' requires 'construct' and 'buffer' blocks to link probing_reactions.")

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
                        s_id = db_api.get_sample_id(ctx.db, seqrun_id, sample_name, fq_dir)
                        if s_id is None:
                            raise ValueError(f"Could not resolve sequencing sample id for name='{sample_name}', fq_dir='{fq_dir}'")

                        reaction = {
                            "temperature": int(s.get("temperature")),
                            "replicate": int(s.get("replicate")),
                            "reaction_time": int(s.get("reaction_time")),
                            "probe_concentration": float(s.get("probe_concentration")),
                            "probe": str(s.get("probe")),
                            "RT": str(s.get("RT")),
                            "done_by": str(s.get("done_by", "NO_DONEBY")),
                            "treated": int(s.get("treated", 1)),
                            "buffer_id": int(buffer_id) if buffer_id is not None else None,
                            "construct_id": int(construct_id) if construct_id is not None else None,
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
