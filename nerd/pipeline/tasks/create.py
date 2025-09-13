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
            
            log.info("'create' task consumption completed successfully.")

        except Exception as e:
            log.exception("An error occurred during the 'create' task consumption: %s", e)
            # Re-raise the exception to ensure the task is marked as failed
            raise

        # Pure-Python ingestion; no external command required.
        return None
