"""Write the sheet out as nerd `create` inputs.

Default mode is "hybrid", which is what the demo project already does and
what scales: buffers/constructs/sequencing runs inline in a YAML (a
handful of entries, and YAML keeps the nt_info references readable), and
the samples as a CSV (hundreds to thousands of rows, where YAML would be
unusable). The user should not have to think about this split -- it is the
default and the UI just says "generate".

`csv` mode writes every table as CSV instead, for anyone who wants to hand
the sheets to an existing config of their own.
"""
from __future__ import annotations

import csv
from pathlib import Path
import shlex
from typing import Any, Dict, List, Optional

import yaml

from nerd.project import ProjectConfig

from .catalog import EntityCatalog
from .model import SAMPLE_COLUMNS, Sheet

HYBRID = "hybrid"
CSV_ONLY = "csv"


def _write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="" + utf-8 (no BOM): the existing demo sheet carries a BOM
    # that makes its first header cell "﻿sequencing_run_name".
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_yaml(path: Path, document: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(document, sort_keys=False, allow_unicode=True))


def write_nt_info(path: Path, nt_rows: List[Dict[str, Any]]) -> None:
    _write_csv(path, ["site", "base", "base_region"], nt_rows)


def default_nt_rows(sequence: str) -> List[Dict[str, Any]]:
    """1-based numbering over the sequence, matching nerd's own fallback."""
    return [
        {"site": index + 1, "base": base.upper(), "base_region": "1"}
        for index, base in enumerate(sequence or "")
    ]


def validate_primer_annotations(nt_rows: List[Dict[str, Any]]) -> None:
    """Validate the manually assigned primer-site labels for a construct."""
    regions = [str(row.get("base_region", "")).strip() for row in nt_rows]
    invalid = sorted({region for region in regions if region not in {"0", "1", "2"}})
    if invalid:
        raise ValueError(
            "Primer-site labels must be 0 (5'-end primer), 1 (target), or "
            "2 (3'-end primer/RT)."
        )
    if "1" not in regions:
        raise ValueError("Primer-site annotations must include a target region labeled 1.")
    if all(region == "1" for region in regions):
        raise ValueError(
            "Manually annotate the primer sites before creating this construct; "
            "the annotation cannot consist entirely of 1s. Label each region as "
            "0 (5'-end primer), 1 (target), or 2 (3'-end primer/RT). A 0 region "
            "is not required."
        )


def export(
    sheet: Sheet,
    catalog: EntityCatalog,
    project_dir: str,
    label: str,
    mode: str = HYBRID,
    db_path: Optional[str] = None,
    project_config: Optional[ProjectConfig] = None,
) -> Dict[str, Any]:
    project = project_config.root if project_config is not None else Path(project_dir)
    configs = project / "configs"
    to_import = configs / "to_import"
    nt_info_dir = configs / "nt_info"

    written: List[str] = []

    # Constructs: nt_rows go to their own CSV, referenced by relative path,
    # exactly as create.py resolves it (relative to <label>/configs/).
    constructs: List[Dict[str, Any]] = []
    for record in catalog.staged.get("construct", []):
        entry = {k: v for k, v in record.items() if k != "nt_rows"}
        nt_rows = record.get("nt_rows")
        if nt_rows:
            filename = "%s.csv" % str(record.get("disp_name") or "construct").replace("/", "_")
            write_nt_info(nt_info_dir / filename, nt_rows)
            written.append(str(nt_info_dir / filename))
            entry["nt_info"] = "nt_info/%s" % filename
        constructs.append(entry)

    buffers = [dict(r) for r in catalog.staged.get("buffer", [])]
    seqruns = [dict(r) for r in catalog.staged.get("sequencing_run", [])]

    rows = [{c: row.get(c) for c in SAMPLE_COLUMNS} for row in sheet.rows]

    samples_csv = to_import / "probing_samples.csv"
    _write_csv(samples_csv, SAMPLE_COLUMNS, rows)
    written.append(str(samples_csv))

    # Phase 4 owns output/executor defaults in project.toml. Omitting them
    # here lets resolve_config inherit those settings. Legacy projects keep
    # an explicit project-root output directory.
    run_block = (
        {"label": label}
        if project_config is not None
        else {"output_dir": "..", "label": label, "backend": "local"}
    )

    meta_written = False
    if mode == CSV_ONLY:
        if buffers:
            path = to_import / "buffers.csv"
            _write_csv(path, ["name", "disp_name", "pH", "composition"], buffers)
            written.append(str(path))
        if constructs:
            path = to_import / "constructs.csv"
            _write_csv(path, ["family", "name", "version", "sequence", "disp_name", "nt_info"], constructs)
            written.append(str(path))
        if seqruns:
            path = to_import / "sequencing_runs.csv"
            _write_csv(path, ["run_name", "date", "sequencer", "run_manager"], seqruns)
            written.append(str(path))
        meta_config = configs / "create_meta.yaml"
        create_block: Dict[str, Any] = {}
        if buffers:
            create_block["buffers"] = "to_import/buffers.csv"
        if constructs:
            create_block["constructs"] = "to_import/constructs.csv"
        if seqruns:
            create_block["sequencing_runs"] = "to_import/sequencing_runs.csv"
        if create_block:
            _write_yaml(meta_config, {"run": run_block, "create": create_block})
            written.append(str(meta_config))
            meta_written = True
    else:
        meta_config = configs / "create_meta.yaml"
        create_block = {}
        if buffers:
            create_block["buffers"] = buffers
        if constructs:
            create_block["constructs"] = constructs
        if seqruns:
            create_block["sequencing_runs"] = seqruns
        if create_block:
            _write_yaml(meta_config, {"run": run_block, "create": create_block})
            written.append(str(meta_config))
            meta_written = True

    samples_config = configs / "create_probing_samples.yaml"
    _write_yaml(samples_config, {
        "run": run_block,
        "create": {"samples": "to_import/probing_samples.csv"},
    })
    written.append(str(samples_config))

    def command_for(config_path: Path) -> str:
        args = ["nerd"]
        if project_config is not None:
            args.extend(["--project", str(project_config.root)])
        uses_database_override = db_path is not None and (
            project_config is None
            or Path(db_path).expanduser().resolve() != project_config.database
        )
        if uses_database_override:
            args.extend(["--db", str(db_path)])
        args.extend(["run", "create", str(config_path)])
        return shlex.join(args)

    commands: List[str] = []
    if meta_written:
        commands.append(command_for(configs / "create_meta.yaml"))
    commands.append(command_for(samples_config))

    return {
        "written": written,
        "commands": commands,
        "mode": mode,
        "sample_count": len(rows),
        "entity_counts": {
            "constructs": len(constructs), "buffers": len(buffers),
            "sequencing_runs": len(seqruns),
        },
    }
