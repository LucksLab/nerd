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
from typing import Any, Dict, List, Optional

import yaml

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


def export(
    sheet: Sheet,
    catalog: EntityCatalog,
    project_dir: str,
    label: str,
    mode: str = HYBRID,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    project = Path(project_dir)
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

    run_block = {"output_dir": ".", "label": label, "backend": "local"}

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

    samples_config = configs / "create_probing_samples.yaml"
    _write_yaml(samples_config, {
        "run": run_block,
        "create": {"samples": "to_import/probing_samples.csv"},
    })
    written.append(str(samples_config))

    db_flag = " --db %s" % db_path if db_path else ""
    commands = []
    if (configs / "create_meta.yaml").exists():
        commands.append("nerd run create%s %s" % (db_flag, configs / "create_meta.yaml"))
    commands.append("nerd run create%s %s" % (db_flag, samples_config))

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
