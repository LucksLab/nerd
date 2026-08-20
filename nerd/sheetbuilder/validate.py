"""Validation: one pure function over (sheet, catalog) -> issues.

This drives two things at once, which is why it is a single pass rather
than scattered checks: the header bar that tells the user what is still
missing (and hands the entity gaps to the creation wizards as a work
queue), and the gate that stops Generate from emitting a sheet create.py
would reject.

Checks mirror create.py's own behaviour: required columns non-empty,
construct/buffer/sequencing_run references resolvable, numeric columns
numeric, sample_name unique, and fastq files present on disk for local
paths (remote/Quest paths are reported as unverified rather than missing,
since nerd cannot stat them from here either).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .catalog import EntityCatalog
from .model import ENTITY_COLUMNS, REQUIRED_COLUMNS, Sheet
from nerd.fastq_sources import FastqSourceError, LOCAL, SRA, normalize_source, profile_for_source

ERROR = "error"
WARNING = "warning"


def _text(value: Any) -> str:
    """Stringify a cell without treating falsy-but-real values as empty.

    treated=0 (untreated) and temperature=0 are legitimate values; a plain
    `value or ""` would silently report them as blank.
    """
    if value is None:
        return ""
    return str(value).strip()

NUMERIC_COLUMNS = ("temperature", "reaction_time", "probe_concentration", "treated")

@dataclass
class Issue:
    severity: str
    code: str
    message: str
    row_uid: Optional[int] = None
    column: Optional[str] = None
    value: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def validate(
    sheet: Sheet,
    catalog: EntityCatalog,
    project_dir: Optional[str] = None,
    check_files: bool = True,
    executors: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    issues: List[Issue] = []

    if not sheet.rows:
        return _summarize([Issue(WARNING, "empty_sheet", "The sheet has no rows yet.")])

    # --- required columns ---
    for row in sheet.rows:
        for column in REQUIRED_COLUMNS:
            value = _text(row.get(column))
            if not value:
                issues.append(Issue(
                    ERROR, "missing_required",
                    "%s is required by nerd create but is empty." % column,
                    row_uid=row.uid, column=column,
                ))

    # --- numeric columns ---
    for row in sheet.rows:
        for column in NUMERIC_COLUMNS:
            value = _text(row.get(column))
            if not value:
                continue
            try:
                float(value)
            except ValueError:
                issues.append(Issue(
                    ERROR, "not_numeric",
                    "%s must be numeric, got %r." % (column, value),
                    row_uid=row.uid, column=column, value=value,
                ))

    # --- duplicate sample_name ---
    seen: Dict[str, int] = {}
    for row in sheet.rows:
        name = _text(row.get("sample_name"))
        if not name:
            continue
        if name in seen:
            issues.append(Issue(
                ERROR, "duplicate_sample_name",
                "Duplicate sample_name %r (also on row %d)." % (name, seen[name]),
                row_uid=row.uid, column="sample_name", value=name,
            ))
        else:
            seen[name] = row.uid

    # --- entity references (bulk, one index per column) ---
    for column, entity_type in ENTITY_COLUMNS.items():
        values = sheet.distinct(column)
        if not values:
            continue
        statuses = catalog.resolve_many(entity_type, values)
        missing = [v for v, status in statuses.items() if status == "missing"]
        for value in missing:
            issues.append(Issue(
                ERROR, "unresolved_entity",
                "No %s named %r exists yet -- create it." % (entity_type.replace("_", " "), value),
                column=column, value=value,
            ))

    # --- fastq presence ---
    if check_files:
        unverified = 0
        sra_placeholders = 0
        for row in sheet.rows:
            fq_dir = _text(row.get("fq_dir"))
            if not fq_dir:
                continue
            try:
                source = normalize_source(row.get("fq_source"))
                if source not in {LOCAL, SRA}:
                    profile_for_source(source, executors or {})
            except FastqSourceError as exc:
                issues.append(Issue(
                    ERROR, "invalid_fq_source", str(exc), row_uid=row.uid,
                    column="fq_source", value=_text(row.get("fq_source")),
                ))
                continue
            if source == SRA:
                sra_placeholders += 1
                continue
            if source != LOCAL:
                unverified += 1
                continue
            base = Path(fq_dir)
            if not base.is_absolute() and project_dir:
                base = Path(project_dir) / "configs" / fq_dir
            if not base.is_dir():
                issues.append(Issue(
                    ERROR, "missing_fq_dir",
                    "fq_dir does not exist: %s" % fq_dir,
                    row_uid=row.uid, column="fq_dir", value=fq_dir,
                ))
                continue
            for column in ("r1_file", "r2_file"):
                filename = _text(row.get(column))
                if filename and not (base / filename).is_file():
                    issues.append(Issue(
                        ERROR, "missing_fastq",
                        "%s not found in fq_dir: %s" % (column, filename),
                        row_uid=row.uid, column=column, value=filename,
                    ))
        if unverified:
            issues.append(Issue(
                WARNING, "remote_fq_dir",
                "%d row(s) point at a cluster path that cannot be checked from "
                "the table; the selected remote HPC alias is checked during listing "
                "and nerd create." % unverified,
            ))
        if sra_placeholders:
            issues.append(Issue(
                WARNING, "sra_placeholder",
                "%d row(s) use the reserved SRA source; automatic SRA pulling is not implemented yet."
                % sra_placeholders,
            ))

    return _summarize(issues)


def _summarize(issues: List[Issue]) -> Dict[str, Any]:
    errors = [i for i in issues if i.severity == ERROR]
    warnings = [i for i in issues if i.severity == WARNING]

    # Entity gaps become the resolution queue the wizards consume.
    queue: List[Dict[str, str]] = []
    seen = set()
    for issue in errors:
        if issue.code != "unresolved_entity":
            continue
        key = (issue.column, issue.value)
        if key in seen:
            continue
        seen.add(key)
        queue.append({
            "entity_type": ENTITY_COLUMNS.get(issue.column or "", ""),
            "column": issue.column or "",
            "value": issue.value or "",
        })

    by_code: Dict[str, int] = {}
    for issue in errors + warnings:
        by_code[issue.code] = by_code.get(issue.code, 0) + 1

    return {
        "ok": not errors,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "by_code": by_code,
        "resolution_queue": queue,
        # Cap the transported list: the counts above stay exact, but a
        # 1000-row sheet with an empty column should not ship 1000 issues.
        "issues": [i.to_dict() for i in (errors + warnings)[:300]],
        "truncated": len(errors) + len(warnings) > 300,
    }
