"""FastAPI layer for the nerd create, edit, and view workspaces.

Deliberately thin: every endpoint translates HTTP to a call into
nerd.sheetbuilder and back. All the logic -- provenance, pattern
compilation, entity resolution, validation, export -- lives in that
package, which has no web dependency and is unit-testable on its own.

Every mutating endpoint returns the *whole* new state (rows, entity
resolution, validation summary) so the frontend never has to stitch
together partial updates or re-query to find out what changed.

Launch inside a Phase 4 project with ``nerd webui create``, ``edit``, ``view``,
or ``analyze``; each command also accepts ``--project`` and ``--db`` overrides.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from nerd.sheetbuilder import export as export_mod
from nerd.sheetbuilder import fillers
from nerd.sheetbuilder.catalog import ENTITY_FIELDS, LOOKUP_FIELDS
from nerd.sheetbuilder.groups import (
    GROUP_KEY_COLUMNS, ReactionGroup, parse_ladder_text,
)
from nerd.sheetbuilder.model import ENTITY_COLUMNS, SAMPLE_COLUMNS, MANUAL
from nerd.sheetbuilder.pattern import TokenSpec, compile_pattern, parse_name
from nerd.sheetbuilder.pattern.compile import PatternError
from nerd.sheetbuilder.session import Session
from nerd.sheetbuilder.validate import validate
from nerd.fastq_sources import (
    FastqSourceError, LOCAL, SRA, list_remote_directory, normalize_source,
    profile_for_source, remote_profiles,
)
from nerd.webui.analysis import (
    analysis_catalog, kinetic_rates, modification_rates, timecourse_data,
    timecourse_options,
)

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="nerd web UI")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

session = Session()
_server: Optional[Any] = None
WEBUI_MODES = {"create", "edit", "view", "analyze"}
_mode = "create"


# ---------------------------------------------------------------- helpers

def _require_session() -> Session:
    if not session.connected:
        raise HTTPException(400, "Connect to a project folder first.")
    return session


def set_server(server: Optional[Any]) -> None:
    """Register the running uvicorn server so the local UI can stop it."""
    global _server
    _server = server


def set_mode(mode: str) -> None:
    """Select the workspace exposed by this server process."""
    global _mode
    if mode not in WEBUI_MODES:
        raise ValueError("Unknown Web UI mode %r." % mode)
    _mode = mode


def _require_mode(expected: str) -> None:
    if _mode != expected:
        raise HTTPException(403, "This operation is only available in Web UI %s mode." % expected)


def _request_server_shutdown() -> None:
    if _server is not None:
        _server.should_exit = True


def _state(save: bool = True) -> Dict[str, Any]:
    """The single payload every mutation returns."""
    if save and session.connected:
        session.save_draft()

    resolution: Dict[str, Dict[str, str]] = {}
    for column, entity_type in ENTITY_COLUMNS.items():
        values = session.sheet.distinct(column)
        if values:
            resolution[column] = session.catalog.resolve_many(entity_type, values)

    report = validate(
        session.sheet,
        session.catalog,
        project_dir=str(session.project_dir) if session.project_dir else None,
        executors=session.project_config.executors if session.project_config else {},
    ) if session.sheet.rows else {
        "ok": False, "error_count": 0, "warning_count": 0, "by_code": {},
        "resolution_queue": [], "issues": [], "truncated": False,
    }

    return {
        "mode": _mode,
        "connected": session.connected,
        "project_dir": str(session.project_dir) if session.project_dir else None,
        "project_id": session.project_config.name if session.project_config else None,
        "project_file": str(session.project_config.source_path) if session.project_config else None,
        "db_path": str(session.db_path) if session.db_path else None,
        "output_dir": str(session.output_dir) if session.output_dir else None,
        "label": session.label,
        "pattern": session.pattern,
        "fq_dir": session.fq_dir,
        "fq_source": session.fq_source,
        "fq_source_choices": [
            {"value": LOCAL, "label": "Local — this computer"},
            *remote_profiles(session.project_config.executors if session.project_config else {}),
            {"value": SRA, "label": "SRA — coming soon", "disabled": True},
        ],
        "columns": SAMPLE_COLUMNS,
        "entity_columns": ENTITY_COLUMNS,
        "rows": [row.to_dict() for row in session.sheet.rows],
        "resolution": resolution,
        "validation": report,
        "staged": session.catalog.staged,
        "db_entity_counts": {k: len(v) for k, v in session.catalog.db.items()},
        "groups": session.groups.summary(session.sheet) if session.sheet.rows else [],
        "ladders": [l.to_dict() for l in session.groups.ladders.values()],
        "group_key_columns": list(GROUP_KEY_COLUMNS),
        "label_template": session.groups.label_template,
    }


@app.middleware("http")
async def enforce_workspace_mode(request: Request, call_next):
    """Keep view/analyze read-only and creation writes out of maintenance mode."""
    if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
        always_allowed = {"/api/session/connect", "/api/shutdown"}
        edit_allowed = {
            "/api/database/constructs/base-regions",
            "/api/database/probe-samples",
            "/api/database/probe-samples/bulk",
        }
        path = request.url.path
        if path not in always_allowed:
            if _mode in {"view", "analyze"}:
                return JSONResponse(
                    {"detail": "Web UI %s mode is read-only." % _mode}, status_code=403,
                )
            if _mode == "edit" and path not in edit_allowed:
                return JSONResponse(
                    {"detail": "This creation operation is unavailable in Web UI edit mode."},
                    status_code=403,
                )
            if _mode == "create" and path in edit_allowed:
                return JSONResponse(
                    {"detail": "Database maintenance requires Web UI edit mode."},
                    status_code=403,
                )
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        # The WebUI is a local development-style app. Never let a browser keep
        # an old HTML/JS bundle across server restarts and source updates.
        response.headers["Cache-Control"] = "no-store"
    return response


# ---------------------------------------------------------------- pages

@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


# ---------------------------------------------------------------- session

class ConnectRequest(BaseModel):
    project_dir: str
    db_path: Optional[str] = None
    label: str = "sample_import"


@app.post("/api/session/connect")
def connect(req: ConnectRequest) -> Dict[str, Any]:
    try:
        info = session.connect(
            req.project_dir, req.db_path, req.label, read_only=_mode in {"view", "analyze"}
        )
    except Exception as exc:
        raise HTTPException(400, "Could not open project: %s" % exc)
    return {**info, "state": _state(save=False)}


@app.get("/api/state")
def get_state() -> Dict[str, Any]:
    return _state(save=False)


# ---------------------------------------------------------------- analysis

def _analysis_connection() -> Any:
    active = _require_session()
    if active.conn is None:
        raise HTTPException(400, "No database connection is available.")
    return active.conn


@app.get("/api/analyze/catalog")
def get_analysis_catalog() -> Dict[str, Any]:
    return analysis_catalog(_analysis_connection())


@app.get("/api/analyze/modification-rates")
def get_modification_rates(
    run_id: List[int] = Query(...),
    valtype: str = Query(...),
) -> Dict[str, Any]:
    try:
        return modification_rates(_analysis_connection(), run_id, valtype)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/analyze/kinetic-rates")
def get_kinetic_rates(
    rg_id: List[int] = Query(...),
    valtype: str = Query(...),
) -> Dict[str, Any]:
    try:
        return kinetic_rates(_analysis_connection(), rg_id, valtype)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/analyze/timecourse-options")
def get_timecourse_options(rg_id: int = Query(...)) -> Dict[str, Any]:
    return timecourse_options(_analysis_connection(), rg_id)


@app.get("/api/analyze/timecourse")
def get_timecourse_data(
    rg_id: int = Query(...),
    nt_id: List[int] = Query(...),
    valtype: str = Query(...),
    include_fits: bool = Query(True),
) -> Dict[str, Any]:
    try:
        return timecourse_data(
            _analysis_connection(), rg_id, nt_id, valtype, include_fits=include_fits,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


# ---------------------------------------------------------------- tokens

@app.get("/api/tokens")
def list_tokens() -> Dict[str, Any]:
    return {
        name: {
            "name": spec.name, "maps_to": spec.maps_to,
            "normalize": spec.normalize, "description": spec.description,
            "value_map": spec.value_map, "match": spec.match,
        }
        for name, spec in session.registry.all().items()
    }


class TokenUpsert(BaseModel):
    name: str
    match: Optional[str] = None
    maps_to: Optional[str] = None
    normalize: Optional[str] = None
    value_map: Dict[str, str] = {}
    description: str = ""


@app.post("/api/tokens")
def upsert_token(req: TokenUpsert) -> Dict[str, Any]:
    if req.maps_to and req.maps_to not in SAMPLE_COLUMNS:
        raise HTTPException(400, "maps_to must be one of the sheet columns.")
    session.registry.register(TokenSpec(**req.dict()))
    session.registry.save()
    return {"ok": True}


@app.delete("/api/tokens/{name}")
def delete_token(name: str) -> Dict[str, Any]:
    if not session.registry.remove(name):
        raise HTTPException(404, "No token named %r." % name)
    session.registry.save()
    return {"ok": True}


# ---------------------------------------------------------------- ingest

class FastqIngest(BaseModel):
    fq_dir: str
    fq_source: str = LOCAL
    listing: Optional[str] = None   # pasted `ls` output
    scan_local: bool = False        # or scan fq_dir on this machine
    scan_source: bool = False       # list through the selected local/remote source
    replace: bool = True


@app.post("/api/ingest/fastq")
def ingest_fastq(req: FastqIngest) -> Dict[str, Any]:
    _require_session()
    try:
        source = normalize_source(req.fq_source)
    except FastqSourceError as exc:
        raise HTTPException(400, str(exc))
    if source == SRA:
        raise HTTPException(400, "SRA pulling is reserved for a future release; choose local or a remote HPC alias.")
    if req.scan_local or (req.scan_source and source == LOCAL):
        try:
            filenames = fillers.scan_directory(req.fq_dir)
        except FileNotFoundError as exc:
            raise HTTPException(400, str(exc))
        if not filenames:
            raise HTTPException(400, "No fastq files found in %s" % req.fq_dir)
    elif not req.scan_source:
        filenames = fillers.parse_listing(req.listing or "")
        if not filenames:
            raise HTTPException(
                400,
                "No fastq filenames found in that listing. Paste the output of "
                "`ls` (or `ls -l`) from the folder holding the fastq files.",
            )
    else:
        try:
            executors = session.project_config.executors if session.project_config else {}
            filenames = list_remote_directory(req.fq_dir, profile_for_source(source, executors))
            filenames = [
                name for name in filenames
                if name.lower().endswith(fillers.FASTQ_SUFFIXES)
            ]
        except (FastqSourceError, FileNotFoundError) as exc:
            raise HTTPException(400, str(exc))
        if not filenames:
            raise HTTPException(400, "No files found in %s via %s." % (req.fq_dir, source))
    result = fillers.fastq_scan(
        session.sheet, filenames, req.fq_dir, fq_source=source, replace=req.replace
    )
    session.fq_dir = req.fq_dir
    session.fq_source = source
    return {"result": result, "state": _state()}


class NamesIngest(BaseModel):
    names: str
    replace: bool = True


@app.post("/api/ingest/names")
def ingest_names(req: NamesIngest) -> Dict[str, Any]:
    """Fallback path: plain sample names, no fastq files to pair."""
    _require_session()
    names = [n.strip() for n in (req.names or "").splitlines() if n.strip()]
    if not names:
        raise HTTPException(400, "No sample names provided.")
    if req.replace:
        session.sheet.clear()
    for name in names:
        session.sheet.add_row({"sample_name": name}, origin=fillers.FASTQ)
    return {"result": {"added": len(names)}, "state": _state()}


# ---------------------------------------------------------------- fills

class PatternPreview(BaseModel):
    pattern: str
    limit: int = 6


@app.post("/api/pattern/preview")
def preview_pattern(req: PatternPreview) -> Dict[str, Any]:
    _require_session()
    try:
        compiled = compile_pattern(req.pattern, session.registry)
    except PatternError as exc:
        raise HTTPException(400, str(exc))
    rows = []
    for row in session.sheet.rows[: max(1, req.limit)]:
        name = str(row.get("sample_name") or "")
        parsed = parse_name(name, compiled, session.registry)
        rows.append({"sample_name": name, **parsed})
    return {"rows": rows, "warnings": compiled.warnings, "regex": compiled.regex.pattern}


class PatternApply(BaseModel):
    pattern: str
    uids: Optional[List[int]] = None
    force: bool = False


@app.post("/api/fill/pattern")
def fill_pattern(req: PatternApply) -> Dict[str, Any]:
    _require_session()
    try:
        result = fillers.pattern_fill(
            session.sheet, req.pattern, session.registry,
            uids=req.uids, force=req.force,
        )
    except PatternError as exc:
        raise HTTPException(400, str(exc))
    session.pattern = req.pattern
    return {"result": result, "state": _state()}


class BatchApply(BaseModel):
    column: str
    value: Any = ""
    uids: Optional[List[int]] = None


@app.post("/api/fill/batch")
def fill_batch(req: BatchApply) -> Dict[str, Any]:
    _require_session()
    try:
        result = fillers.batch_fill(session.sheet, req.column, req.value, uids=req.uids)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"result": result, "state": _state()}


# ---------------------------------------------------------------- rows

class CellEdit(BaseModel):
    uid: int
    column: str
    value: Any = ""


@app.post("/api/rows/cell")
def edit_cell(req: CellEdit) -> Dict[str, Any]:
    _require_session()
    row = session.sheet.by_uid(req.uid)
    if row is None:
        raise HTTPException(404, "No row %s." % req.uid)
    if req.column not in SAMPLE_COLUMNS:
        raise HTTPException(400, "Unknown column %r." % req.column)
    row.set(req.column, req.value, MANUAL, force=True)
    return {"state": _state()}


class CellEdits(BaseModel):
    edits: List[CellEdit]


@app.post("/api/rows/cells")
def edit_cells(req: CellEdits) -> Dict[str, Any]:
    """Apply a clipboard-sized group of create-table edits in one render."""
    _require_session()
    resolved = []
    for edit in req.edits:
        row = session.sheet.by_uid(edit.uid)
        if row is None:
            raise HTTPException(404, "No row %s." % edit.uid)
        if edit.column not in SAMPLE_COLUMNS:
            raise HTTPException(400, "Unknown column %r." % edit.column)
        resolved.append((row, edit))
    for row, edit in resolved:
        row.set(edit.column, edit.value, MANUAL, force=True)
    return {"edited": len(resolved), "state": _state()}


class AutofillDown(BaseModel):
    columns: List[str]
    source_uids: List[int]


@app.post("/api/rows/autofill-down")
def autofill_rows_down(req: AutofillDown) -> Dict[str, Any]:
    _require_session()
    try:
        result = fillers.autofill_down(session.sheet, req.columns, req.source_uids)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"result": result, "state": _state()}


class RowAdd(BaseModel):
    count: int = 1


@app.post("/api/rows/add")
def add_rows(req: RowAdd) -> Dict[str, Any]:
    _require_session()
    for _ in range(max(1, req.count)):
        session.sheet.add_row()
    return {"state": _state()}


class RowDelete(BaseModel):
    uids: List[int]


@app.post("/api/rows/delete")
def delete_rows(req: RowDelete) -> Dict[str, Any]:
    _require_session()
    removed = session.sheet.delete(req.uids)
    return {"result": {"removed": removed}, "state": _state()}


# ---------------------------------------------------------------- entities

@app.get("/api/entities/schema")
def entity_schema() -> Dict[str, Any]:
    return {"fields": ENTITY_FIELDS, "lookup_fields": LOOKUP_FIELDS}


@app.get("/api/entities")
def list_entities() -> Dict[str, Any]:
    return {"db": session.catalog.db, "staged": session.catalog.staged}


@app.get("/api/database/entities")
def database_entities() -> Dict[str, Any]:
    """Return the small metadata catalog used by the view/edit workspaces."""
    active = _require_session()
    if active.conn is None:
        raise HTTPException(400, "No database connection is available.")
    active.refresh_catalog()

    result = {
        entity_type: [dict(record) for record in records]
        for entity_type, records in active.catalog.db.items()
    }
    reference_queries = {
        "construct": (
            "SELECT COUNT(*) FROM probe_reactions WHERE construct_id = ?",
            "SELECT COUNT(*) FROM probe_tempgrad_groups WHERE construct_id = ?",
        ),
        "buffer": (
            "SELECT COUNT(*) FROM probe_reactions WHERE buffer_id = ?",
            "SELECT COUNT(*) FROM probe_tempgrad_groups WHERE buffer_id = ?",
        ),
        "sequencing_run": (
            "SELECT COUNT(*) FROM sequencing_samples WHERE seqrun_id = ?",
        ),
    }
    for entity_type, records in result.items():
        for record in records:
            record["reference_count"] = sum(
                int(active.conn.execute(query, (record["id"],)).fetchone()[0])
                for query in reference_queries.get(entity_type, ())
            )
    result["probe_sample"] = _database_probe_samples(active.conn)
    reaction_groups = [
        dict(row) for row in active.conn.execute(
            "SELECT rg_id, rg_label FROM probe_reaction_groups "
            "ORDER BY LOWER(COALESCE(rg_label, '')), rg_id"
        ).fetchall()
    ]
    return {
        "mode": _mode,
        "entities": result,
        "reaction_groups": reaction_groups,
    }


def _database_probe_samples(
    conn: Any, reaction_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Return the creation-sheet fields joined onto each probe reaction."""
    where = "WHERE pr.id = ?" if reaction_id is not None else ""
    rows = conn.execute(
        """
        SELECT
            pr.id AS reaction_id,
            ss.id AS sample_id,
            ss.sample_name,
            ss.seqrun_id,
            sr.run_name AS sequencing_run_name,
            ss.fq_source,
            ss.fq_dir,
            ss.r1_file,
            ss.r2_file,
            ss.to_drop,
            pr.rg_id,
            rg.rg_label AS reaction_group,
            pr.temperature,
            pr.replicate,
            pr.reaction_time,
            pr.probe,
            pr.probe_concentration,
            pr.rt_protocol AS RT,
            pr.treated,
            pr.buffer_id,
            b.disp_name AS buffer,
            pr.construct_id,
            c.disp_name AS construct,
            pr.done_by,
            (SELECT COUNT(*) FROM probe_fmod_runs fr WHERE fr.s_id = ss.id)
                AS reference_count
        FROM probe_reactions pr
        JOIN sequencing_samples ss ON ss.id = pr.s_id
        JOIN sequencing_runs sr ON sr.id = ss.seqrun_id
        JOIN probe_reaction_groups rg ON rg.rg_id = pr.rg_id
        JOIN meta_buffers b ON b.id = pr.buffer_id
        JOIN meta_constructs c ON c.id = pr.construct_id
        %s
        ORDER BY LOWER(ss.sample_name), pr.id
        """ % where,
        (reaction_id,) if reaction_id is not None else (),
    ).fetchall()
    return [dict(row) for row in rows]


class ProbeSampleUpdate(BaseModel):
    reaction_id: int
    sample_id: int
    sample_name: str
    seqrun_id: int
    fq_source: str
    fq_dir: str
    r1_file: str
    r2_file: str
    to_drop: int = 0
    rg_id: int
    temperature: float
    replicate: int
    reaction_time: float
    probe: str
    probe_concentration: float
    RT: str
    treated: int
    buffer_id: int
    construct_id: int
    done_by: str


@app.patch("/api/database/probe-samples")
def update_probe_sample(req: ProbeSampleUpdate) -> Dict[str, Any]:
    """Correct one joined sequencing-sample/probe-reaction record in place."""
    _require_mode("edit")
    active = _require_session()
    if active.conn is None:
        raise HTTPException(400, "No database connection is available.")

    try:
        with active.conn:
            saved, changes = _apply_probe_sample_update(active.conn, req)
    except sqlite3.IntegrityError as exc:
        raise HTTPException(
            409, "That edit conflicts with an existing database record: %s" % exc
        ) from exc

    audit_logged = _log_probe_sample_changes([
        {
            "reaction_id": req.reaction_id,
            "sample_id": req.sample_id,
            "changes": changes,
        }
    ])
    return {
        "ok": True,
        "changed": len(changes),
        "audit_logged": audit_logged,
        "record": saved,
    }


class ProbeSampleBulkUpdate(BaseModel):
    records: List[ProbeSampleUpdate]


@app.patch("/api/database/probe-samples/bulk")
def update_probe_samples_bulk(req: ProbeSampleBulkUpdate) -> Dict[str, Any]:
    """Apply spreadsheet edits atomically across multiple probe samples."""
    _require_mode("edit")
    active = _require_session()
    if active.conn is None:
        raise HTTPException(400, "No database connection is available.")
    if not req.records:
        raise HTTPException(400, "No changed probe samples were submitted.")
    reaction_ids = [record.reaction_id for record in req.records]
    if len(reaction_ids) != len(set(reaction_ids)):
        raise HTTPException(400, "Each probe reaction may be submitted only once.")

    saved_records: List[Dict[str, Any]] = []
    audit_changes: List[Dict[str, Any]] = []
    try:
        with active.conn:
            for record in req.records:
                saved, changes = _apply_probe_sample_update(active.conn, record)
                saved_records.append(saved)
                audit_changes.append({
                    "reaction_id": record.reaction_id,
                    "sample_id": record.sample_id,
                    "changes": changes,
                })
    except sqlite3.IntegrityError as exc:
        raise HTTPException(
            409, "Those edits conflict with an existing database record: %s" % exc
        ) from exc

    changed_fields = sum(len(item["changes"]) for item in audit_changes)
    return {
        "ok": True,
        "changed_records": sum(bool(item["changes"]) for item in audit_changes),
        "changed_fields": changed_fields,
        "audit_logged": _log_probe_sample_changes(audit_changes),
        "records": saved_records,
    }


def _apply_probe_sample_update(
    conn: Any, req: ProbeSampleUpdate,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Validate and execute one update inside the caller's transaction."""

    current = conn.execute(
        "SELECT pr.*, ss.sample_name, ss.seqrun_id, ss.fq_source, ss.fq_dir, "
        "ss.r1_file, ss.r2_file, ss.to_drop "
        "FROM probe_reactions pr JOIN sequencing_samples ss ON ss.id = pr.s_id "
        "WHERE pr.id = ? AND ss.id = ?",
        (req.reaction_id, req.sample_id),
    ).fetchone()
    if current is None:
        raise HTTPException(404, "No matching probe sample was found.")

    text_fields = {
        "sample_name": req.sample_name,
        "fq_source": req.fq_source,
        "fq_dir": req.fq_dir,
        "r1_file": req.r1_file,
        "r2_file": req.r2_file,
        "probe": req.probe,
        "RT": req.RT,
        "done_by": req.done_by,
    }
    missing = [name for name, value in text_fields.items() if not str(value).strip()]
    if missing:
        raise HTTPException(400, "Missing required field(s): %s" % ", ".join(missing))
    try:
        fq_source = normalize_source(req.fq_source)
    except FastqSourceError as exc:
        raise HTTPException(400, str(exc)) from exc
    if req.treated not in {0, 1, 2}:
        raise HTTPException(400, "treated must be 0 (untreated), 1 (treated), or 2 (mixed).")
    if req.to_drop not in {0, 1}:
        raise HTTPException(400, "to_drop must be 0 or 1.")

    foreign_keys = {
        "sequencing run": ("sequencing_runs", "id", req.seqrun_id),
        "reaction group": ("probe_reaction_groups", "rg_id", req.rg_id),
        "buffer": ("meta_buffers", "id", req.buffer_id),
        "construct": ("meta_constructs", "id", req.construct_id),
    }
    for label, (table, column, value) in foreign_keys.items():
        if conn.execute(
            "SELECT 1 FROM %s WHERE %s = ?" % (table, column), (value,)
        ).fetchone() is None:
            raise HTTPException(400, "The selected %s no longer exists." % label)

    before = next(
        row for row in _database_probe_samples(conn, req.reaction_id)
        if int(row["reaction_id"]) == req.reaction_id
    )
    conn.execute(
        "UPDATE sequencing_samples SET seqrun_id = ?, sample_name = ?, "
        "fq_source = ?, fq_dir = ?, r1_file = ?, r2_file = ?, to_drop = ? "
        "WHERE id = ?",
        (
            req.seqrun_id, req.sample_name.strip(), fq_source, req.fq_dir.strip(),
            req.r1_file.strip(), req.r2_file.strip(), req.to_drop, req.sample_id,
        ),
    )
    conn.execute(
        "UPDATE probe_reactions SET rg_id = ?, construct_id = ?, buffer_id = ?, "
        "temperature = ?, replicate = ?, reaction_time = ?, "
        "probe_concentration = ?, probe = ?, rt_protocol = ?, done_by = ?, "
        "treated = ? WHERE id = ?",
        (
            req.rg_id, req.construct_id, req.buffer_id, req.temperature,
            req.replicate, req.reaction_time, req.probe_concentration,
            req.probe.strip(), req.RT.strip(), req.done_by.strip(),
            req.treated, req.reaction_id,
        ),
    )

    saved = next(
        row for row in _database_probe_samples(conn, req.reaction_id)
        if int(row["reaction_id"]) == req.reaction_id
    )
    changes = {
        key: {"old": before.get(key), "new": saved.get(key)}
        for key in saved
        if key in before and before.get(key) != saved.get(key)
    }
    return saved, changes


def _log_probe_sample_changes(entries: List[Dict[str, Any]]) -> bool:
    changed = [entry for entry in entries if entry["changes"]]
    if not changed:
        return True
    try:
        _write_maintenance_log({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": "edit",
            "entity_type": "probe_sample",
            "operation": (
                "update_probe_sample" if len(changed) == 1
                else "bulk_update_probe_samples"
            ),
            "records": changed,
        })
    except OSError:
        return False
    return True


class EntityCreate(BaseModel):
    entity_type: str
    record: Dict[str, Any]
    nt_rows: Optional[List[Dict[str, Any]]] = None


@app.post("/api/entities")
def create_entity(req: EntityCreate) -> Dict[str, Any]:
    _require_session()
    if req.entity_type not in ENTITY_FIELDS:
        raise HTTPException(400, "Unknown entity type %r." % req.entity_type)

    record = dict(req.record)
    missing = [
        field["name"] for field in ENTITY_FIELDS[req.entity_type]
        if field.get("required") and not str(record.get(field["name"], "")).strip()
    ]
    if missing:
        raise HTTPException(400, "Missing required field(s): %s" % ", ".join(missing))

    collision = session.catalog.collision(req.entity_type, record)
    if collision:
        raise HTTPException(409, collision)

    if req.entity_type == "construct":
        try:
            record["nt_rows"] = req.nt_rows or export_mod.default_nt_rows(
                record.get("sequence", "")
            )
            export_mod.validate_primer_annotations(record["nt_rows"])
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    session.catalog.stage(req.entity_type, record)
    return {"state": _state()}


class EntityDelete(BaseModel):
    entity_type: str
    value: str


@app.post("/api/entities/delete")
def delete_entity(req: EntityDelete) -> Dict[str, Any]:
    _require_session()
    if not session.catalog.unstage(req.entity_type, req.value):
        raise HTTPException(404, "No staged %s named %r." % (req.entity_type, req.value))
    return {"state": _state()}


@app.get("/api/constructs/nt_rows")
def nt_rows(
    construct_id: Optional[int] = None,
    disp_name: Optional[str] = None,
    sequence: Optional[str] = None,
) -> Dict[str, Any]:
    """Seed the numbering grid: default 1-based, or copy an existing construct's."""
    if sequence and not disp_name and construct_id is None:
        try:
            rows = export_mod.default_nt_rows(sequence)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"nt_rows": rows, "source": "case-detected"}
    if construct_id is not None:
        active = _require_session()
        rows = active.conn.execute(
            "SELECT site, base, base_region FROM meta_nucleotides "
            "WHERE construct_id = ? ORDER BY site",
            (construct_id,),
        ).fetchall() if active.conn is not None else []
        if rows:
            template_rows = [dict(row) for row in rows]
            if sequence:
                try:
                    template_rows = export_mod.copy_nt_row_annotations(sequence, template_rows)
                except ValueError as exc:
                    raise HTTPException(400, str(exc)) from exc
            return {"nt_rows": template_rows, "source": "database"}
        raise HTTPException(404, "No numbering found for construct id %s." % construct_id)
    if not disp_name:
        raise HTTPException(400, "Pass construct_id=, sequence=, or disp_name=.")

    staged = next(
        (c for c in session.catalog.staged.get("construct", [])
         if str(c.get("disp_name", "")).lower() == disp_name.lower()), None
    )
    if staged and staged.get("nt_rows"):
        template_rows = staged["nt_rows"]
        if sequence:
            try:
                template_rows = export_mod.copy_nt_row_annotations(sequence, template_rows)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
        return {"nt_rows": template_rows, "source": "staged"}

    resolved = session.catalog.resolve("construct", disp_name)
    if resolved.status == "db" and session.conn is not None:
        rows = session.conn.execute(
            "SELECT site, base, base_region FROM meta_nucleotides "
            "WHERE construct_id = ? ORDER BY site",
            (resolved.entity_id,),
        ).fetchall()
        if rows:
            template_rows = [dict(r) for r in rows]
            if sequence:
                try:
                    template_rows = export_mod.copy_nt_row_annotations(sequence, template_rows)
                except ValueError as exc:
                    raise HTTPException(400, str(exc)) from exc
            return {"nt_rows": template_rows, "source": "database"}
    raise HTTPException(404, "No numbering found for %r." % disp_name)


class BaseRegionRow(BaseModel):
    site: int
    base_region: str


class BaseRegionUpdate(BaseModel):
    construct_id: int
    rows: List[BaseRegionRow]


def _write_maintenance_log(entry: Dict[str, Any]) -> None:
    if session.project_dir is None:
        return
    path = session.project_dir / ".nerd" / "maintenance.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


@app.patch("/api/database/constructs/base-regions")
def update_construct_base_regions(req: BaseRegionUpdate) -> Dict[str, Any]:
    """Correct a construct's region labels without changing nucleotide IDs."""
    _require_mode("edit")
    active = _require_session()
    if active.conn is None:
        raise HTTPException(400, "No database connection is available.")

    construct = active.conn.execute(
        "SELECT id, disp_name FROM meta_constructs WHERE id = ?", (req.construct_id,)
    ).fetchone()
    if construct is None:
        raise HTTPException(404, "No construct with id %s." % req.construct_id)

    current_rows = active.conn.execute(
        "SELECT id, site, base, base_region FROM meta_nucleotides "
        "WHERE construct_id = ? ORDER BY site",
        (req.construct_id,),
    ).fetchall()
    if not current_rows:
        raise HTTPException(400, "This construct has no nucleotide rows to edit.")

    provided = {row.site: row.base_region.strip() for row in req.rows}
    if len(provided) != len(req.rows):
        raise HTTPException(400, "Each nucleotide site must appear exactly once.")
    current_by_site = {int(row["site"]): row for row in current_rows}
    if set(provided) != set(current_by_site):
        raise HTTPException(400, "The submitted sites must exactly match the construct's nucleotide sites.")

    proposed_rows = [
        {
            "site": site,
            "base": current_by_site[site]["base"],
            "base_region": provided[site],
        }
        for site in sorted(provided)
    ]
    try:
        export_mod.validate_primer_annotations(proposed_rows)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    changes = [
        {
            "nucleotide_id": int(current_by_site[site]["id"]),
            "site": site,
            "old": str(current_by_site[site]["base_region"]),
            "new": provided[site],
        }
        for site in sorted(provided)
        if str(current_by_site[site]["base_region"]) != provided[site]
    ]
    audit_logged = True
    if changes:
        with active.conn:
            active.conn.executemany(
                "UPDATE meta_nucleotides SET base_region = ? "
                "WHERE construct_id = ? AND site = ?",
                [(change["new"], req.construct_id, change["site"]) for change in changes],
            )
        try:
            _write_maintenance_log({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "mode": "edit",
                "entity_type": "construct",
                "entity_id": req.construct_id,
                "disp_name": construct["disp_name"],
                "operation": "update_base_regions",
                "changes": changes,
            })
        except OSError:
            audit_logged = False

    saved = active.conn.execute(
        "SELECT site, base, base_region FROM meta_nucleotides "
        "WHERE construct_id = ? ORDER BY site",
        (req.construct_id,),
    ).fetchall()
    return {
        "ok": True,
        "changed": len(changes),
        "audit_logged": audit_logged,
        "nt_rows": [dict(row) for row in saved],
    }


# ---------------------------------------------------------------- generate

class GenerateRequest(BaseModel):
    mode: str = export_mod.HYBRID
    ignore_errors: bool = False


@app.post("/api/generate")
def generate(req: GenerateRequest) -> Dict[str, Any]:
    active = _require_session()
    report = validate(
        active.sheet, active.catalog, project_dir=str(active.project_dir),
        executors=active.project_config.executors if active.project_config else {},
    )
    if not report["ok"] and not req.ignore_errors:
        raise HTTPException(
            400,
            "%d problem(s) still need fixing before this sheet can be created."
            % report["error_count"],
        )
    result = export_mod.export(
        active.sheet, active.catalog, str(active.project_dir), active.label,
        mode=req.mode, db_path=str(active.db_path) if active.db_path else None,
        project_config=active.project_config,
    )
    return {"result": result, "validation": report, "state": _state()}


@app.post("/api/shutdown")
def shutdown_server(background_tasks: BackgroundTasks) -> Dict[str, bool]:
    """Return the response, then stop the local helper and restore the CLI prompt."""
    if _server is None:
        raise HTTPException(503, "The sample-input server cannot be stopped from this session.")
    background_tasks.add_task(_request_server_shutdown)
    return {"ok": True}


# ---------------------------------------------------------------- reaction groups

class GroupDerive(BaseModel):
    label_template: Optional[str] = None
    relabel_all: bool = False


@app.post("/api/groups/derive")
def derive_groups(req: GroupDerive) -> Dict[str, Any]:
    """(Re)build the group list from the sheet's buffer/replicate/temperature/
    construct columns, then write labels and any ladder-derived times."""
    active = _require_session()
    if req.label_template:
        active.groups.label_template = req.label_template
    if req.relabel_all:
        for group in active.groups.groups:
            group.label = ""
    result = active.groups.derive(active.sheet)
    applied = active.groups.apply(active.sheet)
    return {"result": {**result, **applied}, "state": _state()}


class GroupUpdate(BaseModel):
    key_id: str
    label: Optional[str] = None
    ladder_id: Optional[str] = None
    clear_ladder: bool = False


@app.post("/api/groups/update")
def update_group(req: GroupUpdate) -> Dict[str, Any]:
    active = _require_session()
    group = next((g for g in active.groups.groups if g.key_id == req.key_id), None)
    if group is None:
        raise HTTPException(404, "No group %r." % req.key_id)
    if req.label is not None:
        group.label = req.label.strip()
    if req.clear_ladder:
        group.ladder_id = None
    elif req.ladder_id is not None:
        if req.ladder_id not in active.groups.ladders:
            raise HTTPException(404, "No ladder %r." % req.ladder_id)
        group.ladder_id = req.ladder_id
    active.groups.apply(active.sheet)
    return {"state": _state()}


class GroupSplit(BaseModel):
    key_id: str
    uids: List[int]
    label: Optional[str] = None


@app.post("/api/groups/split")
def split_group(req: GroupSplit) -> Dict[str, Any]:
    """Move selected rows into their own group.

    Needed because the same buffer/replicate/temperature/construct can be run
    as more than one timecourse -- 9 such tuples exist in the demo sheet, one
    of them covering four separate groups.
    """
    active = _require_session()
    source = next((g for g in active.groups.groups if g.key_id == req.key_id), None)
    if source is None:
        raise HTTPException(404, "No group %r." % req.key_id)
    if not req.uids:
        raise HTTPException(400, "Select the rows to split out first.")

    suffix = 2
    existing = {g.key_id for g in active.groups.groups}
    while "|".join(source.key[:-1] + (source.key[-1] + " #%d" % suffix,)) in existing:
        suffix += 1
    new_key = source.key[:-1] + (source.key[-1] + " #%d" % suffix,)

    clone = ReactionGroup(key=new_key, label=(req.label or "").strip(),
                          ladder_id=source.ladder_id, pinned_uids=list(req.uids))
    active.groups.groups.append(clone)
    # Rows pinned elsewhere must not stay pinned to the source.
    source.pinned_uids = [u for u in source.pinned_uids if u not in set(req.uids)]
    active.groups._autolabel()
    active.groups.apply(active.sheet)
    return {"result": {"new_key_id": clone.key_id, "moved": len(req.uids)}, "state": _state()}


@app.post("/api/groups/merge")
def merge_group(req: GroupSplit) -> Dict[str, Any]:
    """Undo a split: drop a pinned group so its rows fall back to their key."""
    active = _require_session()
    before = len(active.groups.groups)
    active.groups.groups = [
        g for g in active.groups.groups
        if not (g.key_id == req.key_id and g.pinned_uids)
    ]
    if len(active.groups.groups) == before:
        raise HTTPException(404, "No split group %r to merge back." % req.key_id)
    active.groups.derive(active.sheet)
    active.groups.apply(active.sheet)
    return {"state": _state()}


class LadderUpsert(BaseModel):
    ladder_id: Optional[str] = None
    name: str = ""
    points: Optional[List[float]] = None
    text: Optional[str] = None       # pasted: commas/tabs/newlines, unit suffixes ok
    assign_to: Optional[List[str]] = None


@app.post("/api/ladders")
def upsert_ladder(req: LadderUpsert) -> Dict[str, Any]:
    active = _require_session()
    points = list(req.points) if req.points else parse_ladder_text(req.text or "")
    if not points:
        raise HTTPException(
            400,
            "No times found. Enter them separated by commas, tabs or newlines "
            "-- bare numbers are seconds, or add a unit like 5min or 1h.",
        )

    if req.ladder_id:
        ladder = active.groups.ladders.get(req.ladder_id)
        if ladder is None:
            raise HTTPException(404, "No ladder %r." % req.ladder_id)
        ladder.points = points
        if req.name:
            ladder.name = req.name
    else:
        ladder = active.groups.new_ladder(req.name or "ladder", points)

    for key_id in req.assign_to or []:
        group = next((g for g in active.groups.groups if g.key_id == key_id), None)
        if group is not None:
            group.ladder_id = ladder.id

    applied = active.groups.apply(active.sheet)
    return {"result": {"ladder": ladder.to_dict(), **applied}, "state": _state()}


@app.delete("/api/ladders/{ladder_id}")
def delete_ladder(ladder_id: str) -> Dict[str, Any]:
    active = _require_session()
    if not active.groups.delete_ladder(ladder_id):
        raise HTTPException(404, "No ladder %r." % ladder_id)
    return {"state": _state()}


class LadderGrid(BaseModel):
    """Bulk entry: one pasted row per group, tab/comma separated times."""
    text: str
    key_ids: List[str]


@app.post("/api/ladders/grid")
def ladder_grid(req: LadderGrid) -> Dict[str, Any]:
    active = _require_session()
    lines = [l for l in (req.text or "").splitlines() if l.strip()]
    if not lines:
        raise HTTPException(400, "Nothing to paste.")
    if len(lines) != len(req.key_ids):
        raise HTTPException(
            400,
            "Got %d line(s) for %d group(s). Paste one line of times per group, "
            "in the order shown." % (len(lines), len(req.key_ids)),
        )

    created = reused = 0
    for key_id, line in zip(req.key_ids, lines):
        points = parse_ladder_text(line)
        if not points:
            continue
        group = next((g for g in active.groups.groups if g.key_id == key_id), None)
        if group is None:
            continue
        existing = active.groups.ladder_matching(points)
        if existing is not None:
            group.ladder_id = existing.id
            reused += 1
        else:
            ladder = active.groups.new_ladder(group.label or key_id, points)
            group.ladder_id = ladder.id
            created += 1

    applied = active.groups.apply(active.sheet)
    return {"result": {"ladders_created": created, "ladders_reused": reused, **applied},
            "state": _state()}
