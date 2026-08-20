"""FastAPI layer for the nerd sample-input helper.

Deliberately thin: every endpoint translates HTTP to a call into
nerd.sheetbuilder and back. All the logic -- provenance, pattern
compilation, entity resolution, validation, export -- lives in that
package, which has no web dependency and is unit-testable on its own.

Every mutating endpoint returns the *whole* new state (rows, entity
resolution, validation summary) so the frontend never has to stitch
together partial updates or re-query to find out what changed.

Launch inside a Phase 4 project with ``nerd webui serve``, or select one
explicitly with ``nerd webui serve --project <project_root> [--db <path>]``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
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

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="nerd sample input helper")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

session = Session()
_server: Optional[Any] = None


# ---------------------------------------------------------------- helpers

def _require_session() -> Session:
    if not session.connected:
        raise HTTPException(400, "Connect to a project folder first.")
    return session


def set_server(server: Optional[Any]) -> None:
    """Register the running uvicorn server so the local UI can stop it."""
    global _server
    _server = server


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
        info = session.connect(req.project_dir, req.db_path, req.label)
    except Exception as exc:
        raise HTTPException(400, "Could not open project: %s" % exc)
    return {**info, "state": _state(save=False)}


@app.get("/api/state")
def get_state() -> Dict[str, Any]:
    return _state(save=False)


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
        record["nt_rows"] = req.nt_rows or export_mod.default_nt_rows(record.get("sequence", ""))
        try:
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
def nt_rows(disp_name: Optional[str] = None, sequence: Optional[str] = None) -> Dict[str, Any]:
    """Seed the numbering grid: default 1-based, or copy an existing construct's."""
    if sequence:
        return {"nt_rows": export_mod.default_nt_rows(sequence), "source": "default"}
    if not disp_name:
        raise HTTPException(400, "Pass either sequence= or disp_name=.")

    staged = next(
        (c for c in session.catalog.staged.get("construct", [])
         if str(c.get("disp_name", "")).lower() == disp_name.lower()), None
    )
    if staged and staged.get("nt_rows"):
        return {"nt_rows": staged["nt_rows"], "source": "staged"}

    resolved = session.catalog.resolve("construct", disp_name)
    if resolved.status == "db" and session.conn is not None:
        rows = session.conn.execute(
            "SELECT site, base, base_region FROM meta_nucleotides "
            "WHERE construct_id = ? ORDER BY site",
            (resolved.entity_id,),
        ).fetchall()
        if rows:
            return {"nt_rows": [dict(r) for r in rows], "source": "database"}
    raise HTTPException(404, "No numbering found for %r." % disp_name)


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
