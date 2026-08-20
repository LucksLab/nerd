"""The sample sheet: rows of nerd `create` columns, with cell provenance.

Provenance is what makes the sheet re-fillable. Every cell records which
filler wrote it, and a filler may only overwrite a cell whose origin has
precedence <= its own. A cell the user typed by hand (MANUAL) therefore
survives any number of re-parses, which is what lets someone iterate on a
naming pattern without losing the columns they filled in by hand.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

# Column order matches nerd's probing_samples.csv convention. `RT` is the
# header nerd normalizes to rt_protocol; reaction_group is a free-text
# label nerd maps to a numeric rg_id at ingest.
SAMPLE_COLUMNS: List[str] = [
    "sequencing_run_name", "sample_name", "fq_source", "fq_dir", "r1_file", "r2_file",
    "reaction_group", "temperature", "replicate", "reaction_time", "probe",
    "probe_concentration", "RT", "treated", "buffer", "construct", "done_by",
]

# Columns create.py refuses to ingest when empty. Note sequencing_run_name
# is NOT among them -- it is empty in all 1131 rows of the demo sheet.
REQUIRED_COLUMNS: List[str] = [
    "sample_name", "fq_source", "fq_dir", "r1_file", "r2_file", "reaction_group",
    "temperature", "replicate", "reaction_time", "probe",
    "probe_concentration", "RT", "treated", "buffer", "construct", "done_by",
]

# Columns whose value must resolve to a row in the nerd DB (or to something
# staged in this session) before create.py will accept the sheet.
ENTITY_COLUMNS: Dict[str, str] = {
    "construct": "construct",
    "buffer": "buffer",
    "sequencing_run_name": "sequencing_run",
}

EMPTY = ""
FASTQ = "fastq"
PATTERN = "pattern"
GROUP = "group"     # derived from a reaction group (label, ladder time)
BATCH = "batch"
MANUAL = "manual"

PRECEDENCE: Dict[str, int] = {
    EMPTY: 0, FASTQ: 10, PATTERN: 20, GROUP: 25, BATCH: 30, MANUAL: 40,
}


@dataclass
class Row:
    uid: int
    values: Dict[str, Any] = field(default_factory=dict)
    origins: Dict[str, str] = field(default_factory=dict)
    # Tokens the pattern parsed that map to no sheet column -- [id],
    # [construct_family], and crucially [tp_num]. Not written to the sheet,
    # but needed downstream: the timepoint index is what a reaction group's
    # time ladder is looked up by.
    context: Dict[str, Any] = field(default_factory=dict)
    # Set by pattern_fill when the pattern failed to match this row's name.
    unmatched: bool = False

    def get(self, column: str) -> Any:
        return self.values.get(column, "")

    def origin(self, column: str) -> str:
        return self.origins.get(column, EMPTY)

    def can_write(self, column: str, origin: str) -> bool:
        return PRECEDENCE.get(self.origin(column), 0) <= PRECEDENCE.get(origin, 0)

    def set(self, column: str, value: Any, origin: str, force: bool = False) -> bool:
        """Write a cell if provenance allows. Returns True if written."""
        if not force and not self.can_write(column, origin):
            return False
        self.values[column] = value
        self.origins[column] = origin
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uid": self.uid,
            "values": {c: self.values.get(c, "") for c in SAMPLE_COLUMNS},
            "origins": {c: self.origins.get(c, EMPTY) for c in SAMPLE_COLUMNS},
            "context": dict(self.context),
            "unmatched": self.unmatched,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Row":
        values = dict(data.get("values") or {})
        origins = dict(data.get("origins") or {})
        if not values.get("fq_source"):
            values["fq_source"] = "local"
            origins["fq_source"] = FASTQ
        return cls(
            uid=int(data["uid"]),
            values=values,
            origins=origins,
            context=dict(data.get("context") or {}),
            unmatched=bool(data.get("unmatched", False)),
        )


class Sheet:
    """An ordered collection of rows plus uid assignment."""

    def __init__(self, rows: Optional[List[Row]] = None, next_uid: int = 1):
        self.rows: List[Row] = rows or []
        self._next_uid = max([r.uid for r in self.rows], default=0) + 1
        if next_uid > self._next_uid:
            self._next_uid = next_uid

    def new_uid(self) -> int:
        uid = self._next_uid
        self._next_uid += 1
        return uid

    def add_row(self, values: Optional[Dict[str, Any]] = None, origin: str = MANUAL) -> Row:
        row = Row(uid=self.new_uid())
        for column in SAMPLE_COLUMNS:
            row.values[column] = ""
            row.origins[column] = EMPTY
        for column, value in (values or {}).items():
            if column in SAMPLE_COLUMNS:
                row.set(column, value, origin, force=True)
        self.rows.append(row)
        return row

    def by_uid(self, uid: int) -> Optional[Row]:
        return next((r for r in self.rows if r.uid == uid), None)

    def delete(self, uids: Iterable[int]) -> int:
        targets = set(uids)
        before = len(self.rows)
        self.rows = [r for r in self.rows if r.uid not in targets]
        return before - len(self.rows)

    def distinct(self, column: str) -> List[str]:
        seen, out = set(), []
        for row in self.rows:
            raw = row.get(column)
            value = "" if raw is None else str(raw).strip()
            if value and value not in seen:
                seen.add(value)
                out.append(value)
        return out

    def clear(self) -> None:
        self.rows = []

    def to_dict(self) -> Dict[str, Any]:
        return {"rows": [r.to_dict() for r in self.rows], "next_uid": self._next_uid}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Sheet":
        rows = [Row.from_dict(r) for r in (data.get("rows") or [])]
        return cls(rows=rows, next_uid=int(data.get("next_uid", 1)))

    def __len__(self) -> int:
        return len(self.rows)
