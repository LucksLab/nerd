"""One in-memory view of every entity a sheet can reference.

The sheet references constructs, buffers and sequencing runs by name. Those
names have to resolve either to a row already in the nerd DB or to
something the user staged in this session. Resolving that per cell over
HTTP does not scale -- a real sheet is >1000 rows across three entity
columns -- so the catalog loads the DB once, merges staged entities into
the same lookup, and answers in memory.

Identity follows nerd's own uniqueness rules: a construct is identified by
(family, name, version, sequence, disp_name) and a buffer by
(name, pH, composition, disp_name), so two entries can share a disp_name
while being genuinely different rows. Lookups are by display name (that is
what the sheet stores) but equality is by the full tuple.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

CONSTRUCT_KEY = ("family", "name", "version", "sequence", "disp_name")
BUFFER_KEY = ("name", "pH", "composition", "disp_name")
SEQRUN_KEY = ("run_name", "date", "sequencer", "run_manager")

ENTITY_KEYS: Dict[str, Tuple[str, ...]] = {
    "construct": CONSTRUCT_KEY,
    "buffer": BUFFER_KEY,
    "sequencing_run": SEQRUN_KEY,
}

# Fields a wizard must collect for each entity type, in display order.
ENTITY_FIELDS: Dict[str, List[Dict[str, Any]]] = {
    "construct": [
        {"name": "disp_name", "label": "disp_name", "required": True,
         "help": "How samples refer to this construct."},
        {"name": "family", "label": "family", "required": True},
        {"name": "name", "label": "name", "required": True},
        {"name": "version", "label": "version", "required": True},
        {"name": "sequence", "label": "sequence", "required": True, "type": "textarea"},
    ],
    "buffer": [
        {"name": "disp_name", "label": "disp_name", "required": True},
        {"name": "name", "label": "name", "required": True},
        {"name": "pH", "label": "pH", "required": True, "type": "number"},
        {"name": "composition", "label": "composition", "required": True},
    ],
    "sequencing_run": [
        {"name": "run_name", "label": "run_name", "required": True},
        {"name": "date", "label": "date (YYMMDD)", "required": True},
        {"name": "sequencer", "label": "sequencer", "required": True},
        {"name": "run_manager", "label": "run_manager", "required": True},
    ],
}

# Which field of each entity the sheet column holds.
LOOKUP_FIELDS: Dict[str, Tuple[str, ...]] = {
    "construct": ("disp_name",),
    "buffer": ("disp_name", "name"),
    "sequencing_run": ("run_name",),
}


def identity(record: Dict[str, Any], entity_type: str) -> Tuple[str, ...]:
    return tuple(str(record.get(f, "")).strip() for f in ENTITY_KEYS[entity_type])


@dataclass
class Resolution:
    status: str            # "db" | "staged" | "missing"
    record: Optional[Dict[str, Any]] = None
    entity_id: Optional[int] = None


@dataclass
class EntityCatalog:
    """DB-backed entities plus this session's staged ones."""

    db: Dict[str, List[Dict[str, Any]]] = field(
        default_factory=lambda: {"construct": [], "buffer": [], "sequencing_run": []}
    )
    staged: Dict[str, List[Dict[str, Any]]] = field(
        default_factory=lambda: {"construct": [], "buffer": [], "sequencing_run": []}
    )

    # ---- loading ----

    @classmethod
    def from_connection(cls, conn) -> "EntityCatalog":
        catalog = cls()
        if conn is None:
            return catalog
        catalog.db["construct"] = _fetch(
            conn, "SELECT id, family, name, version, sequence, disp_name FROM meta_constructs"
        )
        catalog.db["buffer"] = _fetch(
            conn, "SELECT id, name, pH, composition, disp_name FROM meta_buffers"
        )
        catalog.db["sequencing_run"] = _fetch(
            conn, "SELECT id, run_name, date, sequencer, run_manager FROM sequencing_runs"
        )
        return catalog

    # ---- lookup ----

    def _index(self, entity_type: str) -> Dict[str, Resolution]:
        index: Dict[str, Resolution] = {}
        for record in self.db.get(entity_type, []):
            for lookup_field in LOOKUP_FIELDS[entity_type]:
                key = str(record.get(lookup_field, "")).strip().lower()
                if key:
                    index.setdefault(key, Resolution("db", record, record.get("id")))
        # Staged entries shadow nothing: a DB hit wins, since that is what
        # create.py will resolve to.
        for record in self.staged.get(entity_type, []):
            for lookup_field in LOOKUP_FIELDS[entity_type]:
                key = str(record.get(lookup_field, "")).strip().lower()
                if key:
                    index.setdefault(key, Resolution("staged", record))
        return index

    def resolve(self, entity_type: str, value: str) -> Resolution:
        key = str(value or "").strip().lower()
        if not key:
            return Resolution("missing")
        return self._index(entity_type).get(key, Resolution("missing"))

    def resolve_many(self, entity_type: str, values: Iterable[str]) -> Dict[str, str]:
        """Bulk status lookup: {value: status}. One index build, no IO."""
        index = self._index(entity_type)
        out: Dict[str, str] = {}
        for value in values:
            key = str(value or "").strip().lower()
            out[value] = index[key].status if key in index else "missing"
        return out

    # ---- staging ----

    def collision(self, entity_type: str, candidate: Dict[str, Any]) -> Optional[str]:
        """Return a message if the display name is taken by a different entity.

        Re-submitting an identical record is fine (it matches nerd's own
        upsert semantics); a same-name-different-identity record is not,
        because nerd's own lookup would resolve it ambiguously.
        """
        lookup_field = LOOKUP_FIELDS[entity_type][0]
        existing = self.resolve(entity_type, candidate.get(lookup_field, ""))
        if existing.status == "missing" or existing.record is None:
            return None
        if identity(existing.record, entity_type) == identity(candidate, entity_type):
            return None
        where = "the target database" if existing.status == "db" else "this session"
        differing = [
            f for f in ENTITY_KEYS[entity_type]
            if str(existing.record.get(f, "")).strip() != str(candidate.get(f, "")).strip()
        ]
        return (
            "A %s called '%s' already exists in %s but differs in: %s. "
            "Use a different %s, or reuse the existing entry."
            % (entity_type.replace("_", " "), candidate.get(lookup_field, ""),
               where, ", ".join(differing), lookup_field)
        )

    def stage(self, entity_type: str, record: Dict[str, Any]) -> None:
        target = self.staged.setdefault(entity_type, [])
        key = identity(record, entity_type)
        for index, existing in enumerate(target):
            if identity(existing, entity_type) == key:
                target[index] = record  # idempotent restage
                return
        target.append(record)

    def unstage(self, entity_type: str, lookup_value: str) -> bool:
        lookup_field = LOOKUP_FIELDS[entity_type][0]
        target = self.staged.setdefault(entity_type, [])
        needle = str(lookup_value).strip().lower()
        for index, record in enumerate(target):
            if str(record.get(lookup_field, "")).strip().lower() == needle:
                target.pop(index)
                return True
        return False

    # ---- persistence ----

    def to_dict(self) -> Dict[str, Any]:
        return {"staged": self.staged}

    def load_staged(self, data: Dict[str, Any]) -> None:
        staged = (data or {}).get("staged") or {}
        for entity_type in ("construct", "buffer", "sequencing_run"):
            self.staged[entity_type] = list(staged.get(entity_type) or [])


def _fetch(conn, sql: str) -> List[Dict[str, Any]]:
    try:
        return [dict(row) for row in conn.execute(sql).fetchall()]
    except Exception:  # table may not exist on a brand-new DB
        return []
