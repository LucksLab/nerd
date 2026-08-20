"""Session state: everything the user has built, and its persistence.

The draft file holds the whole session -- rows *and* staged entities *and*
the pattern -- so a server restart or browser refresh loses nothing. (An
earlier design persisted only rows, which meant half your work survived a
restart and half silently vanished.)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .catalog import EntityCatalog
from .groups import GroupSet
from .model import Sheet
from .pattern import TokenRegistry

DRAFT_FILENAME = ".nerd_sample_draft.json"
DRAFT_VERSION = 3


class Session:
    def __init__(self) -> None:
        self.project_dir: Optional[Path] = None
        self.db_path: Optional[Path] = None
        self.conn = None
        self.label: str = "sample_import"
        self.sheet: Sheet = Sheet()
        self.catalog: EntityCatalog = EntityCatalog()
        self.groups: GroupSet = GroupSet()
        self.registry: TokenRegistry = TokenRegistry.load()
        self.pattern: str = ""
        self.fq_dir: str = ""

    # ---- lifecycle ----

    @property
    def connected(self) -> bool:
        return self.project_dir is not None

    def connect(self, project_dir: str, db_path: Optional[str], label: str) -> Dict[str, Any]:
        from nerd.db import api as db_api

        project = Path(project_dir).expanduser().resolve()
        project.mkdir(parents=True, exist_ok=True)
        database = (
            Path(db_path).expanduser().resolve() if db_path else project / "nerd.sqlite"
        )
        is_new = not database.is_file()

        conn = db_api.connect(database)
        db_api.init_schema(conn)

        self.project_dir = project
        self.db_path = database
        self.conn = conn
        self.label = label or "sample_import"
        self.catalog = EntityCatalog.from_connection(conn)
        self.sheet = Sheet()
        self.groups = GroupSet()
        self.pattern = ""
        self.fq_dir = ""

        restored = self.load_draft()

        return {
            "project_dir": str(project),
            "db_path": str(database),
            "is_new_db": is_new,
            "restored_draft": restored,
            "db_entity_counts": {
                key: len(value) for key, value in self.catalog.db.items()
            },
        }

    def refresh_catalog(self) -> None:
        """Re-read DB entities, keeping staged ones."""
        staged = self.catalog.staged
        self.catalog = EntityCatalog.from_connection(self.conn)
        self.catalog.staged = staged

    # ---- persistence ----

    @property
    def draft_path(self) -> Optional[Path]:
        return self.project_dir / DRAFT_FILENAME if self.project_dir else None

    def save_draft(self) -> None:
        path = self.draft_path
        if path is None:
            return
        payload = {
            "version": DRAFT_VERSION,
            "label": self.label,
            "pattern": self.pattern,
            "fq_dir": self.fq_dir,
            "sheet": self.sheet.to_dict(),
            "catalog": self.catalog.to_dict(),
            "groups": self.groups.to_dict(),
        }
        path.write_text(json.dumps(payload, indent=2, default=str))

    def load_draft(self) -> bool:
        path = self.draft_path
        if path is None or not path.is_file():
            return False
        try:
            payload = json.loads(path.read_text())
        except (ValueError, OSError):
            return False
        if int(payload.get("version", 1)) != DRAFT_VERSION:
            return False
        self.pattern = payload.get("pattern", "") or ""
        self.fq_dir = payload.get("fq_dir", "") or ""
        self.label = payload.get("label") or self.label
        self.sheet = Sheet.from_dict(payload.get("sheet") or {})
        self.catalog.load_staged(payload.get("catalog") or {})
        self.groups = GroupSet.from_dict(payload.get("groups") or {})
        return bool(self.sheet.rows or any(self.catalog.staged.values()))
