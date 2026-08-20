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

from nerd.project import ProjectConfig, ProjectContext, load_project, project_file_for

from .catalog import EntityCatalog
from .groups import GroupSet
from .model import Sheet
from .pattern import TokenRegistry

DRAFT_FILENAME = "sample-draft.json"
LEGACY_DRAFT_FILENAME = ".nerd_sample_draft.json"
DRAFT_VERSION = 3


class Session:
    def __init__(self) -> None:
        self.project_dir: Optional[Path] = None
        self.project_config: Optional[ProjectConfig] = None
        self.db_path: Optional[Path] = None
        self.output_dir: Optional[Path] = None
        self.conn = None
        self.label: str = "sample_import"
        self.sheet: Sheet = Sheet()
        self.catalog: EntityCatalog = EntityCatalog()
        self.groups: GroupSet = GroupSet()
        self.registry: TokenRegistry = TokenRegistry.load()
        self.pattern: str = ""
        self.fq_dir: str = ""
        self.fq_source: str = "local"

    # ---- lifecycle ----

    @property
    def connected(self) -> bool:
        return self.project_dir is not None

    def connect(self, project_dir: str, db_path: Optional[str], label: str) -> Dict[str, Any]:
        from nerd.db import api as db_api

        requested = Path(project_dir).expanduser()
        marker = project_file_for(requested, require=False)
        project_config = load_project(marker) if marker.is_file() else None
        if project_config is not None:
            project = project_config.root
        else:
            if requested.name in {"project.toml", ".nerd"}:
                raise ValueError("No Phase 4 project file found at %s." % marker)
            project = requested.resolve()
            project.mkdir(parents=True, exist_ok=True)

        # UI-entered relative DB paths are project-root-relative. The CLI
        # resolves its own --db value before it reaches this method.
        explicit_db = None
        if db_path:
            explicit_db = Path(db_path).expanduser()
            if not explicit_db.is_absolute():
                explicit_db = project / explicit_db
        context = ProjectContext(db=explicit_db, project=project)
        database = context.resolve_database(cwd=project)
        is_new = not database.is_file()

        conn = db_api.connect(database)
        db_api.init_schema(conn)

        previous_conn = self.conn

        self.project_dir = project
        self.project_config = project_config
        self.db_path = database
        self.output_dir = project_config.output_dir if project_config else project
        self.conn = conn
        self.label = label or "sample_import"
        self.catalog = EntityCatalog.from_connection(conn)
        self.sheet = Sheet()
        self.groups = GroupSet()
        self.pattern = ""
        self.fq_dir = ""
        self.fq_source = "local"

        if previous_conn is not None:
            previous_conn.close()

        restored = self.load_draft()

        return {
            "project_dir": str(project),
            "project_id": project_config.name if project_config else None,
            "project_file": str(project_config.source_path) if project_config else None,
            "db_path": str(database),
            "output_dir": str(self.output_dir),
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
        if self.project_dir is None:
            return None
        if self.project_config is not None:
            return self.project_dir / ".nerd" / DRAFT_FILENAME
        return self.project_dir / LEGACY_DRAFT_FILENAME

    def _draft_candidates(self) -> List[Path]:
        primary = self.draft_path
        if primary is None:
            return []
        candidates = [primary]
        legacy = self.project_dir / LEGACY_DRAFT_FILENAME
        if legacy not in candidates:
            candidates.append(legacy)
        return candidates

    def save_draft(self) -> None:
        path = self.draft_path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": DRAFT_VERSION,
            "label": self.label,
            "pattern": self.pattern,
            "fq_dir": self.fq_dir,
            "fq_source": self.fq_source,
            "sheet": self.sheet.to_dict(),
            "catalog": self.catalog.to_dict(),
            "groups": self.groups.to_dict(),
        }
        path.write_text(json.dumps(payload, indent=2, default=str))

    def load_draft(self) -> bool:
        path = next((item for item in self._draft_candidates() if item.is_file()), None)
        if path is None:
            return False
        try:
            payload = json.loads(path.read_text())
        except (ValueError, OSError):
            return False
        if int(payload.get("version", 1)) != DRAFT_VERSION:
            return False
        self.pattern = payload.get("pattern", "") or ""
        self.fq_dir = payload.get("fq_dir", "") or ""
        self.fq_source = payload.get("fq_source", "local") or "local"
        self.label = payload.get("label") or self.label
        self.sheet = Sheet.from_dict(payload.get("sheet") or {})
        self.catalog.load_staged(payload.get("catalog") or {})
        self.groups = GroupSet.from_dict(payload.get("groups") or {})
        return bool(self.sheet.rows or any(self.catalog.staged.values()))
