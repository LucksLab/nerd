"""
Task for marking sequencing samples as dropped based on a configuration file.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from nerd.utils.logging import get_logger

from .base import Task, TaskContext, TaskScope

log = get_logger(__name__)


class DropTask(Task):
    """
    Marks sequencing samples with the `to_drop` flag.
    """

    name = "drop"
    scope_kind = "global"

    _DEFAULT_FLAG = 1
    _MISSING_POLICIES = {"warn", "error", "ignore"}

    def prepare(self, cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if self.name not in cfg:
            raise ValueError("Configuration must contain a 'drop' section.")

        block = dict(cfg[self.name])
        drop_flag = self._normalize_flag(block.get("to_drop", self._DEFAULT_FLAG))
        missing_policy = str(block.get("on_missing", "warn")).strip().lower()
        if missing_policy not in self._MISSING_POLICIES:
            raise ValueError("drop.on_missing must be one of: warn, error, ignore.")

        sample_names: List[str] = []
        inline_sources = [
            block.get("sample_name"),
            block.get("sample_names"),
            block.get("samples"),
        ]
        for src in inline_sources:
            sample_names.extend(self._normalize_samples(src))

        yaml_sources = (
            block.get("samples_yaml")
            or block.get("samples_config")
            or block.get("config")
            or block.get("config_path")
            or block.get("yaml")
        )
        if yaml_sources:
            if isinstance(yaml_sources, (str, Path)):
                sources_iter = [yaml_sources]
            elif isinstance(yaml_sources, (list, tuple, set)):
                sources_iter = list(yaml_sources)
            else:
                raise TypeError("drop.samples_yaml must be a path or list of paths.")

            for raw_path in sources_iter:
                path = Path(raw_path).expanduser()
                if not path.is_absolute():
                    path = Path.cwd() / path
                if not path.is_file():
                    raise FileNotFoundError(f"Samples config file not found: {path}")
                loaded = self._load_samples_from_yaml(path)
                sample_names.extend(loaded)

        unique_samples = self._unique_preserve_order(sample_names)
        if not unique_samples:
            raise ValueError("drop task did not resolve any sample names to update.")

        inputs = {
            "sample_names": unique_samples,
            "to_drop": drop_flag,
            "missing_policy": missing_policy,
        }
        return inputs, {}

    def command(self, ctx: TaskContext, inputs: Dict[str, Any], params: Dict[str, Any]) -> Optional[str]:
        return None

    def consume_outputs(
        self,
        ctx: TaskContext,
        inputs: Dict[str, Any],
        params: Dict[str, Any],
        run_dir: Path,
        task_id: Optional[int] = None,
    ):
        sample_names = list(inputs.get("sample_names") or [])
        drop_flag = int(inputs.get("to_drop", self._DEFAULT_FLAG))
        missing_policy = str(inputs.get("missing_policy", "warn"))

        if not sample_names:
            log.warning("No sample names provided to drop task; nothing to update.")
            return

        provided_set = set(sample_names)
        found: List[str] = []
        marked: List[str] = []
        missing: List[str] = []
        changed_count = 0
        try:
            with ctx.db:  # type: ignore[arg-type]
                for name in sample_names:
                    row = ctx.db.execute(
                        "SELECT to_drop FROM sequencing_samples WHERE sample_name = ?",
                        (name,),
                    ).fetchone()
                    if row is None:
                        missing.append(name)
                        continue

                    found.append(name)
                    current_flag = int(row["to_drop"])
                    if current_flag != drop_flag:
                        ctx.db.execute(
                            "UPDATE sequencing_samples SET to_drop = ? WHERE sample_name = ?",
                            (drop_flag, name),
                        )
                        changed_count += 1
                    if drop_flag == 1:
                        marked.append(name)
        except sqlite3.Error as exc:
            log.exception("Failed to update sequencing_samples.to_drop: %s", exc)
            raise

        undropped_changed: List[str] = []
        try:
            with ctx.db:  # type: ignore[arg-type]
                rows = ctx.db.execute(
                    "SELECT sample_name, to_drop FROM sequencing_samples"
                ).fetchall()
                for row in rows:
                    name = row["sample_name"]
                    if name in provided_set:
                        continue
                    if int(row["to_drop"]) != 0:
                        ctx.db.execute(
                            "UPDATE sequencing_samples SET to_drop = 0 WHERE sample_name = ?",
                            (name,),
                        )
                        undropped_changed.append(name)
        except sqlite3.Error as exc:
            log.exception("Failed to reset to_drop for unlisted samples: %s", exc)
            raise

        log.info(
            "drop task updated to_drop=%d for %d sequencing_samples (names provided: %d).",
            drop_flag,
            len(found),
            len(sample_names),
        )

        if missing:
            preview = ", ".join(missing[:10])
            msg = (
                f"{len(missing)} sample(s) not found in sequencing_samples. "
                f"First entries: {preview}"
            )
            if missing_policy == "error":
                raise RuntimeError(msg)
            if missing_policy == "warn":
                log.warning("%s", msg)
            else:
                log.info("%s (ignored)", msg)

        summary_lines = [
            f"Total sample names provided: {len(sample_names)}",
            f"Samples found in sequencing_samples: {len(found)}",
            f"Samples changed to to_drop={drop_flag}: {changed_count}",
            f"Samples not found: {len(missing)}",
            f"Samples changed to to_drop=0 (not listed): {len(undropped_changed)}",
        ]
        if drop_flag == 1:
            summary_lines.append(f"Samples flagged to_drop=1: {len(marked)}")

        summary_path = run_dir / "drop_summary.txt"
        summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
        log.info("Wrote drop summary to %s", summary_path)

        if drop_flag == 1:
            marked_path = run_dir / "samples_marked_to_drop.txt"
            marked_path.write_text("\n".join(marked) + ("\n" if marked else ""), encoding="utf-8")
            log.info("Listed %d samples marked to_drop=1 in %s", len(marked), marked_path)

        undropped_path = run_dir / "samples_marked_to_keep.txt"
        undropped_path.write_text("\n".join(undropped_changed) + ("\n" if undropped_changed else ""), encoding="utf-8")
        log.info("Listed %d samples reset to to_drop=0 in %s", len(undropped_changed), undropped_path)

        missing_path = run_dir / "samples_not_found.txt"
        missing_path.write_text("\n".join(missing) + ("\n" if missing else ""), encoding="utf-8")
        log.info("Listed %d samples not found in %s", len(missing), missing_path)

    def resolve_scope(self, ctx: Optional[TaskContext], inputs: Any) -> TaskScope:
        if isinstance(inputs, dict):
            sample_names = inputs.get("sample_names") or []
            label = f"{len(sample_names)} samples" if sample_names else None
            return TaskScope(kind=self.scope_kind, label=label)
        return TaskScope(kind=self.scope_kind)

    @staticmethod
    def _normalize_flag(raw: Any) -> int:
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = 1 if raw else 0
        return 1 if value else 0

    def _normalize_samples(self, raw: Any) -> List[str]:
        if raw in (None, "", False):
            return []
        if isinstance(raw, str):
            value = raw.strip()
            return [value] if value else []
        if isinstance(raw, (list, tuple, set)):
            collected: List[str] = []
            for item in raw:
                collected.extend(self._normalize_samples(item))
            return collected
        if isinstance(raw, dict):
            collected: List[str] = []
            keys = ("sample_name", "sample_names", "samples")
            matched = False
            for key in keys:
                if key in raw:
                    matched = True
                    collected.extend(self._normalize_samples(raw[key]))
            if not matched:
                for value in raw.values():
                    collected.extend(self._normalize_samples(value))
            return collected
        raise TypeError(f"Unsupported sample specification type: {type(raw).__name__}")

    def _load_samples_from_yaml(self, path: Path) -> List[str]:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        samples = self._normalize_samples(data)
        if not samples and isinstance(data, dict):
            # Common pattern: {"sample_name": [...]}
            samples = self._normalize_samples(data.get("sample_name"))
        log.info("Loaded %d sample(s) from %s", len(samples), path)
        return samples

    @staticmethod
    def _unique_preserve_order(values: Sequence[str]) -> List[str]:
        seen = set()
        result: List[str] = []
        for value in values:
            if value not in seen:
                seen.add(value)
                result.append(value)
        return result
