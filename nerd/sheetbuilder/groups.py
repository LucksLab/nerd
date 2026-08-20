"""Reaction groups and their timepoint ladders.

A reaction group is one timecourse: a set of samples sharing buffer,
replicate, temperature and construct, sampled at a series of timepoints.
Sample names carry the timepoint *index* (tp1, tp2, ...), not the elapsed
time -- tp2 means "the second timepoint", not "2 seconds". The seconds come
from the group's **time ladder**, entered once.

Two facts from the real 1131-row sheet shape this design:

  * (buffer, replicate, temperature, construct) yields 123 distinct tuples
    against 134 hand-assigned group labels. So the tuple is a good default
    grouping key, but 9 tuples were run as more than one timecourse and
    have to be splittable by hand.
  * There are only 37 distinct ladders across those 134 groups, and the
    most-used ladder is shared by 13 of them. So the ladder is a named,
    reusable object that groups point at -- not a table retyped per group.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .model import GROUP, Row, Sheet

# The columns whose combination identifies one timecourse.
GROUP_KEY_COLUMNS: Tuple[str, ...] = ("buffer", "replicate", "temperature", "construct")

# Token in a row's context holding the timepoint index.
TP_CONTEXT_KEYS = ("tp_num", "timepoint", "tp")

DEFAULT_LABEL_TEMPLATE = "{temperature}_{n}"


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def group_key(row: Row) -> Tuple[str, ...]:
    return tuple(_text(row.get(column)) for column in GROUP_KEY_COLUMNS)


def timepoint_index(row: Row) -> Optional[int]:
    """The timepoint index for this row, from pattern context or the sheet."""
    for key in TP_CONTEXT_KEYS:
        if key in row.context:
            try:
                return int(float(str(row.context[key])))
            except (TypeError, ValueError):
                continue
    return None


@dataclass
class Ladder:
    """A named list of reaction times, in seconds, indexed by timepoint."""

    id: str
    name: str
    points: List[float] = field(default_factory=list)

    def seconds_for(self, index: int) -> Optional[float]:
        if index is None or index < 1 or index > len(self.points):
            return None
        return self.points[index - 1]

    @property
    def monotonic(self) -> bool:
        return all(b > a for a, b in zip(self.points, self.points[1:]))

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name, "points": list(self.points),
                "monotonic": self.monotonic}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Ladder":
        return cls(id=str(data["id"]), name=str(data.get("name") or data["id"]),
                   points=[float(p) for p in (data.get("points") or [])])


@dataclass
class ReactionGroup:
    key: Tuple[str, ...]
    label: str = ""
    ladder_id: Optional[str] = None
    # Rows explicitly pinned to this group, overriding key-based membership.
    # This is how a tuple that was run as two separate timecourses gets split.
    pinned_uids: List[int] = field(default_factory=list)

    @property
    def key_id(self) -> str:
        return "|".join(self.key)

    def describe(self) -> Dict[str, str]:
        return dict(zip(GROUP_KEY_COLUMNS, self.key))

    def to_dict(self) -> Dict[str, Any]:
        return {"key": list(self.key), "key_id": self.key_id, "label": self.label,
                "ladder_id": self.ladder_id, "pinned_uids": list(self.pinned_uids),
                "fields": self.describe()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReactionGroup":
        return cls(
            key=tuple(data.get("key") or []),
            label=data.get("label") or "",
            ladder_id=data.get("ladder_id"),
            pinned_uids=[int(u) for u in (data.get("pinned_uids") or [])],
        )


class GroupSet:
    """The reaction groups derived from a sheet, plus the ladder library."""

    def __init__(self) -> None:
        self.groups: List[ReactionGroup] = []
        self.ladders: Dict[str, Ladder] = {}
        self.label_template: str = DEFAULT_LABEL_TEMPLATE
        self._ladder_seq = 0

    # ---- ladders ----

    def new_ladder(self, name: str, points: Optional[List[float]] = None) -> Ladder:
        self._ladder_seq += 1
        ladder_id = "L%d" % self._ladder_seq
        while ladder_id in self.ladders:
            self._ladder_seq += 1
            ladder_id = "L%d" % self._ladder_seq
        ladder = Ladder(id=ladder_id, name=name or ladder_id, points=list(points or []))
        self.ladders[ladder_id] = ladder
        return ladder

    def ladder_matching(self, points: Iterable[float]) -> Optional[Ladder]:
        """Find an existing ladder with these exact points (dedupe on entry)."""
        wanted = [float(p) for p in points]
        return next((l for l in self.ladders.values() if l.points == wanted), None)

    def delete_ladder(self, ladder_id: str) -> bool:
        if ladder_id not in self.ladders:
            return False
        del self.ladders[ladder_id]
        for group in self.groups:
            if group.ladder_id == ladder_id:
                group.ladder_id = None
        return True

    # ---- derivation ----

    def members(self, sheet: Sheet) -> Dict[str, List[Row]]:
        """Map group key_id -> rows, honouring pinned overrides."""
        pinned: Dict[int, str] = {}
        for group in self.groups:
            for uid in group.pinned_uids:
                pinned[uid] = group.key_id

        buckets: Dict[str, List[Row]] = {g.key_id: [] for g in self.groups}
        for row in sheet.rows:
            target = pinned.get(row.uid) or "|".join(group_key(row))
            buckets.setdefault(target, []).append(row)
        return buckets

    def derive(self, sheet: Sheet) -> Dict[str, Any]:
        """Refresh the group list from the sheet, keeping existing settings.

        Groups the user has already labelled or assigned a ladder to keep
        those settings; groups whose rows have gone are dropped.
        """
        existing = {g.key_id: g for g in self.groups}
        seen: List[str] = []
        for row in sheet.rows:
            key = group_key(row)
            if not any(key):
                continue
            key_id = "|".join(key)
            if key_id not in existing:
                existing[key_id] = ReactionGroup(key=key)
            if key_id not in seen:
                seen.append(key_id)

        # Keep any group that still has pinned rows even if no row matches
        # its key any more (that is the whole point of pinning).
        live_uids = {row.uid for row in sheet.rows}
        for key_id, group in existing.items():
            group.pinned_uids = [u for u in group.pinned_uids if u in live_uids]
            if key_id not in seen and group.pinned_uids:
                seen.append(key_id)

        self.groups = [existing[key_id] for key_id in seen]
        self._autolabel()
        return {"groups": len(self.groups)}

    def _autolabel(self) -> None:
        """Fill blank labels from the template, disambiguating duplicates.

        `{n}` is a per-template-collision counter, which is how labels like
        37_1 / 37_2 / 37_3 arise for several timecourses at one temperature.
        A construct suffix is added when groups at the same temperature use
        different constructs, mirroring the existing 37_2_A8C convention.
        """
        # Suffix only the *variant* constructs: the most common construct in
        # the run is treated as the default and left off the label, which is
        # how labels like 37_1 (wild type) and 37_2_A8C (mutant) arise.
        counts: Dict[str, int] = {}
        for group in self.groups:
            construct = group.describe().get("construct", "")
            if construct:
                counts[construct] = counts.get(construct, 0) + 1
        default_construct = max(counts, key=lambda c: counts[c]) if counts else ""
        multi_construct = len(counts) > 1

        counters: Dict[str, int] = {}
        used = {g.label for g in self.groups if g.label}
        for group in self.groups:
            if group.label:
                continue
            fields = group.describe()
            base = self.label_template
            for column, value in fields.items():
                base = base.replace("{%s}" % column, value or "")
            base = base.replace("{n}", "").rstrip("_-") or "group"
            counters[base] = counters.get(base, 0) + 1
            label = self.label_template
            for column, value in fields.items():
                label = label.replace("{%s}" % column, value or "")
            label = label.replace("{n}", str(counters[base]))
            label = re.sub(r"__+", "_", label).strip("_-")
            construct = fields.get("construct", "")
            if multi_construct and construct and construct != default_construct:
                label = "%s_%s" % (label, construct)
            while label in used:
                counters[base] += 1
                label = re.sub(r"_\d+(?=(_|$))", "_%d" % counters[base], label, count=1)
                if label in used:  # pragma: no cover - defensive
                    label = "%s_%d" % (label, counters[base])
            used.add(label)
            group.label = label

    # ---- applying to the sheet ----

    def apply(self, sheet: Sheet, write_labels: bool = True,
              write_times: bool = True) -> Dict[str, Any]:
        """Write reaction_group labels and ladder-derived reaction_time.

        Both are written with GROUP provenance, so they are visibly derived
        and still overridable by hand.
        """
        buckets = self.members(sheet)
        by_key = {g.key_id: g for g in self.groups}

        labels_written = times_written = 0
        missing_index: List[int] = []
        missing_ladder: List[str] = []
        out_of_range: List[Dict[str, Any]] = []

        for key_id, rows in buckets.items():
            group = by_key.get(key_id)
            if group is None:
                continue
            ladder = self.ladders.get(group.ladder_id) if group.ladder_id else None
            if write_times and ladder is None and rows:
                missing_ladder.append(group.label or key_id)

            for row in rows:
                if write_labels and group.label:
                    if row.set("reaction_group", group.label, GROUP):
                        labels_written += 1
                if not write_times or ladder is None:
                    continue
                index = timepoint_index(row)
                if index is None:
                    missing_index.append(row.uid)
                    continue
                seconds = ladder.seconds_for(index)
                if seconds is None:
                    out_of_range.append({
                        "uid": row.uid, "tp": index, "ladder": ladder.name,
                        "length": len(ladder.points),
                    })
                    continue
                value = int(seconds) if float(seconds).is_integer() else seconds
                if row.set("reaction_time", value, GROUP):
                    times_written += 1

        return {
            "labels_written": labels_written,
            "times_written": times_written,
            "groups_without_ladder": sorted(set(missing_ladder)),
            "rows_without_timepoint": len(missing_index),
            "timepoints_out_of_range": out_of_range[:20],
            "out_of_range_count": len(out_of_range),
        }

    # ---- reporting ----

    def summary(self, sheet: Sheet) -> List[Dict[str, Any]]:
        buckets = self.members(sheet)
        out: List[Dict[str, Any]] = []
        for group in self.groups:
            rows = buckets.get(group.key_id, [])
            indices = sorted({i for i in (timepoint_index(r) for r in rows) if i})
            ladder = self.ladders.get(group.ladder_id) if group.ladder_id else None
            entry = group.to_dict()
            entry.update({
                "row_count": len(rows),
                "uids": [r.uid for r in rows],
                "timepoints": indices,
                "max_timepoint": max(indices) if indices else 0,
                "ladder_name": ladder.name if ladder else None,
                "ladder_points": ladder.points if ladder else [],
                "needs_ladder": ladder is None and bool(indices),
                "ladder_too_short": bool(ladder and indices and max(indices) > len(ladder.points)),
            })
            out.append(entry)
        return out

    # ---- persistence ----

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label_template": self.label_template,
            "ladder_seq": self._ladder_seq,
            "ladders": [l.to_dict() for l in self.ladders.values()],
            "groups": [g.to_dict() for g in self.groups],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GroupSet":
        group_set = cls()
        if not data:
            return group_set
        group_set.label_template = data.get("label_template") or DEFAULT_LABEL_TEMPLATE
        group_set._ladder_seq = int(data.get("ladder_seq", 0))
        for entry in data.get("ladders") or []:
            ladder = Ladder.from_dict(entry)
            group_set.ladders[ladder.id] = ladder
        group_set.groups = [ReactionGroup.from_dict(e) for e in data.get("groups") or []]
        return group_set


def parse_ladder_text(text: str) -> List[float]:
    """Parse pasted times: commas, tabs, spaces or newlines all work.

    Accepts a unit suffix per value (30s, 5min, 1h) and bare numbers, which
    are taken as seconds. Built for pasting a column straight out of a
    spreadsheet or a notebook.
    """
    points: List[float] = []
    for token in re.split(r"[,\t\n;]+|\s{2,}", text or ""):
        token = token.strip()
        if not token:
            continue
        match = re.match(r"^(-?\d+\.?\d*)\s*([a-zA-Z]*)$", token)
        if not match:
            continue
        value = float(match.group(1))
        unit = match.group(2).lower()
        if unit in ("m", "min", "mins", "minute", "minutes"):
            value *= 60
        elif unit in ("h", "hr", "hrs", "hour", "hours"):
            value *= 3600
        points.append(value)
    return points
