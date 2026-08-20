"""Token registry for the sample-name pattern engine.

A token has two independent halves, and keeping them separate is what
makes the engine robust:

  * a MATCHER, which decides how much of the name the token consumes. By
    default a token does not carry one at all -- the compiler derives it
    from whichever literal the user typed after the token (see
    compile.py). A token only pins `match` when it genuinely needs to be
    narrower than "everything up to the next delimiter".
  * a NORMALIZER, which turns the matched text into the value written to
    the sheet: `value_map` for controlled vocabularies, then `normalize`
    for numeric extraction.

Splitting them means a token like [tp_num] can match "tp2" or "2" or
"tp02" and still yield the integer 2, instead of needing its character
class widened every time a new naming habit shows up.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

DEFAULT_TOKENS_PATH = Path(__file__).parent / "default_tokens.yaml"

WILDCARD = "*"
WILDCARD_ALIASES = {"*", "ignore", "skip"}


def _first_number(text: str) -> Optional[str]:
    m = re.search(r"-?\d+\.?\d*", text)
    return m.group() if m else None


@dataclass
class TokenSpec:
    name: str
    # Explicit matcher. Left None for most tokens so the compiler can infer
    # one from the following literal.
    match: Optional[str] = None
    # Sheet column this token writes to. None = parsed but not written
    # (useful for [id], or for parts the user only wants as context).
    maps_to: Optional[str] = None
    # Applied after value_map: "int" | "float" | "str" | None
    normalize: Optional[str] = None
    # Controlled vocabulary, matched case-insensitively on the raw text.
    value_map: Dict[str, str] = field(default_factory=dict)
    # Multiplier applied after normalize. Lets one token read a time written
    # in minutes and still write seconds ([time_min] -> scale 60), which is
    # how the older "-stop-<minutes>" sample names encode reaction time.
    scale: Optional[float] = None
    description: str = ""

    def apply(self, raw: str) -> Any:
        text = (raw or "").strip()
        mapped = self.value_map.get(text.lower())
        if mapped is not None:
            text = mapped
        if self.normalize in ("int", "float"):
            number = _first_number(text)
            if number is None:
                return text
            try:
                value = float(number)
            except ValueError:
                return text
            if self.scale is not None:
                value *= self.scale
            if self.normalize == "int":
                return int(round(value))
            return int(value) if value.is_integer() else value
        return text


class TokenRegistry:
    def __init__(self, tokens: Optional[Dict[str, TokenSpec]] = None):
        self._tokens: Dict[str, TokenSpec] = tokens or {}

    def get(self, name: str) -> Optional[TokenSpec]:
        if name in WILDCARD_ALIASES:
            return self._tokens.get(WILDCARD) or TokenSpec(name=WILDCARD)
        return self._tokens.get(name)

    def register(self, spec: TokenSpec) -> None:
        self._tokens[spec.name] = spec

    def remove(self, name: str) -> bool:
        return self._tokens.pop(name, None) is not None

    def all(self) -> Dict[str, TokenSpec]:
        return dict(self._tokens)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "TokenRegistry":
        path = Path(path) if path else DEFAULT_TOKENS_PATH
        if not path.is_file():
            return cls()
        data = yaml.safe_load(path.read_text()) or {}
        tokens: Dict[str, TokenSpec] = {}
        for name, cfg in (data.get("tokens") or {}).items():
            cfg = dict(cfg or {})
            cfg.pop("name", None)
            tokens[str(name)] = TokenSpec(name=str(name), **cfg)
        return cls(tokens)

    def save(self, path: Optional[Path] = None) -> None:
        path = Path(path) if path else DEFAULT_TOKENS_PATH
        payload = {
            "tokens": {
                name: {k: v for k, v in asdict(spec).items()
                       if k != "name" and v not in (None, {}, "")}
                for name, spec in self._tokens.items()
            }
        }
        path.write_text(yaml.safe_dump(payload, sort_keys=False))
