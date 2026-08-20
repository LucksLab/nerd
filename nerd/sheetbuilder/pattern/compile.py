"""Compile a user-typed naming pattern into a regex.

Syntax: bracketed token names with arbitrary literal text between them,
e.g. `[id]-[doneby]-[construct]-[temp]-[replicate]-[tp_num]-[treated]_[*]`.
Literals are whatever the user types (any length: `-`, `_`, `+`, `_tp`).

The important design choice is how a token's matcher is derived. Rather
than each token carrying a fixed character class -- which breaks the first
time a value contains a character the class forgot, and needs widening
again for the next one -- a token consumes *everything up to the literal
that follows it*. `[buffer]_` compiles to `[^_]+`, so "0mM-THF" matches
without anyone anticipating the hyphen. Interpretation stays strict via
the token's normalizer; only matching is permissive.

Tokens may still pin an explicit `match` when they need to be narrower
than the delimiter rule allows.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .tokens import TokenRegistry, TokenSpec, WILDCARD, WILDCARD_ALIASES

_TOKEN_RE = re.compile(r"\[([^\[\]]*)\]")


class PatternError(ValueError):
    """Raised for a pattern the user can fix by editing it."""


@dataclass
class CompiledPattern:
    raw: str
    regex: "re.Pattern[str]"
    # (group_name, token_name) in match order; group names are unique even
    # when a token is used more than once.
    groups: List[Tuple[str, str]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def token_names(self) -> List[str]:
        return [token for _, token in self.groups]


def _matcher_for(spec: Optional[TokenSpec], token_name: str, next_literal: str) -> str:
    """Derive the regex fragment a token should consume.

    Preference order: the token's own pinned `match`; else everything up to
    the next literal's first character; else (nothing follows) the rest of
    the string.
    """
    if spec is not None and spec.match:
        return spec.match

    is_wildcard = token_name in WILDCARD_ALIASES

    if next_literal:
        stop = next_literal[0]
        if is_wildcard:
            # Lazy so a trailing [*] does not swallow later delimiters.
            return "[^%s]*?" % re.escape(stop)
        return "[^%s]+" % re.escape(stop)

    # Nothing follows: consume the remainder.
    return ".*" if is_wildcard else ".+"


def compile_pattern(pattern: str, registry: TokenRegistry) -> CompiledPattern:
    if not pattern or not pattern.strip():
        raise PatternError("Pattern is empty.")

    # Split into alternating literals and token names.
    literals: List[str] = []
    token_names: List[str] = []
    cursor = 0
    for match in _TOKEN_RE.finditer(pattern):
        literals.append(pattern[cursor:match.start()])
        name = match.group(1).strip()
        if not name:
            raise PatternError("Found an empty token '[]' in the pattern.")
        token_names.append(name)
        cursor = match.end()
    trailing_literal = pattern[cursor:]

    if not token_names:
        raise PatternError(
            "Pattern has no tokens. Add at least one, e.g. [construct] -- "
            "click a token chip to insert it."
        )

    unknown = [
        name for name in token_names
        if name not in WILDCARD_ALIASES and registry.get(name) is None
    ]
    if unknown:
        raise PatternError(
            "Unknown token(s): %s. Register them in the token panel first."
            % ", ".join("[%s]" % n for n in unknown)
        )

    parts: List[str] = []
    groups: List[Tuple[str, str]] = []
    warnings: List[str] = []
    seen: Dict[str, int] = {}

    for index, token_name in enumerate(token_names):
        parts.append(re.escape(literals[index]))
        next_literal = (
            literals[index + 1] if index + 1 < len(literals) else trailing_literal
        )
        if not next_literal and index + 1 < len(token_names):
            warnings.append(
                "[%s] is immediately followed by another token with no "
                "delimiter between them; the split may be ambiguous."
                % token_name
            )

        spec = registry.get(token_name)
        matcher = _matcher_for(spec, token_name, next_literal)

        # Unique, regex-safe group name (token names may repeat or contain *).
        base = re.sub(r"\W", "_", token_name) or "wild"
        seen[base] = seen.get(base, 0) + 1
        group_name = "g%d_%s" % (seen[base], base)

        is_last = index + 1 == len(token_names)
        if is_last and token_name in WILDCARD_ALIASES and not trailing_literal:
            # A trailing [*] means "ignore whatever else is here" -- so the
            # separator in front of it is optional too. One pattern then
            # covers both "..._p_S54_L001" and the same name with the
            # machine suffix already stripped off.
            literal_part = parts.pop()
            parts.append("(?:%s(?P<%s>%s))?" % (literal_part, group_name, matcher))
        else:
            parts.append("(?P<%s>%s)" % (group_name, matcher))
        groups.append((group_name, token_name))

    parts.append(re.escape(trailing_literal))

    try:
        regex = re.compile("^" + "".join(parts))
    except re.error as exc:  # pragma: no cover - defensive
        raise PatternError("Could not compile pattern: %s" % exc)

    return CompiledPattern(raw=pattern, regex=regex, groups=groups, warnings=warnings)


def parse_name(name: str, compiled: CompiledPattern, registry: TokenRegistry) -> Dict[str, Any]:
    """Parse one sample name.

    Returns {matched, trailing, values: {column: value}, raw: {token: text}}.
    `values` is keyed by sheet column (via each token's maps_to) and omits
    tokens that map nowhere, so the caller can write it straight into a row.
    """
    result: Dict[str, Any] = {
        "matched": False, "trailing": "", "values": {}, "raw": {},
    }
    match = compiled.regex.match(name or "")
    if not match:
        return result

    result["matched"] = True
    if match.end() < len(name or ""):
        result["trailing"] = name[match.end():]

    for group_name, token_name in compiled.groups:
        if token_name in WILDCARD_ALIASES:
            continue
        text = match.group(group_name)
        spec = registry.get(token_name)
        result["raw"][token_name] = text
        if spec is None or not spec.maps_to:
            continue
        result["values"][spec.maps_to] = spec.apply(text)

    return result
