"""Naming-pattern engine: bracketed tokens with user-typed literals."""
from .tokens import TokenSpec, TokenRegistry  # noqa: F401
from .compile import CompiledPattern, compile_pattern, parse_name  # noqa: F401
