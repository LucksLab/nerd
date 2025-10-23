"""
R-backed timecourse fitting engine that delegates all numerical work to an
external R script (nlme-based implementation).
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import json
import math

from nerd.utils.logging import get_logger

from .base import (
    NucleotideSeries,
    PerNucleotideFit,
    RoundResult,
    TimecourseEngine,
    TimecourseRequest,
    TimecourseResult,
    register_timecourse_engine,
)

log = get_logger(__name__)


def _json_default(value: Any) -> Any:
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, (Path,)):
        return str(value)
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def _ensure_float(value: Any) -> Optional[float]:
    try:
        if value in (None, "", False):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_log(value: Any, floor: float = 1e-12) -> Optional[float]:
    numeric = _ensure_float(value)
    if numeric is None:
        return None
    clamped = max(numeric, floor)
    try:
        return math.log(clamped)
    except (ValueError, OverflowError):
        return None


@register_timecourse_engine
class RIntegrationEngine(TimecourseEngine):
    """
    Execute the three-round timecourse workflow via an R/NLME backend.
    """

    name = "r_integration"
    version = "0.1.0"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._options = dict(kwargs or {})
        self._rscript_bin = self._resolve_rscript_bin(self._options.get("rscript_bin"))
        self._runner_path = self._resolve_runner_path(self._options.get("script_path"))

    @staticmethod
    def _resolve_rscript_bin(explicit: Optional[str]) -> str:
        if explicit in (None, ""):
            env_override = os.environ.get("NERD_RSCRIPT_BIN") or os.environ.get("NERD_RSCRIPT")
            return env_override or "Rscript"
        return str(explicit)

    @staticmethod
    def _resolve_runner_path(explicit: Optional[str]) -> Path:
        if explicit not in (None, ""):
            candidate = Path(explicit).expanduser()
            if not candidate.is_file():
                raise FileNotFoundError(f"Configured R runner script not found: {candidate}")
            return candidate

        candidate = Path(__file__).with_name("r_engine_runner.R")
        if not candidate.is_file():
            raise FileNotFoundError(
                f"Bundled R runner script missing: {candidate}. Did the package install correctly?"
            )
        return candidate

    def run(self, request: TimecourseRequest) -> TimecourseResult:
        payload = self._build_payload(request)

        keep_tmp = bool(
            self._options.get("keep_tmp_dir")
            or os.environ.get("NERD_R_ENGINE_KEEP_TMP")
        )

        tmpdir_ctx = None
        try:
            if keep_tmp:
                tmpdir = tempfile.mkdtemp(prefix="nerd_r_engine_")
            else:
                tmpdir_ctx = tempfile.TemporaryDirectory(prefix="nerd_r_engine_")
                tmpdir = tmpdir_ctx.name

            tmp_dir = Path(tmpdir)
            input_path = tmp_dir / "input.json"
            output_path = tmp_dir / "output.json"
            log_path = tmp_dir / "r_stdout.log"

            json_payload = json.dumps(payload, default=_json_default)
            input_path.write_text(json_payload)

            cmd = [self._rscript_bin, str(self._runner_path), str(input_path), str(output_path)]
            log.debug("Invoking R engine: %s", " ".join(cmd))
            try:
                proc = subprocess.run(
                    cmd,
                    check=False,
                    capture_output=True,
                    text=True,
                )
            except FileNotFoundError as exc:
                raise RuntimeError(
                    f"Failed to invoke Rscript '{self._rscript_bin}'. Ensure R is installed and "
                    "the Rscript binary is on PATH or configured via engine_options.rscript_bin."
                ) from exc

            stdout_text = proc.stdout or ""
            stderr_text = proc.stderr or ""
            log_payload = (
                "STDOUT:\n"
                + stdout_text
                + ("\n" if not stdout_text.endswith("\n") else "")
                + "STDERR:\n"
                + stderr_text
            )
            log_path.write_text(log_payload)

            if proc.returncode != 0:
                details = (
                    f"R integration engine exited with status {proc.returncode}.\n"
                    "---- R STDOUT ----\n"
                    f"{stdout_text.strip()}\n"
                    "---- R STDERR ----\n"
                    f"{stderr_text.strip()}"
                )
                if keep_tmp:
                    details += f"\nTemporary files preserved at: {tmp_dir}"
                raise RuntimeError(details)

            if not output_path.is_file():
                raise RuntimeError(
                    f"R integration engine did not produce an output JSON: {output_path}"
                )

            output_data = json.loads(output_path.read_text())
        finally:
            if tmpdir_ctx is not None:
                tmpdir_ctx.cleanup()

        return self._build_result(request, output_data)

    def _resolve_initial_log_kdeg(self, request: TimecourseRequest) -> Optional[float]:
        candidates = [
            request.global_metadata or {},
            request.options or {},
        ]
        for source in candidates:
            for key in ("log_kdeg_initial", "initial_log_kdeg"):
                if key in source and source[key] not in (None, ""):
                    value = _ensure_float(source[key])
                    if value is not None:
                        return value
            for key in ("kdeg_initial", "initial_kdeg"):
                if key in source and source[key] not in (None, ""):
                    logged = _safe_log(source[key])
                    if logged is not None:
                        return logged
        return None

    def _resolve_constrained_log_kdeg(self, request: TimecourseRequest) -> Optional[float]:
        candidates = [
            request.options or {},
            request.global_metadata or {},
        ]
        for source in candidates:
            if "constrained_log_kdeg" in source and source["constrained_log_kdeg"] not in (None, ""):
                value = _ensure_float(source["constrained_log_kdeg"])
                if value is not None:
                    return value
            if "constrained_kdeg" in source and source["constrained_kdeg"] not in (None, ""):
                logged = _safe_log(source["constrained_kdeg"])
                if logged is not None:
                    return logged
        return None

    def _build_payload(self, request: TimecourseRequest) -> Dict[str, Any]:
        rounds = [str(r) for r in (request.rounds or ())]
        series: List[Dict[str, Any]] = []
        for series_idx, nucleotide in enumerate(request.nucleotides or []):
            metadata = dict(nucleotide.metadata or {})
            valtype = metadata.get("valtype")
            if valtype in (None, ""):
                valtype = ""

            fmod_ids = metadata.get("fmod_run_ids")
            if isinstance(fmod_ids, set):
                metadata["fmod_run_ids"] = sorted(fmod_ids)

            series.append(
                {
                    "series_id": f"{nucleotide.nt_id}:{valtype or 'NA'}:{series_idx}",
                    "nt_id": int(nucleotide.nt_id),
                    "valtype": valtype,
                    "timepoints": [float(t) for t in nucleotide.timepoints],
                    "fmod_values": [float(v) for v in nucleotide.fmod_values],
                    "metadata": metadata,
                }
            )

        options = dict(request.options or {})
        global_metadata = dict(request.global_metadata or {})

        payload: Dict[str, Any] = {
            "rg_id": int(request.rg_id),
            "rounds": rounds,
            "series": series,
            "options": options,
            "global_metadata": global_metadata,
            "resolved": {
                "initial_log_kdeg": self._resolve_initial_log_kdeg(request),
                "constrained_log_kdeg": self._resolve_constrained_log_kdeg(request),
            },
        }
        return payload

    def _build_result(
        self,
        request: TimecourseRequest,
        data: Mapping[str, Any],
    ) -> TimecourseResult:
        rounds_payload = data.get("rounds") or []
        round_results: List[RoundResult] = []

        for round_item in rounds_payload:
            per_nt_entries: List[PerNucleotideFit] = []
            for entry in round_item.get("per_nt") or []:
                per_nt_entries.append(
                    PerNucleotideFit(
                        nt_id=int(entry.get("nt_id")),
                        valtype=str(entry.get("valtype") or ""),
                        params=dict(entry.get("params") or {}),
                        diagnostics=dict(entry.get("diagnostics") or {}),
                    )
                )

            round_results.append(
                RoundResult(
                    round_id=str(round_item.get("round_id") or ""),
                    status=str(round_item.get("status") or "failed"),
                    per_nt=tuple(per_nt_entries),
                    global_params=dict(round_item.get("global_params") or {}),
                    qc_metrics=dict(round_item.get("qc_metrics") or {}),
                    notes=round_item.get("notes"),
                )
            )

        metadata = dict(data.get("metadata") or {})
        metadata.setdefault("rg_id", request.rg_id)
        metadata.setdefault("engine", self.name)

        artifacts = dict(data.get("artifacts") or {})

        return TimecourseResult(
            engine=self.name,
            engine_version=self.version,
            metadata=metadata,
            rounds=tuple(round_results),
            artifacts=artifacts,
        )
