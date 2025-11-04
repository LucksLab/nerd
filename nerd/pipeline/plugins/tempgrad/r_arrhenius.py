"""
R-backed Arrhenius fitting engine that delegates log-linear regression to an
external R script.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

import numpy as np

from nerd.utils.logging import get_logger

from .base import (
    SeriesFitResult,
    TempgradEngine,
    TempgradRequest,
    TempgradResult,
    TempgradSeries,
    register_tempgrad_engine,
)

log = get_logger(__name__)

R_GAS_CONSTANT = 1.98720425864083  # cal mol^-1 K^-1


def _json_default(value: Any) -> Any:
    if isinstance(value, (Path,)):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def _coerce_float_sequence(values: Iterable[Any], *, kind: str, series_id: str) -> np.ndarray:
    try:
        array = np.asarray(list(values), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Series {series_id} contains non-numeric {kind}.") from exc
    if array.ndim != 1:
        raise ValueError(f"Series {series_id} {kind} must be one-dimensional.")
    return array


def _convert_temperatures(temps: np.ndarray, *, unit: str, series_id: str) -> np.ndarray:
    norm = unit.lower()
    if norm in {"c", "celsius"}:
        temps_k = temps + 273.15
    elif norm in {"k", "kelvin"}:
        temps_k = temps
    elif norm in {"f", "fahrenheit"}:
        temps_k = (temps - 32.0) * 5.0 / 9.0 + 273.15
    else:
        raise ValueError(f"Series {series_id} has unsupported temperature unit '{unit}'.")

    if np.any(~np.isfinite(temps_k)):
        raise ValueError(f"Series {series_id} contains non-finite temperatures after conversion.")
    if np.any(temps_k <= 0):
        raise ValueError(f"Series {series_id} contains non-positive absolute temperatures.")
    return temps_k


def _extract_log_rate_std_errors(series: TempgradSeries, rates: np.ndarray) -> Optional[List[float]]:
    metadata = series.metadata or {}
    candidate_log_keys = (
        "log_rate_std_errors",
        "log_rate_std_err",
        "log_rate_sigma",
        "log_rate_se",
    )
    for key in candidate_log_keys:
        if key in metadata and metadata[key] not in (None, ""):
            values = _coerce_float_sequence(metadata[key], kind=key, series_id=series.series_id)
            if values.shape != rates.shape:
                raise ValueError(f"Series {series.series_id} has mismatched lengths for '{key}'.")
            if np.any(values <= 0) or np.any(~np.isfinite(values)):
                raise ValueError(f"Series {series.series_id} contains invalid entries for '{key}'.")
            return [float(x) for x in values]

    candidate_rate_keys = (
        "rate_std_errors",
        "rate_std_err",
        "rate_sigma",
        "rate_se",
        "std_errors",
        "std_err",
    )
    for key in candidate_rate_keys:
        if key in metadata and metadata[key] not in (None, ""):
            values = _coerce_float_sequence(metadata[key], kind=key, series_id=series.series_id)
            if values.shape != rates.shape:
                raise ValueError(f"Series {series.series_id} has mismatched lengths for '{key}'.")
            if np.any(values <= 0) or np.any(~np.isfinite(values)):
                raise ValueError(f"Series {series.series_id} contains invalid entries for '{key}'.")
            log_errors = np.asarray(values, dtype=float) / rates
            log_errors = np.clip(log_errors, 1e-12, None)
            return [float(x) for x in log_errors]
    return None


def _build_series_payload(series: TempgradSeries, options: MutableMapping[str, Any]) -> Dict[str, Any]:
    temps = _coerce_float_sequence(series.x_values, kind="temperatures", series_id=series.series_id)
    rates = _coerce_float_sequence(series.y_values, kind="rate constants", series_id=series.series_id)
    if temps.size != rates.size:
        raise ValueError(f"Series {series.series_id} has mismatched temperature/rate lengths.")
    if temps.size < 2:
        raise ValueError(f"Series {series.series_id} requires at least two observations for Arrhenius fitting.")

    temp_unit = (
        options.get("temperature_unit")
        or series.metadata.get("temperature_unit")
        or "c"
    )
    temps_k = _convert_temperatures(temps, unit=str(temp_unit), series_id=series.series_id)

    if np.any(rates <= 0):
        raise ValueError(f"Series {series.series_id} contains non-positive rate constants.")

    if np.any(~np.isfinite(rates)):
        raise ValueError(f"Series {series.series_id} contains non-finite rate constants.")

    inv_t = 1.0 / temps_k
    log_rates = np.log(rates)

    weights_payload: Optional[List[float]] = None
    if series.weights is not None:
        weights = _coerce_float_sequence(series.weights, kind="weights", series_id=series.series_id)
        if weights.shape != inv_t.shape:
            raise ValueError(f"Series {series.series_id} has mismatched weight dimensions.")
        if np.any(weights <= 0):
            raise ValueError(f"Series {series.series_id} contains non-positive weights.")
        weights_payload = [float(x) for x in weights]

    payload: Dict[str, Any] = {
        "series_id": series.series_id,
        "inv_t": [float(x) for x in inv_t],
        "log_rates": [float(x) for x in log_rates],
        "weights": weights_payload,
        "metadata": dict(series.metadata or {}),
    }

    log_rate_std_errors = _extract_log_rate_std_errors(series, rates)
    if log_rate_std_errors is not None:
        payload["log_rate_std_errors"] = log_rate_std_errors

    group_id = (series.metadata or {}).get("group_id")
    if group_id is None:
        group_id = (series.metadata or {}).get("group")
    if group_id is not None:
        payload["group_id"] = group_id

    return payload


@register_tempgrad_engine
class ArrheniusREngine(TempgradEngine):
    """Log-linear Arrhenius regression via an external R backend."""

    name = "arrhenius_r"
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

        candidate = Path(__file__).with_name("r_arrhenius_runner.R")
        if not candidate.is_file():
            raise FileNotFoundError(
                f"Bundled Arrhenius R runner script missing: {candidate}. "
                "Did the package install correctly?"
            )
        return candidate

    def run(self, request: TempgradRequest) -> TempgradResult:
        series_metadata = {series.series_id: dict(series.metadata or {}) for series in request.series}
        payload = self._build_payload(request)
        output_payload = self._invoke_r(payload)
        return self._build_result(request, output_payload, series_metadata)

    def _build_payload(self, request: TempgradRequest) -> Dict[str, Any]:
        options = dict(request.options or {})
        series_payload = []
        for series in request.series:
            series_payload.append(_build_series_payload(series, options))

        payload: Dict[str, Any] = {
            "mode": request.mode,
            "options": options,
            "metadata": dict(request.metadata or {}),
            "series": series_payload,
        }
        engine_options = {
            key: value
            for key, value in (self._options or {}).items()
            if key not in {"script_path", "rscript_bin"}
        }
        if engine_options:
            payload["engine_options"] = engine_options
        return payload

    def _invoke_r(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        keep_tmp = bool(
            self._options.get("keep_tmp_dir")
            or os.environ.get("NERD_R_ENGINE_KEEP_TMP")
        )

        tmpdir_ctx = None
        try:
            if keep_tmp:
                tmpdir = tempfile.mkdtemp(prefix="nerd_arrhenius_r_")
            else:
                tmpdir_ctx = tempfile.TemporaryDirectory(prefix="nerd_arrhenius_r_")
                tmpdir = tmpdir_ctx.name

            tmp_dir = Path(tmpdir)
            input_path = tmp_dir / "input.json"
            output_path = tmp_dir / "output.json"
            log_path = tmp_dir / "r_stdout.log"

            input_path.write_text(json.dumps(payload, default=_json_default))

            cmd = [self._rscript_bin, str(self._runner_path), str(input_path), str(output_path)]
            log.debug("Invoking Arrhenius R engine: %s", " ".join(cmd))
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
            log_path.write_text(
                "STDOUT:\n"
                + stdout_text
                + ("" if stdout_text.endswith("\n") else "\n")
                + "STDERR:\n"
                + stderr_text
            )

            if proc.returncode != 0:
                details = (
                    f"Arrhenius R engine exited with status {proc.returncode}.\n"
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
                    f"Arrhenius R engine did not produce an output JSON: {output_path}"
                )

            output_payload = json.loads(output_path.read_text())
        finally:
            if tmpdir_ctx is not None:
                tmpdir_ctx.cleanup()

        if not isinstance(output_payload, Mapping):
            raise RuntimeError("Arrhenius R engine produced an invalid payload (expected JSON object).")
        return output_payload

    def _build_result(
        self,
        request: TempgradRequest,
        data: Mapping[str, Any],
        series_metadata: Mapping[str, Dict[str, Any]],
    ) -> TempgradResult:
        series_results_payload = data.get("series") or []
        if not isinstance(series_results_payload, list):
            raise RuntimeError("Arrhenius R engine output missing 'series' list.")

        series_index = {}
        for entry in series_results_payload:
            if isinstance(entry, Mapping):
                sid = str(entry.get("series_id") or "")
                if sid:
                    series_index[sid] = entry

        results: List[SeriesFitResult] = []
        for series in request.series:
            record = series_index.get(series.series_id)
            if record is None:
                diagnostics = {
                    "status": "failed",
                    "reason": "Series missing from R output.",
                }
                params = {}
                artifacts = {}
            else:
                params = dict(record.get("params") or {})
                diagnostics = dict(record.get("diagnostics") or {})
                artifacts = dict(record.get("artifacts") or {})
                status = record.get("status")
                reason = record.get("reason")
                if status:
                    diagnostics.setdefault("status", status)
                if reason:
                    diagnostics.setdefault("reason", reason)

            metadata = series_metadata.get(series.series_id)
            if metadata:
                diagnostics.update(metadata)

            results.append(
                SeriesFitResult(
                    series_id=series.series_id,
                    params=params,
                    diagnostics=diagnostics,
                    artifacts=artifacts,
                )
            )

        metadata = dict(request.metadata or {})
        metadata.setdefault("mode", request.mode)
        metadata.update(dict(data.get("metadata") or {}))

        artifacts = dict(data.get("artifacts") or {})
        global_params = dict(data.get("global_params") or {})

        return TempgradResult(
            engine=self.name,
            engine_version=self.version,
            metadata=metadata,
            series_results=tuple(results),
            global_params=global_params,
            artifacts=artifacts,
        )


@register_tempgrad_engine
class ArrheniusRBayesianEngine(ArrheniusREngine):
    """Bayesian Arrhenius regression backed by brms with measurement-error support."""

    name = "arrhenius_r_bayesian"
    version = "0.1.0"

    @staticmethod
    def _resolve_runner_path(explicit: Optional[str]) -> Path:
        if explicit not in (None, ""):
            candidate = Path(explicit).expanduser()
            if not candidate.is_file():
                raise FileNotFoundError(f"Configured Bayesian R runner script not found: {candidate}")
            return candidate

        candidate = Path(__file__).with_name("r_arrhenius_bayesian_runner.R")
        if not candidate.is_file():
            raise FileNotFoundError(
                f"Bundled Arrhenius Bayesian R runner script missing: {candidate}. "
                "Did the package install correctly?"
            )
        return candidate

    def _build_payload(self, request: TempgradRequest) -> Dict[str, Any]:
        payload = super()._build_payload(request)
        bayes_options: Dict[str, Any] = {}

        engine_bayes = self._options.get("bayes", {})
        if isinstance(engine_bayes, Mapping):
            bayes_options.update(engine_bayes)

        request_options = dict(payload.get("options") or {})
        for key, value in list(request_options.items()):
            if key.startswith("bayes_"):
                bayes_options[key[6:]] = value

        if bayes_options:
            payload["bayes_options"] = bayes_options

        return payload
