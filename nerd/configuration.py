"""Project-aware analysis configuration validation, resolution, and templates."""

from __future__ import annotations

from copy import deepcopy
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import yaml

from nerd.project import ProjectConfig, ProjectContext
from nerd.scheduler.profiles import load_executor_profile
from nerd.utils.config import LoadedConfig, load_config


WORKFLOWS = {
    "create", "mut_count", "nmr_create", "nmr_kinetic_fit", "drop",
    "probe_timecourse", "tempgrad_fit",
}
ROOT_SECTIONS = WORKFLOWS | {"run", "executors", "containers"}
RUN_KEYS = {
    "label", "output_dir", "executor", "executor_profile", "backend",
    "threads", "mem_gb", "time", "remote", "slurm", "ssh", "slurm_config",
    "executors", "partition", "account", "qos", "constraint",
}
WORKFLOW_KEYS = {
    "create": {
        "samples", "sample", "construct", "constructs", "buffer", "buffers",
        "sequencing_run", "sequencing_runs", "derived_samples", "source", "force_run",
    },
    "mut_count": {
        "plugin", "tool", "reaction_group", "reaction_group_id", "rg_id", "params",
        "force_run", "overwrite", "samples", "sample_names", "derived_samples",
        "threads", "dry_run", "amplicon", "dms_mode", "output_N7",
        "output_parsed_mutations", "per_read_histograms",
    },
    "nmr_create": {"search_roots", "reactions", "force_run", "overwrite"},
    "nmr_kinetic_fit": {
        "fit_type", "search_roots", "plugin", "plugin_options", "fit_params", "species",
        "reaction_ids", "reactions", "select", "trace_roles", "model", "substrates",
        "force_run", "overwrite",
    },
    "drop": {
        "sample_name", "sample_names", "samples", "samples_yaml", "samples_config",
        "config", "config_path", "yaml", "to_drop", "on_missing", "force_run",
    },
    "probe_timecourse": {
        "engine", "rounds", "rg_ids", "reaction_group_ids", "reaction_groups", "rg",
        "nt_ids", "valtype", "fmod_run_id", "include_dropped_samples", "outliers",
        "rt_protocol", "RT_protocol", "done_by",
        "min_points", "overwrite", "force_run", "engine_options", "global_metadata",
        "filters",
    },
    "tempgrad_fit": {
        "mode", "engine", "data_source", "filters", "engine_options", "metadata",
        "group_by", "aggregate", "overwrite", "force_run", "series", "use_probe_tc",
        "model", "fit_name", "outliers", "seed_from_fit", "temperature_unit",
    },
}
SECRET_MARKERS = (
    "password", "passphrase", "private_key", "secret", "token", "api_key",
    "access_key", "identity_file", "key_file",
)


class ConfigValidationError(ValueError):
    """An actionable analysis configuration error."""


def _suggest(value: str, choices: Iterable[str]) -> str:
    match = get_close_matches(value, sorted(choices), n=1, cutoff=0.6)
    return " Did you mean %r?" % match[0] if match else ""


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def resolve_config(
    path: Path, *, context: Optional[ProjectContext] = None,
    executor: Optional[str] = None,
) -> Tuple[LoadedConfig, Optional[ProjectConfig]]:
    """Merge project defaults -> source-relative YAML -> explicit executor."""
    source = load_config(path)
    forbidden = set(source) & {"project", "project_id", "project_name"}
    if forbidden:
        raise ConfigValidationError(
            "Analysis YAML must not redefine project identity (%s); set the canonical "
            "name only in .nerd/project.toml." % ", ".join(sorted(forbidden))
        )
    project = (context or ProjectContext()).resolve_project()
    merged: Dict[str, Any] = dict(source)
    inherited: Dict[str, Any] = {}
    if project is not None:
        inherited = project.analysis_defaults()
        merged = _deep_merge(inherited, merged)
    if executor:
        run_block = merged.setdefault("run", {})
        if not isinstance(run_block, dict):
            raise ConfigValidationError("run must be a mapping.")
        run_block["executor"] = executor
    resolved = LoadedConfig(merged, source.source_path)
    # Scientific cache identity uses authored YAML plus effective executor/tool
    # settings, not project name/root/database metadata.
    resolved.hash_data = deepcopy(source.hash_data)
    resolved.authored = deepcopy(source.hash_data)
    if project is not None:
        effective_run = merged.get("run", {}) or {}
        effective_profiles = merged.get("executors", {}) or {}
        if not isinstance(effective_run, dict):
            effective_run = {}
        if not isinstance(effective_profiles, dict):
            effective_profiles = {}
        if effective_run.get("executor"):
            resolved.hash_data.setdefault("run", {})["executor"] = effective_run["executor"]
        selected = effective_run.get("executor") or effective_run.get("executor_profile")
        if selected and selected in effective_profiles:
            hash_profiles = resolved.hash_data.setdefault("executors", {})
            if isinstance(hash_profiles, dict):
                hash_profiles[selected] = deepcopy(effective_profiles[selected])
        if project.containers:
            resolved.hash_data["containers"] = deepcopy(project.containers)
    if executor:
        hash_run = resolved.hash_data.setdefault("run", {})
        if isinstance(hash_run, dict):
            hash_run["executor"] = executor
    resolved.project = project
    resolved.inherited = inherited
    return resolved, project


def selected_workflow(cfg: Mapping[str, Any], requested: Optional[str] = None) -> str:
    present = [name for name in WORKFLOWS if name in cfg]
    if requested:
        if requested not in WORKFLOWS:
            raise ConfigValidationError(
                "Unknown workflow %r.%s" % (requested, _suggest(requested, WORKFLOWS))
            )
        if requested not in cfg:
            raise ConfigValidationError("Configuration must contain a %r section." % requested)
        return requested
    if len(present) != 1:
        raise ConfigValidationError(
            "Configuration must select exactly one workflow section; found %s. "
            "Choose one of: %s." % (len(present), ", ".join(sorted(WORKFLOWS)))
        )
    return present[0]


def _check_known_sections(cfg: Mapping[str, Any]) -> None:
    for key in cfg:
        if key not in ROOT_SECTIONS:
            raise ConfigValidationError(
                "Unknown top-level configuration section %r.%s" % (key, _suggest(str(key), ROOT_SECTIONS))
            )
    run = cfg.get("run") or {}
    if not isinstance(run, dict):
        raise ConfigValidationError("run must be a mapping.")
    for key in run:
        if key not in RUN_KEYS:
            raise ConfigValidationError(
                "Unknown run key %r.%s" % (key, _suggest(str(key), RUN_KEYS))
            )
    executors = cfg.get("executors") or {}
    if not isinstance(executors, dict):
        raise ConfigValidationError("executors must be a mapping of named profiles.")


def _validate_plugin(workflow: str, block: Mapping[str, Any]) -> None:
    try:
        if workflow == "mut_count":
            plugin = block.get("plugin")
            if not plugin:
                raise ConfigValidationError("mut_count.plugin is required (for example, shapemapper).")
            from nerd.pipeline.plugins.mutcount import load_mutcount_plugin
            load_mutcount_plugin(str(plugin))
        elif workflow == "nmr_kinetic_fit" and block.get("plugin"):
            from nerd.pipeline.plugins.nmr_fit_kinetics import load_nmr_fit_plugin
            load_nmr_fit_plugin(str(block["plugin"]))
        elif workflow == "probe_timecourse" and block.get("engine"):
            from nerd.pipeline.plugins.timecourse import load_timecourse_engine
            load_timecourse_engine(str(block["engine"]))
        elif workflow == "tempgrad_fit" and block.get("engine"):
            from nerd.pipeline.plugins.tempgrad import load_tempgrad_engine
            load_tempgrad_engine(str(block["engine"]))
    except ConfigValidationError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigValidationError(str(exc)) from exc


def _validate_required(workflow: str, block: Mapping[str, Any]) -> None:
    if workflow == "create" and not any(
        block.get(key) for key in (
            "samples", "sample", "construct", "constructs", "buffer", "buffers",
            "sequencing_run", "sequencing_runs", "derived_samples", "source",
        )
    ):
        raise ConfigValidationError(
            "create requires at least one sample or metadata record/sheet."
        )
    if workflow == "mut_count" and not any(
        block.get(key) for key in (
            "samples", "sample_names", "derived_samples", "reaction_group", "reaction_groups"
        )
    ):
        raise ConfigValidationError(
            "mut_count requires samples, derived_samples, or reaction_group."
        )
    if workflow == "nmr_create":
        reactions = block.get("reactions")
        if not isinstance(reactions, list) or not reactions:
            raise ConfigValidationError("nmr_create.reactions must be a non-empty list.")
    if workflow == "nmr_kinetic_fit" and block.get("fit_type") not in {
        "degradation", "adduction",
    }:
        raise ConfigValidationError(
            "nmr_kinetic_fit.fit_type must be 'degradation' or 'adduction'."
        )
    if workflow == "drop" and not any(
        block.get(key) for key in (
            "sample_name", "sample_names", "samples", "samples_yaml", "samples_config",
            "config", "config_path", "yaml",
        )
    ):
        raise ConfigValidationError("drop requires sample names or a referenced samples config.")
    if workflow == "probe_timecourse" and not any(
        block.get(key) for key in ("rg_ids", "reaction_group_ids", "reaction_groups", "rg")
    ):
        raise ConfigValidationError("probe_timecourse requires reaction-group IDs (rg_ids).")
    if workflow == "tempgrad_fit" and block.get("mode") not in {
        "arrhenius", "two_state_melt",
    }:
        raise ConfigValidationError(
            "tempgrad_fit.mode must be 'arrhenius' or 'two_state_melt'."
        )


def validate_config(
    cfg: Mapping[str, Any], *, workflow: Optional[str] = None,
    check_inputs: bool = True,
) -> str:
    """Validate non-mutating structure, workflow, executor, and direct inputs."""
    if not isinstance(cfg, Mapping):
        raise ConfigValidationError("Configuration root must be a mapping.")
    _check_known_sections(cfg)
    chosen = selected_workflow(cfg, workflow)
    run = cfg.get("run") or {}
    if not run.get("label"):
        raise ConfigValidationError(
            "run.label is required and is a per-run label (for example, baseline-import); "
            "it does not need to repeat the project identifier."
        )
    block = cfg.get(chosen)
    if not isinstance(block, dict):
        raise ConfigValidationError("%s must be a mapping." % chosen)
    known_keys = WORKFLOW_KEYS[chosen]
    for key in block:
        if key not in known_keys:
            raise ConfigValidationError(
                "Unknown %s key %r.%s" % (chosen, key, _suggest(str(key), known_keys))
            )
    _validate_required(chosen, block)
    profiles = cfg.get("executors") or run.get("executors") or {}
    for name in profiles:
        try:
            load_executor_profile(dict(cfg), str(name))
        except ValueError as exc:
            raise ConfigValidationError(str(exc)) from exc
    try:
        load_executor_profile(dict(cfg))
    except ValueError as exc:
        raise ConfigValidationError(str(exc)) from exc
    _validate_plugin(chosen, block)
    if check_inputs:
        _validate_input_paths(cfg, chosen)
    return chosen


def _validate_input_paths(cfg: Mapping[str, Any], workflow: str) -> None:
    block = cfg.get(workflow) or {}
    direct_keys = {
        "samples", "samples_yaml", "samples_config", "source", "config_path",
        "constructs", "buffers", "sequencing_runs",
    }
    for key in direct_keys:
        value = block.get(key)
        if isinstance(value, str):
            path = Path(value)
            if not path.exists():
                raise ConfigValidationError("Input %s.%s does not exist: %s" % (workflow, key, path))


def redact(value: Any, key: str = "") -> Any:
    if any(marker in key.lower() for marker in SECRET_MARKERS):
        return "<redacted>"
    if isinstance(value, dict):
        return {str(k): redact(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


def resolved_view(
    cfg: LoadedConfig, project: Optional[ProjectConfig], workflow: str,
) -> Dict[str, Any]:
    run = cfg.get("run") or {}
    block = cfg.get(workflow) or {}
    profile = load_executor_profile(cfg)
    plugin = block.get("plugin")
    engine = block.get("engine")
    values: Dict[str, Any] = {
        "schema_version": "1.0",
        "project": {
            "id": project.name if project else None,
            "root": str(project.root) if project else None,
        },
        "database": str(project.database) if project else None,
        "output_directory": run.get("output_dir"),
        "config_base": str(cfg.base_dir),
        "config_file": str(cfg.source_path),
        "workflow": workflow,
        "task_label": run.get("label"),
        "executor": {"name": profile.name, "type": profile.executor_type},
        "plugin": plugin,
        "engine": engine,
        "resolved_input_paths": _collect_paths(block),
        "configured": redact(getattr(cfg, "authored", cfg.hash_data)),
        "inherited_defaults": redact(getattr(cfg, "inherited", {})),
    }
    return redact(values)


def _collect_paths(value: Any, key: str = "", prefix: str = "") -> Dict[str, str]:
    result: Dict[str, str] = {}
    if isinstance(value, dict):
        for child_key, child in value.items():
            child_prefix = "%s.%s" % (prefix, child_key) if prefix else str(child_key)
            result.update(_collect_paths(child, str(child_key), child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.update(_collect_paths(child, key, "%s[%d]" % (prefix, index)))
    elif isinstance(value, str) and (
        key in {
            "samples", "source", "path", "samples_yaml", "samples_config",
            "config", "config_path", "fq_dir", "kinetic_data_dir", "nt_info",
            "search_roots", "cache_dir", "runtime_dir", "container_sif",
        }
        or key.endswith("_path")
    ):
        result[prefix] = value
    return result


def template_for(workflow: str, *, samples_file: str = "samples.csv") -> Dict[str, Any]:
    """Return a minimal, parseable starter for an approved workflow."""
    if workflow not in WORKFLOWS:
        raise ConfigValidationError("Unknown workflow %r.%s" % (workflow, _suggest(workflow, WORKFLOWS)))
    base: Dict[str, Any] = {"run": {"label": "%s-run" % workflow}}
    blocks: Dict[str, Dict[str, Any]] = {
        "create": {"samples": samples_file},
        "mut_count": {"plugin": "shapemapper", "reaction_group": "1", "params": {}},
        "nmr_create": {"search_roots": ["data"], "reactions": [{
            "reaction_type": "deg", "temperature": 25.0, "replicate": 1,
            "num_scans": 64, "time_per_read": 5.0, "total_kinetic_reads": 96,
            "total_kinetic_time": 28800.0, "probe": "dms", "probe_conc": 0.01585,
            "probe_solvent": "etoh", "substrate": "none", "substrate_conc": 0.0,
            "buffer": "buffer-name", "nmr_machine": "instrument-name",
            "kinetic_data_dir": "data/reaction-01",
        }]},
        "nmr_kinetic_fit": {"fit_type": "degradation", "plugin": "lmfit_deg", "species": "dms"},
        "drop": {"sample_names": ["EKC.07.00.000_sample_001"], "on_missing": "warn"},
        "probe_timecourse": {"engine": "python_baseline", "rg_ids": [1], "valtype": "modrate", "min_points": 3},
        "tempgrad_fit": {"mode": "arrhenius", "engine": "arrhenius_python", "data_source": "nmr", "filters": {}},
    }
    base[workflow] = blocks[workflow]
    return base


def write_template(workflow: str, output: Path) -> Tuple[Path, Optional[Path]]:
    output = output.expanduser().resolve()
    if output.exists():
        raise ConfigValidationError("Refusing to overwrite existing config: %s" % output)
    output.parent.mkdir(parents=True, exist_ok=True)
    companion: Optional[Path] = None
    samples_name = "%s-samples.csv" % output.stem
    cfg = template_for(workflow, samples_file=samples_name)
    output.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    if workflow == "create":
        companion = output.with_name(samples_name)
        if companion.exists():
            output.unlink()
            raise ConfigValidationError("Refusing to overwrite existing sample sheet: %s" % companion)
        companion.write_text(
            "sample_name,fq_source,fq_dir,r1_file,r2_file,reaction_group,temperature,replicate,"
            "reaction_time,probe,probe_concentration,rt_protocol,treated,buffer,construct,done_by\n",
            encoding="utf-8",
        )
    return output, companion
