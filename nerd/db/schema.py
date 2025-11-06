"""
Schema definitions for initializing the project database.

Tables are grouped by domain:
  • core_*      task orchestration metadata
  • meta_*      shared metadata (buffers, constructs, nucleotides)
  • probe_*     probing chemistry experiments and derived fits
  • nmr_*       NMR experiments and fits
  • tempgrad_*  temperature-gradient / melt fits

Tall parameter tables follow the pattern (fit_run_id, param_name, param_numeric,
param_text, units) so new metrics can be added without migrations.
"""

# ---------------------------------------------------------------------------
# Metadata tables
# ---------------------------------------------------------------------------

CREATE_META_BUFFERS = """
CREATE TABLE IF NOT EXISTS meta_buffers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    pH REAL NOT NULL,
    composition TEXT NOT NULL,
    disp_name TEXT NOT NULL,
    UNIQUE(name, pH, composition, disp_name)
);
"""

CREATE_META_CONSTRUCTS = """
CREATE TABLE IF NOT EXISTS meta_constructs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    family TEXT NOT NULL,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    sequence TEXT NOT NULL,
    disp_name TEXT NOT NULL,
    UNIQUE(family, name, version, sequence, disp_name)
);
"""

CREATE_META_NUCLEOTIDES = """
CREATE TABLE IF NOT EXISTS meta_nucleotides (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    construct_id INTEGER NOT NULL,
    site INTEGER NOT NULL,
    base TEXT NOT NULL,
    base_region TEXT NOT NULL,
    FOREIGN KEY(construct_id) REFERENCES meta_constructs(id) ON DELETE CASCADE,
    UNIQUE(construct_id, site)
);
"""

# ---------------------------------------------------------------------------
# Sequencing metadata
# ---------------------------------------------------------------------------

CREATE_SEQUENCING_RUNS = """
CREATE TABLE IF NOT EXISTS sequencing_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_name TEXT NOT NULL,
    date TEXT NOT NULL,
    sequencer TEXT NOT NULL,
    run_manager TEXT NOT NULL,
    UNIQUE(run_name, date, sequencer, run_manager)
);
"""

CREATE_SEQUENCING_SAMPLES = """
CREATE TABLE IF NOT EXISTS sequencing_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    seqrun_id INTEGER NOT NULL,
    sample_name TEXT NOT NULL,
    fq_dir TEXT NOT NULL,
    r1_file TEXT NOT NULL,
    r2_file TEXT NOT NULL,
    to_drop INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY(seqrun_id) REFERENCES sequencing_runs(id),
    UNIQUE(seqrun_id, sample_name, fq_dir)
);
"""

CREATE_SEQUENCING_DERIVED_SAMPLES = """
CREATE TABLE IF NOT EXISTS sequencing_derived_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_sample_id INTEGER NOT NULL,
    child_name TEXT NOT NULL,
    kind TEXT NOT NULL,
    tool TEXT NOT NULL,
    cmd_template TEXT NOT NULL,
    params_json TEXT,
    cache_key TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY(parent_sample_id) REFERENCES sequencing_samples(id) ON DELETE CASCADE,
    UNIQUE(parent_sample_id, child_name)
);
"""

# ---------------------------------------------------------------------------
# Probe experiments
# ---------------------------------------------------------------------------

CREATE_PROBE_REACTION_GROUPS = """
CREATE TABLE IF NOT EXISTS probe_reaction_groups (
    rg_id INTEGER PRIMARY KEY,
    rg_label TEXT UNIQUE
);
"""

CREATE_PROBE_REACTIONS = """
CREATE TABLE IF NOT EXISTS probe_reactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rg_id INTEGER NOT NULL,
    s_id INTEGER NOT NULL,
    construct_id INTEGER NOT NULL,
    buffer_id INTEGER NOT NULL,
    temperature REAL NOT NULL,
    replicate INTEGER NOT NULL,
    reaction_time REAL NOT NULL,
    probe_concentration REAL NOT NULL,
    probe TEXT NOT NULL,
    rt_protocol TEXT NOT NULL,
    done_by TEXT NOT NULL,
    treated INTEGER NOT NULL,
    FOREIGN KEY(rg_id) REFERENCES probe_reaction_groups(rg_id),
    FOREIGN KEY(s_id) REFERENCES sequencing_samples(id),
    FOREIGN KEY(construct_id) REFERENCES meta_constructs(id),
    FOREIGN KEY(buffer_id) REFERENCES meta_buffers(id)
);
"""

CREATE_PROBE_TEMPGRAD_GROUPS = """
CREATE TABLE IF NOT EXISTS probe_tempgrad_groups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tg_id INTEGER NOT NULL,
    rg_id INTEGER NOT NULL,
    buffer_id INTEGER NOT NULL,
    construct_id INTEGER NOT NULL,
    rt_protocol TEXT NOT NULL,
    probe TEXT NOT NULL,
    probe_concentration REAL NOT NULL,
    temperature REAL NOT NULL,
    replicate INTEGER NOT NULL,
    FOREIGN KEY(rg_id) REFERENCES probe_reaction_groups(rg_id),
    FOREIGN KEY(buffer_id) REFERENCES meta_buffers(id),
    FOREIGN KEY(construct_id) REFERENCES meta_constructs(id),
    UNIQUE(tg_id, rg_id)
);
"""

CREATE_PROBE_FMOD_RUNS = """
CREATE TABLE IF NOT EXISTS probe_fmod_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    software_name TEXT NOT NULL,
    software_version TEXT,
    run_args TEXT,
    run_datetime TEXT NOT NULL,
    output_dir TEXT NOT NULL UNIQUE,
    s_id INTEGER NOT NULL,
    FOREIGN KEY(s_id) REFERENCES sequencing_samples(id),
    UNIQUE(software_name, software_version, run_datetime, output_dir, s_id)
);
"""

CREATE_PROBE_FMOD_VALUES = """
CREATE TABLE IF NOT EXISTS probe_fmod_values (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nt_id INTEGER NOT NULL,
    fmod_run_id INTEGER NOT NULL,
    rxn_id INTEGER NOT NULL,
    valtype TEXT NOT NULL,
    fmod_val REAL,
    read_depth INTEGER NOT NULL,
    outlier INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY(nt_id) REFERENCES meta_nucleotides(id),
    FOREIGN KEY(fmod_run_id) REFERENCES probe_fmod_runs(id),
    FOREIGN KEY(rxn_id) REFERENCES probe_reactions(id)
);
"""

CREATE_PROBE_TC_FIT_RUNS = """
CREATE TABLE IF NOT EXISTS probe_tc_fit_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fit_kind TEXT NOT NULL,
    fmod_run_id INTEGER,
    rg_id INTEGER,
    nt_id INTEGER,
    valtype TEXT,
    model TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY(fmod_run_id) REFERENCES probe_fmod_runs(id),
    FOREIGN KEY(rg_id) REFERENCES probe_reaction_groups(rg_id),
    FOREIGN KEY(nt_id) REFERENCES meta_nucleotides(id)
);
"""

CREATE_PROBE_TC_FIT_PARAMS = """
CREATE TABLE IF NOT EXISTS probe_tc_fit_params (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fit_run_id INTEGER NOT NULL,
    param_name TEXT NOT NULL,
    param_numeric REAL,
    param_text TEXT,
    units TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY(fit_run_id) REFERENCES probe_tc_fit_runs(id) ON DELETE CASCADE,
    UNIQUE(fit_run_id, param_name)
);
"""

# ---------------------------------------------------------------------------
# Temperature gradient fits
# ---------------------------------------------------------------------------

CREATE_TEMPGRAD_FIT_RUNS = """
CREATE TABLE IF NOT EXISTS tempgrad_fit_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fit_kind TEXT NOT NULL,
    task_id INTEGER,
    scope_kind TEXT,
    scope_id INTEGER,
    data_source TEXT,
    target_label TEXT,
    rg_id INTEGER,
    tg_id INTEGER,
    nt_id INTEGER,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY(task_id) REFERENCES core_tasks(id) ON DELETE SET NULL
);
"""

CREATE_TEMPGRAD_FIT_PARAMS = """
CREATE TABLE IF NOT EXISTS tempgrad_fit_params (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fit_run_id INTEGER NOT NULL,
    param_name TEXT NOT NULL,
    param_numeric REAL,
    param_text TEXT,
    units TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY(fit_run_id) REFERENCES tempgrad_fit_runs(id) ON DELETE CASCADE,
    UNIQUE(fit_run_id, param_name)
);
"""

# ---------------------------------------------------------------------------
# NMR experiments
# ---------------------------------------------------------------------------

CREATE_NMR_REACTIONS = """
CREATE TABLE IF NOT EXISTS nmr_reactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reaction_type TEXT NOT NULL,
    temperature REAL NOT NULL,
    replicate INTEGER NOT NULL,
    num_scans INTEGER NOT NULL,
    time_per_read REAL NOT NULL,
    total_kinetic_reads INTEGER NOT NULL,
    total_kinetic_time INTEGER NOT NULL,
    probe TEXT NOT NULL,
    probe_conc REAL NOT NULL,
    probe_solvent TEXT NOT NULL,
    substrate TEXT NOT NULL,
    substrate_conc REAL NOT NULL,
    buffer_id INTEGER NOT NULL,
    nmr_machine TEXT NOT NULL,
    kinetic_data_dir TEXT NOT NULL UNIQUE,
    mnova_analysis_dir TEXT,
    raw_fid_dir TEXT,
    FOREIGN KEY(buffer_id) REFERENCES meta_buffers(id)
);
"""

CREATE_NMR_TRACE_FILES = """
CREATE TABLE IF NOT EXISTS nmr_trace_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nmr_reaction_id INTEGER NOT NULL,
    role TEXT NOT NULL,
    path TEXT NOT NULL,
    species TEXT,
    checksum TEXT,
    task_id INTEGER,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY(nmr_reaction_id) REFERENCES nmr_reactions(id) ON DELETE CASCADE,
    FOREIGN KEY(task_id) REFERENCES core_tasks(id) ON DELETE SET NULL,
    UNIQUE(nmr_reaction_id, role)
);
"""

CREATE_NMR_FIT_RUNS = """
CREATE TABLE IF NOT EXISTS nmr_fit_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id INTEGER NOT NULL,
    nmr_reaction_id INTEGER NOT NULL,
    plugin TEXT NOT NULL,
    model TEXT,
    species TEXT,
    params_json TEXT,
    status TEXT NOT NULL DEFAULT 'running',
    message TEXT,
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    finished_at TEXT,
    FOREIGN KEY(task_id) REFERENCES core_tasks(id) ON DELETE CASCADE,
    FOREIGN KEY(nmr_reaction_id) REFERENCES nmr_reactions(id) ON DELETE CASCADE
);
"""

CREATE_NMR_FIT_PARAMS = """
CREATE TABLE IF NOT EXISTS nmr_fit_params (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fit_run_id INTEGER NOT NULL,
    param_name TEXT NOT NULL,
    param_numeric REAL,
    param_text TEXT,
    units TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY(fit_run_id) REFERENCES nmr_fit_runs(id) ON DELETE CASCADE,
    UNIQUE(fit_run_id, param_name)
);
"""

# ---------------------------------------------------------------------------
# Core task tables
# ---------------------------------------------------------------------------

CREATE_CORE_TASKS = """
CREATE TABLE IF NOT EXISTS core_tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_name TEXT NOT NULL,
    scope_kind TEXT NOT NULL,
    scope_id INTEGER,
    output_dir TEXT NOT NULL,
    label TEXT NOT NULL,
    backend TEXT NOT NULL DEFAULT 'local',
    cache_key TEXT,
    tool TEXT,
    tool_version TEXT,
    config_hash TEXT,
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    ended_at TEXT,
    state TEXT NOT NULL DEFAULT 'pending',
    message TEXT
);
"""

CREATE_CORE_TASK_ATTEMPTS = """
CREATE TABLE IF NOT EXISTS core_task_attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id INTEGER NOT NULL,
    try_index INTEGER NOT NULL,
    command TEXT,
    resources_json TEXT,
    exit_code INTEGER,
    log_path TEXT,
    stderr_tail TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY(task_id) REFERENCES core_tasks(id) ON DELETE CASCADE,
    UNIQUE(task_id, try_index)
);
"""

CREATE_CORE_ARTIFACTS = """
CREATE TABLE IF NOT EXISTS core_artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id INTEGER NOT NULL,
    kind TEXT NOT NULL,
    key TEXT,
    path TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY(task_id) REFERENCES core_tasks(id) ON DELETE CASCADE,
    UNIQUE(task_id, kind, key)
);
"""

CREATE_CORE_TASK_SCOPE_MEMBERS = """
CREATE TABLE IF NOT EXISTS core_task_scope_members (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id INTEGER NOT NULL,
    member_kind TEXT NOT NULL,
    member_id INTEGER,
    member_label TEXT,
    extra_json TEXT,
    FOREIGN KEY(task_id) REFERENCES core_tasks(id) ON DELETE CASCADE
);
"""

# ---------------------------------------------------------------------------
# Indexes
# ---------------------------------------------------------------------------

CREATE_INDEX_CORE_TASKS_LABEL = """
CREATE INDEX IF NOT EXISTS idx_core_tasks_label ON core_tasks (label);
"""

CREATE_INDEX_CORE_TASKS_SCOPE = """
CREATE INDEX IF NOT EXISTS idx_core_tasks_scope ON core_tasks (scope_kind, scope_id);
"""

CREATE_INDEX_CORE_TASK_ATTEMPTS = """
CREATE INDEX IF NOT EXISTS idx_core_task_attempts_task ON core_task_attempts (task_id, try_index);
"""

CREATE_INDEX_CORE_ARTIFACTS = """
CREATE INDEX IF NOT EXISTS idx_core_artifacts_task ON core_artifacts (task_id, kind);
"""

CREATE_INDEX_CORE_TASK_SCOPE_MEMBERS = """
CREATE INDEX IF NOT EXISTS idx_core_task_scope_members_task ON core_task_scope_members (task_id);
"""

CREATE_INDEX_PROBE_REACTION_GROUPS = """
CREATE INDEX IF NOT EXISTS idx_probe_reaction_groups_label ON probe_reaction_groups (rg_label);
"""

CREATE_INDEX_DERIVED_CHILD = """
CREATE INDEX IF NOT EXISTS idx_derived_samples_child ON sequencing_derived_samples (child_name);
"""

CREATE_INDEX_DERIVED_PARENT = """
CREATE INDEX IF NOT EXISTS idx_derived_samples_parent ON sequencing_derived_samples (parent_sample_id);
"""

CREATE_INDEX_PROBE_TC_PARAMS = """
CREATE INDEX IF NOT EXISTS idx_probe_tc_params_name ON probe_tc_fit_params (param_name);
"""

CREATE_INDEX_NMR_FIT_PARAMS = """
CREATE INDEX IF NOT EXISTS idx_nmr_fit_params_name ON nmr_fit_params (param_name);
"""

CREATE_INDEX_TEMPGRAD_FIT_PARAMS = """
CREATE INDEX IF NOT EXISTS idx_tempgrad_fit_params_name ON tempgrad_fit_params (param_name);
"""

# ---------------------------------------------------------------------------
# Aggregate table/index lists
# ---------------------------------------------------------------------------

ALL_TABLES = [
    CREATE_META_BUFFERS,
    CREATE_META_CONSTRUCTS,
    CREATE_META_NUCLEOTIDES,
    CREATE_SEQUENCING_RUNS,
    CREATE_SEQUENCING_SAMPLES,
    CREATE_SEQUENCING_DERIVED_SAMPLES,
    CREATE_PROBE_REACTION_GROUPS,
    CREATE_PROBE_REACTIONS,
    CREATE_PROBE_TEMPGRAD_GROUPS,
    CREATE_PROBE_FMOD_RUNS,
    CREATE_PROBE_FMOD_VALUES,
    CREATE_PROBE_TC_FIT_RUNS,
    CREATE_PROBE_TC_FIT_PARAMS,
    CREATE_NMR_REACTIONS,
    CREATE_NMR_TRACE_FILES,
    CREATE_NMR_FIT_RUNS,
    CREATE_NMR_FIT_PARAMS,
    CREATE_TEMPGRAD_FIT_RUNS,
    CREATE_TEMPGRAD_FIT_PARAMS,
    CREATE_CORE_TASKS,
    CREATE_CORE_TASK_ATTEMPTS,
    CREATE_CORE_ARTIFACTS,
    CREATE_CORE_TASK_SCOPE_MEMBERS,
]

ALL_INDEXES = [
    CREATE_INDEX_CORE_TASKS_LABEL,
    CREATE_INDEX_CORE_TASKS_SCOPE,
    CREATE_INDEX_CORE_TASK_ATTEMPTS,
    CREATE_INDEX_CORE_ARTIFACTS,
    CREATE_INDEX_CORE_TASK_SCOPE_MEMBERS,
    CREATE_INDEX_PROBE_REACTION_GROUPS,
    CREATE_INDEX_DERIVED_CHILD,
    CREATE_INDEX_DERIVED_PARENT,
    CREATE_INDEX_PROBE_TC_PARAMS,
    CREATE_INDEX_NMR_FIT_PARAMS,
    CREATE_INDEX_TEMPGRAD_FIT_PARAMS,
]


# Views
VIEW_DEFINITIONS = [
    (
        "probe_tc_arrhenius_view",
        """
        CREATE VIEW probe_tc_arrhenius_view AS
        SELECT
            r.id              AS fit_run_id,
            r.rg_id,
            r.nt_id,
            r.fit_kind,
            rg.rg_label,
            pr.temperature,
            pr.probe,
            pr.probe_conc,
            pr.rt_protocol,
            pr.buffer_id,
            pr.construct_id,
            mc.disp_name      AS construct_name,
            mb.name           AS buffer_name,
            mn.site           AS nt_site,
            mn.base           AS nt_base,
            k.param_numeric   AS kobs,
            lk.param_numeric  AS log_kobs,
            kd.param_numeric  AS log_kdeg,
            kobs_err.param_numeric AS log_kobs_err,
            kd_err.param_numeric   AS log_kdeg_err,
            meta.param_text   AS metadata_json
        FROM probe_tc_fit_runs r
        JOIN probe_tc_fit_params k
             ON k.fit_run_id = r.id AND k.param_name = 'kobs'
        LEFT JOIN probe_tc_fit_params lk
             ON lk.fit_run_id = r.id AND lk.param_name = 'log_kobs'
        LEFT JOIN probe_tc_fit_params kd
             ON kd.fit_run_id = r.id AND kd.param_name = 'log_kdeg'
        LEFT JOIN probe_tc_fit_params kobs_err
             ON kobs_err.fit_run_id = r.id AND kobs_err.param_name = 'log_kobs_err'
        LEFT JOIN probe_tc_fit_params kd_err
             ON kd_err.fit_run_id = r.id AND kd_err.param_name = 'log_kdeg_err'
        LEFT JOIN probe_tc_fit_params meta
             ON meta.fit_run_id = r.id AND meta.param_name = 'metadata'
        JOIN (
            SELECT
                rg_id,
                MAX(temperature)         AS temperature,
                MAX(probe)               AS probe,
                MAX(probe_concentration) AS probe_conc,
                MAX(rt_protocol)         AS rt_protocol,
                MAX(buffer_id)           AS buffer_id,
                MAX(construct_id)        AS construct_id
            FROM probe_reactions
            GROUP BY rg_id
        ) pr ON pr.rg_id = r.rg_id
        JOIN probe_reaction_groups rg ON rg.rg_id = r.rg_id
        LEFT JOIN meta_constructs  mc ON mc.id = pr.construct_id
        LEFT JOIN meta_buffers     mb ON mb.id = pr.buffer_id
        LEFT JOIN meta_nucleotides mn ON mn.id = r.nt_id;
        """.strip(),
    ),
]
