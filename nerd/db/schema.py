# nerd/db/schema.py
"""
This module defines the SQLite schema used by the nerd CLI tool.
Each table creation SQL is written explicitly for clarity and maintainability.
"""

# === Table: probing_reactions ===
CREATE_PROBING_REACTIONS = """ 
CREATE TABLE IF NOT EXISTS probing_reactions (
    id INTEGER UNIQUE,                           -- Primary key for this table
    temperature INTEGER NOT NULL,                -- Reaction temperature (Celsius)
    replicate INTEGER NOT NULL,                  -- Replicate number
    reaction_time INTEGER NOT NULL,              -- Reaction time (seconds)
    probe_concentration REAL NOT NULL,           -- Concentration of probe (M)
    probe TEXT NOT NULL,                         -- Name of chemical probe (e.g. "dms")
    RT TEXT NOT NULL,                            -- Reverse transcription enzyme or protocol ("MRT", "SSIII", etc.)
    done_by TEXT NOT NULL,                       -- Initials of person who performed the reaction
    treated INTEGER NOT NULL,                    -- 1 if treated, 0 if control
    buffer_id INTEGER NOT NULL,                  -- Foreign key to buffers table
    construct_id INTEGER NOT NULL,               -- Foreign key to constructs table
    rg_id INTEGER NOT NULL,                      -- Foreign key to reaction_groups table
    s_id INTEGER NOT NULL,                       -- Foreign key to sequencing_samples table
    PRIMARY KEY(id AUTOINCREMENT),
    FOREIGN KEY(buffer_id) REFERENCES buffers(id),
    FOREIGN KEY(construct_id) REFERENCES constructs(id),
    FOREIGN KEY(rg_id) REFERENCES reaction_groups(rg_id),
    FOREIGN KEY(s_id) REFERENCES sequencing_samples(id)
);
"""

# === Table: reaction_groups ===
CREATE_REACTION_GROUPS = """ 
CREATE TABLE IF NOT EXISTS reaction_groups (
    id INTEGER NOT NULL UNIQUE,                  -- Primary key for this table
    rg_id INTEGER NOT NULL,                      -- Reaction group identifier
    rxn_id INTEGER NOT NULL,                     -- Foreign key to probing_reactions table
    PRIMARY KEY(id AUTOINCREMENT),
    FOREIGN KEY(rxn_id) REFERENCES probing_reactions(rxn_id),
    UNIQUE(rg_id, rxn_id)                               -- Ensure unique reaction group IDs
);
"""

# === Table: constructs ===
CREATE_CONSTRUCTS = """ 
CREATE TABLE IF NOT EXISTS constructs (
    id INTEGER NOT NULL UNIQUE,                  -- Primary key for this table
    family TEXT NOT NULL,                        -- Family or group the construct belongs to
    name TEXT NOT NULL,                          -- Name of the construct
    version TEXT NOT NULL,                       -- Version identifier for the construct
    sequence TEXT NOT NULL,                      -- Nucleotide sequence of the construct
    disp_name TEXT NOT NULL,                     -- Display name for the construct
    PRIMARY KEY(id AUTOINCREMENT),
    UNIQUE(family, name, version, sequence, disp_name)      -- Ensure unique constructs by family, name, and version
);
"""

# === Table: buffers ===
CREATE_BUFFERS = """ 
CREATE TABLE IF NOT EXISTS buffers (
    id INTEGER NOT NULL UNIQUE,                  -- Primary key for this table
    name TEXT NOT NULL,                          -- Name of the buffer (e.g. "Tris", "HEPES")
    pH REAL NOT NULL,                            -- pH of the buffer
    composition TEXT NOT NULL,                   -- Buffer composition (e.g. "50 mM Tris, 100 mM NaCl")
    disp_name TEXT NOT NULL,                     -- Display name for the buffer
    PRIMARY KEY(id AUTOINCREMENT),
    UNIQUE(name, pH, composition, disp_name)     -- Ensure unique buffers by name, pH, composition, and display name
);
"""

# === Table: nucleotides ===
CREATE_NUCLEOTIDES = """ 
CREATE TABLE IF NOT EXISTS nucleotides (
    id INTEGER NOT NULL UNIQUE,                  -- Primary key for this table
    site INTEGER NOT NULL,                       -- Nucleotide position (1-based)
    base TEXT NOT NULL,                          -- Nucleotide base (A, C, G, U)
    base_region TEXT NOT NULL,                   -- Region annotation (e.g. "loop", "stem")
    construct_id INTEGER NOT NULL,               -- Foreign key to constructs table
    PRIMARY KEY(id AUTOINCREMENT),
    FOREIGN KEY(construct_id) REFERENCES constructs(id) ON DELETE CASCADE,
    UNIQUE(site, base, base_region, construct_id)  -- Ensure unique nucleotides by site, base, region, and construct
);
"""

# === Table: sequencing_runs ===
CREATE_SEQUENCING_RUNS = """ 
CREATE TABLE IF NOT EXISTS sequencing_runs (
    id INTEGER NOT NULL UNIQUE,                  -- Primary key for this table
    run_name TEXT NOT NULL,                      -- Name of the sequencing run
    date TEXT NOT NULL,                          -- Date of the sequencing run
    sequencer TEXT NOT NULL,                     -- Sequencer used (e.g. "Illumina", "Nanopore")
    run_manager TEXT NOT NULL,                   -- Person responsible for the run
    PRIMARY KEY(id AUTOINCREMENT),
    UNIQUE(run_name, date, sequencer, run_manager)  -- Ensure unique runs by name, date, sequencer, and manager
);
"""

# === Table: sequencing_samples ===
CREATE_SEQUENCING_SAMPLES = """ 
CREATE TABLE IF NOT EXISTS sequencing_samples (
    id INTEGER NOT NULL UNIQUE,                  -- Primary key for this table
    seqrun_id INTEGER NOT NULL,                  -- Foreign key to sequencing_runs table
    sample_name TEXT NOT NULL,                   -- Name of the sequencing sample
    fq_dir TEXT NOT NULL,                        -- Directory containing fastq files
    PRIMARY KEY(id AUTOINCREMENT),
    FOREIGN KEY(seqrun_id) REFERENCES sequencing_runs(id),
    UNIQUE(seqrun_id, sample_name, fq_dir)       -- Ensure unique samples by run ID, name, and directory
);
"""

# === Table: fmod_calc_runs ===
CREATE_FMOD_CALC_RUNS = """ 
CREATE TABLE IF NOT EXISTS fmod_calc_runs (
    id INTEGER NOT NULL UNIQUE,                      -- Primary key for this table
    software_name TEXT NOT NULL,                     -- Name of the software used for calculation
    software_version INTEGER NOT NULL,               -- Version of the software
    run_args INTEGER NOT NULL,                       -- Arguments used for the run (should likely be TEXT)
    run_datetime TEXT NOT NULL,                      -- Date and time of the run
    output_dir TEXT NOT NULL UNIQUE,                 -- Directory where output files are stored
    s_id INTEGER NOT NULL,                           -- Foreign key to sequencing_samples table
    PRIMARY KEY(id AUTOINCREMENT),
    FOREIGN KEY(s_id) REFERENCES sequencing_samples(id),
    UNIQUE(software_name, software_version, run_args, run_datetime, output_dir, s_id)  -- Ensure unique calculation runs
);
"""

# === Table: fmod_vals ===
CREATE_FMOD_VALS = """ 
CREATE TABLE IF NOT EXISTS fmod_vals (
    id INTEGER NOT NULL UNIQUE,                      -- Primary key for this table
    nt_id INTEGER NOT NULL,                          -- Foreign key to nucleotides table
    fmod_calc_run_id INTEGER NOT NULL,               -- Foreign key to fmod_calc_runs table
    fmod_val REAL,                                   -- Fraction modified value (may be NULL)
    valtype TEXT NOT NULL,                           -- Type of value (e.g. "raw", "normalized")
    read_depth INTEGER NOT NULL,                     -- Read depth at this nucleotide
    rxn_id INTEGER NOT NULL,                         -- Foreign key to probing_reactions table
    PRIMARY KEY(id AUTOINCREMENT),
    FOREIGN KEY(nt_id) REFERENCES nucleotides(id),
    FOREIGN KEY(fmod_calc_run_id) REFERENCES fmod_calc_runs(id),
    FOREIGN KEY(rxn_id) REFERENCES probing_reactions(id)
);
"""

# === Table: free_tc_fits ===
CREATE_FREE_TC_FITS = """ 
CREATE TABLE IF NOT EXISTS free_tc_fits (
    id INTEGER NOT NULL UNIQUE,                      -- Primary key for this table
    rg_id INTEGER NOT NULL,                          -- Foreign key to reaction_groups table
    nt_id INTEGER NOT NULL,                          -- Foreign key to nucleotides table
    kobs_val REAL NOT NULL,                          -- Observed rate constant value
    kobs_err REAL NOT NULL,                          -- Error in observed rate constant
    kdeg_val REAL NOT NULL,                          -- Degradation rate constant value
    kdeg_err REAL NOT NULL,                          -- Error in degradation rate constant
    fmod0 REAL NOT NULL,                             -- Initial fraction modified
    fmod0_err REAL NOT NULL,                         -- Error in initial fraction modified
    r2 REAL NOT NULL,                                -- R-squared value of the fit
    chisq REAL NOT NULL,                             -- Chi-squared value of the fit
    time_min REAL NOT NULL,                          -- Minimum time (in seconds) used in fit
    time_max REAL NOT NULL,                          -- Maximum time (in seconds) used in fit
    PRIMARY KEY(id AUTOINCREMENT),
    FOREIGN KEY(rg_id) REFERENCES reaction_groups(rg_id),
    FOREIGN KEY(nt_id) REFERENCES nucleotides(id),
    UNIQUE(rg_id, nt_id)                             -- Ensure unique fits per reaction group and nucleotide
);
"""

# === Table: global_tc_deg_fits ===
CREATE_GLOBAL_TC_DEG_FITS = """ 
CREATE TABLE IF NOT EXISTS global_tc_deg_fits (
    id INTEGER NOT NULL UNIQUE,                      -- Primary key for this table
    rg_id INTEGER NOT NULL,                          -- Foreign key to reaction_groups table
    nt_id INTEGER NOT NULL,                          -- Foreign key to nucleotides table
    kobs_val REAL NOT NULL,                          -- Observed rate constant value (global fit)
    kobs_err REAL NOT NULL,                          -- Error in observed rate constant (global fit)
    kdeg_val REAL NOT NULL,                          -- Degradation rate constant value (global fit)
    kdeg_err REAL NOT NULL,                          -- Error in degradation rate constant (global fit)
    fmod0 REAL NOT NULL,                             -- Initial fraction modified (global fit)
    fmod_err REAL NOT NULL,                          -- Error in initial fraction modified (global fit)
    r2 REAL NOT NULL,                                -- R-squared value of the fit
    chisq REAL NOT NULL,                             -- Chi-squared value of the fit
    time_min REAL NOT NULL,                          -- Minimum time (in seconds) used in fit
    time_max REAL NOT NULL,                          -- Maximum time (in seconds) used in fit
    PRIMARY KEY(id AUTOINCREMENT),
    FOREIGN KEY(rg_id) REFERENCES reaction_groups(rg_id),
    FOREIGN KEY(nt_id) REFERENCES nucleotides(id),
    UNIQUE(rg_id, nt_id)                             -- Ensure unique fits per reaction group and nucleotide
);
"""

# === Table: probe_melt_fits ===
CREATE_PROBE_MELT_FITS = """
CREATE TABLE IF NOT EXISTS probe_melt_fits (
    id INTEGER NOT NULL UNIQUE,                      -- Primary key for this table
    rg_id INTEGER NOT NULL,                          -- Foreign key to reaction_groups table
    nt_id INTEGER NOT NULL,                          -- Foreign key to nucleotides table
    a REAL NOT NULL,                                 -- Slope of the unfolded state
    b REAL NOT NULL,                                 -- Y-intercept of the unfolded state
    c REAL NOT NULL,                                 -- Slope of the folded state
    d REAL NOT NULL,                                 -- Y-intercept of the folded state
    f REAL NOT NULL,                                 -- Energy of the transition state
    g REAL NOT NULL,                                 -- Temperature of the transition state
    r2 REAL NOT NULL,                                -- R-squared value of the fit
    chisq REAL NOT NULL,                             -- Chi-squared value of the fit
    PRIMARY KEY(id AUTOINCREMENT),
    FOREIGN KEY(rg_id) REFERENCES reaction_groups(rg_id),
    FOREIGN KEY(nt_id) REFERENCES nucleotides(id)
);
"""

# === Table: nmr_reactions ===
CREATE_NMR_REACTIONS = """
CREATE TABLE IF NOT EXISTS nmr_reactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reaction_type TEXT NOT NULL,                 -- "deg", "add"
    temperature REAL NOT NULL,                   -- Temperature in Celsius
    replicate INTEGER NOT NULL,                  -- Replicate number
    num_scans INTEGER NOT NULL,                  -- Number of scans per read
    time_per_read REAL NOT NULL,                 -- Time per NMR read in mins
    total_kinetic_reads INTEGER NOT NULL,        -- Total number of kinetic reads
    total_kinetic_time INTEGER NOT NULL,         -- Total time for kinetic reads in seconds
    probe TEXT NOT NULL,                         -- Chemical probe used (e.g. "dms")
    probe_conc REAL NOT NULL,                    -- Concentration of the probe in M
    probe_solvent TEXT NOT NULL,                 -- Solvent used for the probe (e.g. "etoh", "dmso")
    substrate TEXT NOT NULL,                     -- Substrate used (e.g. "ATP", "GTP", "none" if degradation)
    substrate_conc REAL NOT NULL,                -- Concentration of the substrate in M (0 if degradation)
    buffer TEXT NOT NULL,                        -- Buffer used for the reaction
    nmr_machine TEXT NOT NULL,                   -- NMR machine used for the experiment
    kinetic_data_dir TEXT NOT NULL,              -- Directory containing kinetic data csv file
    mnova_analysis_dir TEXT,                     -- Directory containing MNova analysis files
    raw_fid_dir TEXT,                             -- Directory containing raw FID files
    UNIQUE(kinetic_data_dir)                   -- Ensure unique directory for kinetic data
);
"""


# === Table: nmr_kinetic_rates ===
CREATE_NMR_KINETIC_RATES = """
CREATE TABLE IF NOT EXISTS nmr_kinetic_rates (
    id INTEGER NOT NULL UNIQUE,                  -- Primary key for this table
    nmr_reaction_id INTEGER,                      -- Foreign key to nmr_reactions table
    model TEXT NOT NULL,                         -- Model used for fitting (e.g. "exponential", "ode")
    k_value REAL NOT NULL,                       -- Fitted kinetic rate value
    k_error REAL NOT NULL,                       -- Fitted kinetic rate std error
    r2 REAL NOT NULL,                            -- R-squared value of the fit
    chisq REAL NOT NULL,                         -- Chi-squared value of the fit
    species TEXT NOT NULL,                       -- Species for which the rate is calculated (e.g. "dms", "atp-c8", etc.)
    PRIMARY KEY(id AUTOINCREMENT),
    FOREIGN KEY(nmr_reaction_id) REFERENCES nmr_reactions(id),
    UNIQUE(nmr_reaction_id, species)            -- Ensure unique rates per reaction and species
);
"""

# === Table: arrhenius_fits ===
CREATE_ARRHENIUS_FITS = """
CREATE TABLE IF NOT EXISTS arrhenius_fits (
    id INTEGER NOT NULL UNIQUE,
    reaction_type TEXT NOT NULL,                 -- "deg_nmr", "add_nmr", "rna_probe"
    data_source TEXT NOT NULL,                   -- "nmr", "probe_global", "probe_free"
    substrate TEXT NOT NULL,                     -- Substrate for which the fit is calculated (e.g. "ATP", "GTP", "none" for degradation)
    slope REAL NOT NULL,                         -- Slope of the Arrhenius fit (activation energy)
    slope_err REAL NOT NULL,                     -- Standard error of the slope
    intercept REAL NOT NULL,                     -- Intercept of the Arrhenius fit (log pre-exponential factor)
    intercept_err REAL NOT NULL,                 -- Standard error of the intercept
    r2 REAL NOT NULL,                            -- R-squared value of the fit
    model_file TEXT,                             -- optional path to serialized fit object
    PRIMARY KEY(id AUTOINCREMENT),
    UNIQUE(reaction_type, data_source, substrate) -- Ensure unique fits per reaction type and data source
);
"""

# === Aggregate all table creation statements ===
ALL_TABLES = [
    CREATE_PROBING_REACTIONS,
    CREATE_REACTION_GROUPS,
    CREATE_CONSTRUCTS,
    CREATE_BUFFERS,
    CREATE_NUCLEOTIDES,
    CREATE_SEQUENCING_RUNS,
    CREATE_SEQUENCING_SAMPLES,
    CREATE_FMOD_CALC_RUNS,
    CREATE_FMOD_VALS,
    CREATE_FREE_TC_FITS,
    CREATE_GLOBAL_TC_DEG_FITS,
    CREATE_NMR_KINETIC_RATES,
    CREATE_NMR_REACTIONS,
    CREATE_ARRHENIUS_FITS,
    CREATE_PROBE_MELT_FITS
]