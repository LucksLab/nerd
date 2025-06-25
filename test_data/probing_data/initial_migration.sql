CREATE TABLE IF NOT EXISTS "reaction_groups" (
	"id" INTEGER NOT NULL UNIQUE,
	"rg_id"	INTEGER NOT NULL,
	"rxn_id"	INTEGER NOT NULL,
	PRIMARY KEY("id" AUTOINCREMENT),
	FOREIGN KEY("rxn_id") REFERENCES "probing_reactions"("rxn_id")
);
CREATE TABLE IF NOT EXISTS "probing_reactions" (
	"id"	INTEGER UNIQUE,
	"temperature"	INTEGER NOT NULL,
	"replicate"	INTEGER NOT NULL,
	"reaction_time"	INTEGER NOT NULL,
	"probe_concentration"	REAL NOT NULL,
	"probe"	TEXT NOT NULL,
	"buffer_id"	INTEGER NOT NULL,
	"construct_id"	INTEGER NOT NULL,
	"RT" TEXT NOT NULL,
	"done_by" TEXT NOT NULL,
	"treated" INTEGER NOT NULL, -- 1 if plus, 0 if minus, 2 if combined
	"rg_id" TEXT NOT NULL,
	"s_id" TEXT NOT NULL,
	PRIMARY KEY("id" AUTOINCREMENT),
	FOREIGN KEY("buffer_id") REFERENCES "buffers"("id"),
	FOREIGN KEY("construct_id") REFERENCES "constructs"("id")
	FOREIGN KEY("rg_id") REFERENCES "reaction_groups"("rg_id")
	FOREIGN KEY("s_id") REFERENCES "sequencing_samples"("id")
);
CREATE TABLE IF NOT EXISTS "constructs" (
	"id"	INTEGER NOT NULL UNIQUE,
	"family" TEXT NOT NULL,
	"name"	TEXT NOT NULL,
	"version"	TEXT NOT NULL,
	"sequence"	TEXT NOT NULL,
	"disp_name" TEXT NOT NULL,
	PRIMARY KEY("id" AUTOINCREMENT)
);
CREATE TABLE IF NOT EXISTS "buffers" (
	"id"	INTEGER NOT NULL UNIQUE,
	"name" TEXT NOT NULL,
	"pH"	REAL NOT NULL,
	"composition"	TEXT NOT NULL,
	"disp_name" TEXT NOT NULL,
	PRIMARY KEY("id" AUTOINCREMENT)
);
CREATE TABLE IF NOT EXISTS "nucleotides" (
	"id"	INTEGER NOT NULL UNIQUE,
	"site"	INTEGER NOT NULL,
	"base"	TEXT NOT NULL,
	"base_region" TEXT NOT NULL, -- 5' or 3' primer binding site, or internal ROI
	"construct_id"	INTEGER NOT NULL,
	PRIMARY KEY("id" AUTOINCREMENT),
	FOREIGN KEY("construct_id") REFERENCES "constructs"("id")
);
CREATE TABLE IF NOT EXISTS "sequencing_runs" (
	"id"	INTEGER NOT NULL UNIQUE,
	"run_name"	TEXT NOT NULL,
	"date"	TEXT NOT NULL, -- Store date as 'YYMMDD'
	"sequencer"	TEXT NOT NULL,
	"run_manager" TEXT NOT NULL,
	PRIMARY KEY("id" AUTOINCREMENT)
);
CREATE TABLE IF NOT EXISTS "sequencing_samples" (
	"id"	INTEGER NOT NULL UNIQUE,
	"seqrun_id"	INTEGER NOT NULL,
	"sample_name"	TEXT NOT NULL,
	"fq_dir"	TEXT NOT NULL, -- Must contain R1 and R2 files
	PRIMARY KEY("id" AUTOINCREMENT),
	FOREIGN KEY("seqrun_id") REFERENCES "sequencing_runs"("id")
);
CREATE TABLE IF NOT EXISTS "free_tc_fits" (
	"id"	INTEGER NOT NULL UNIQUE,
	"rg_id"	INTEGER NOT NULL,
	"nt_id" INTEGER NOT NULL,
	"kobs_val"	REAL NOT NULL,
	"kobs_err"	REAL NOT NULL,
	"kdeg_val"	REAL NOT NULL,
	"kdeg_err"	REAL NOT NULL,
	"fmod0" REAL NOT NULL,
	"fmod0_err" REAL NOT NULL,
	"r2"	REAL NOT NULL,
	"chisq"	REAL NOT NULL,
	"time_min" REAL NOT NULL,
	"time_max" REAL NOT NULL,
	PRIMARY KEY("id" AUTOINCREMENT),
	FOREIGN KEY("rg_id") REFERENCES "reaction_groups"("rg_id"),
	FOREIGN KEY("nt_id") REFERENCES "nucleotides"("id")
);
CREATE TABLE IF NOT EXISTS "global_tc_deg_fits" (
	"id"	INTEGER NOT NULL UNIQUE,
	"rg_id"	INTEGER NOT NULL,
	"nt_id" INTEGER NOT NULL,
	"kobs_val"	REAL NOT NULL,
	"kobs_err"	REAL NOT NULL,
	"kdeg_val"	REAL NOT NULL,
	"kdeg_err"	REAL NOT NULL,
	"fmod0" REAL NOT NULL,
	"fmod_err" REAL NOT NULL,
	"r2"	REAL NOT NULL,
	"chisq"	REAL NOT NULL,
	"time_min" REAL NOT NULL,
	"time_max" REAL NOT NULL,
	PRIMARY KEY("id" AUTOINCREMENT),
	FOREIGN KEY("rg_id") REFERENCES "reaction_groups"("rg_id"),
	FOREIGN KEY("nt_id") REFERENCES "nucleotides"("id")
);
CREATE TABLE IF NOT EXISTS "fmod_calc_runs" (
	"id"	INTEGER NOT NULL UNIQUE,
	"software_name"	TEXT NOT NULL,
	"software_version"	INTEGER NOT NULL,
	"run_args"	INTEGER NOT NULL,
	"output_dir"	TEXT NOT NULL,
	"s_id"	INTEGER NOT NULL,
	PRIMARY KEY("id" AUTOINCREMENT)
	FOREIGN KEY("s_id") REFERENCES "sequencing_samples"("id")
);
CREATE TABLE IF NOT EXISTS "fmod_vals" (
	"id"	INTEGER NOT NULL UNIQUE,
	"nt_id"	INTEGER NOT NULL,
	"fmod_calc_run_id"	INTEGER NOT NULL,
	"fmod_val"	REAL,
	"valtype" TEXT NOT NULL, -- 'beta' or 'modrate' or 'gamodrate'
	"read_depth"	INTEGER NOT NULL,
	"rxn_id" INTEGER NOT NULL,
	PRIMARY KEY("id" AUTOINCREMENT),
	FOREIGN KEY("nt_id") REFERENCES "nucleotides"("id"),
	FOREIGN KEY("fmod_calc_run_id") REFERENCES "fmod_calcs_runs"("id"),
	FOREIGN KEY("rxn_id") REFERENCES "probing_reactions"("id")
);

