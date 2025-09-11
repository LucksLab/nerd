#!/usr/bin/env python3

import argparse
import logging
from pathlib import Path
from nerd.main import run_command
from nerd.utils.logging import setup_logging, get_logger
log = get_logger(__name__)
def build_parser():
    parser = argparse.ArgumentParser(
        prog="nerd",
        description="Nucleotide energetics from reactivity data"
    )

    # Global logging flags
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose (DEBUG) logging"
    )
    parser.add_argument(
        "--log-file", type=str, default=None,
        help="Path to log file (enables file logging when provided)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- nerd import ---
    import_parser = subparsers.add_parser("import", help="Import samples or data into SQLite3 DB")
    import_parser.add_argument("source", choices=["csv", "shapemapper", "nmr"], help="Type of import")
    import_parser.add_argument("input", help="Path to input file or directory")

    # --- nerd degradation ---
    deg_parser = subparsers.add_parser("degradation", help="Fit exponential degradation curve from NMR")
    deg_parser.add_argument("input", help="CSV file with time vs signal data")
    deg_parser.add_argument("--reaction-id", help="Optional reaction ID to associate with DB entry")
    deg_parser.add_argument("--db", help="Path to SQLite3 DB")

    # --- nerd adduction ---
    add_parser = subparsers.add_parser("adduction", help="Fit adduction timecourse")
    add_parser.add_argument("input", help="CSV file with time vs signal data")
    add_parser.add_argument("--reaction-id", help="Optional reaction ID to associate with DB entry")
    add_parser.add_argument("--db", help="Path to SQLite3 DB")

    # --- nerd arrhenius ---
    arr_parser = subparsers.add_parser("arrhenius", help="Fit Arrhenius plot from k vs T")
    arr_parser.add_argument("input", help="CSV file with temperature and k values")
    arr_parser.add_argument("--reaction-type", default="degradation")
    arr_parser.add_argument("--data-source", default="nmr")
    arr_parser.add_argument("--db", help="Path to SQLite3 DB")

    # --- nerd timecourse ---
    tc_parser = subparsers.add_parser("timecourse", help="Fit per-nucleotide k_obs from probing time-course")
    tc_parser.add_argument("reaction_group_id", help="Reaction group identifier")
    tc_parser.add_argument("--db", help="Path to SQLite3 DB")

    # --- nerd calc_energy ---
    energy_parser = subparsers.add_parser("calc_energy", help="Calculate K or melt thermodynamics")
    energy_parser.add_argument("--mode", choices=["2state", "singleK"], required=True)
    energy_parser.add_argument("input", nargs="+", help="For 2state: CSV file. For singleK: k_obs k_add [k_deg]")

    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()
    # Initialize global logging once
    level = logging.DEBUG if getattr(args, "verbose", False) else logging.INFO
    if getattr(args, "log_file", None):
        setup_logging(level=level, log_to_file=True, log_file=Path(args.log_file))
    else:
        setup_logging(level=level, log_to_file=False)
    try:
        run_command(args)
    except Exception as e:
        log.exception("Fatal error: %s", e)
        raise