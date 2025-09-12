# nerd/main.py

from nerd.importers import fmod_calc, probing_sample, nmr
from nerd.kinetics import degradation, adduction, arrhenius, timecourse
from nerd.energy import meltfit, calc_K
from rich.console import Console

console = Console()

# test change
def run_command(args):
    """
    Dispatch CLI command to the appropriate module.
    """
    if args.command == "import":
        if args.source == "csv":
            probing_sample.run(args.input, db_path=None)
        elif args.source == "shapemapper":
            fmod_calc.run(args.input, db_path=None)
        elif args.source == "nmr":
            nmr.run(args.input, db_path=None)

    elif args.command == "degradation":
        degradation.run(
            csv_path=args.input,
            reaction_id=args.reaction_id,
            db_path=args.db
        )

    elif args.command == "adduction":
        adduction.run(
            csv_path=args.input,
            reaction_id=args.reaction_id,
            db_path=args.db
        )

    elif args.command == "arrhenius":
        arrhenius.run(
            csv_path=args.input,
            reaction_type=args.reaction_type,
            data_source=args.data_source,
            db_path=args.db
        )

    elif args.command == "timecourse":
        timecourse.run(
            reaction_group_id=args.reaction_group_id,
            db_path=args.db
        )

    elif args.command == "calc_energy":
        if args.mode == "2state":
            meltfit.run(args.input[0])
        elif args.mode == "singleK":
            # Expected args: k_obs k_add [k_deg]
            try:
                k_obs = float(args.input[0])
                k_add = float(args.input[1])
                k_deg = float(args.input[2]) if len(args.input) > 2 else 0.0
                calc_K.run([str(k_obs), str(k_add), str(k_deg)])
            except Exception as e:
                console.print(f"[red]Error parsing K calculation input:[/red] {e}")
        else:
            console.print(f"[red]Unknown mode:[/red] {args.mode}")

    else:
        console.print(f"[red]Unknown command:[/red] {args.command}")