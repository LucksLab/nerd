# nerd/importers/shapemapper.py

from pathlib import Path
from rich.console import Console
from nerd.db.io import connect_db, init_db  # You’ll define insert function later

console = Console()


def run(shapemapper_dir: str, db_path: str = None):
    """
    Import metadata from a SHAPEMapper output directory.
    For now, this assumes the structure: <shapemapper_dir>/output/<sample>/
    """
    base_path = Path(shapemapper_dir)
    if not base_path.exists():
        console.print(f"[red]Error:[/red] SHAPEMapper directory not found: {base_path}")
        return

    output_dir = base_path / "output"
    if not output_dir.exists():
        console.print(f"[red]Error:[/red] No 'output/' folder found in {base_path}")
        return

    conn = connect_db(db_path)
    init_db(conn)

    sample_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
    if not sample_dirs:
        console.print("[yellow]Warning:[/yellow] No sample folders found in output/")
        return

    count = 0
    for sample_path in sample_dirs:
        sample_name = sample_path.name
        profile_file = sample_path / f"{sample_name}.profile.txt"
        mut_file = sample_path / f"{sample_name}.mutation_rates.txt"

        if not profile_file.exists():
            console.print(f"[yellow]Skipping:[/yellow] Missing {profile_file.name}")
            continue

        # You can collect metadata here for later insertion
        record = {
            "sample_name": sample_name,
            "profile_path": str(profile_file.resolve()),
            "mutation_path": str(mut_file.resolve()) if mut_file.exists() else None,
            "shapemapper_dir": str(base_path.resolve())
            # Add more metadata fields here as needed
        }

        # Placeholder: insert into db (implement this function)
        # insert_shapemapper_run(conn, record)
        console.print(f"[green]✓ Found SHAPEMapper output for sample:[/green] {sample_name}")
        count += 1

    console.print(f"[green]✓ Imported {count} SHAPEMapper sample outputs[/green]")