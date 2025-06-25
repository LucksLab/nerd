# nerd/energy/calc_K.py

from rich.console import Console

console = Console()

def calculate_K(k_obs, k_add, k_deg=0.0, k_obs_err=None, k_add_err=None, k_deg_err=None):
    """
    Calculate equilibrium constant K = (k_obs - k_deg) / k_add
    Optionally propagate error via basic error propagation rules.
    """
    numerator = k_obs - k_deg
    if k_add == 0:
        raise ValueError("k_add cannot be zero.")

    K = numerator / k_add

    if all(x is not None for x in [k_obs_err, k_add_err, k_deg_err]):
        # Error propagation: ΔK = K * sqrt((Δnumerator/numerator)^2 + (Δkadd/kadd)^2)
        delta_num = (k_obs_err**2 + k_deg_err**2)**0.5
        rel_err_K = ((delta_num / numerator) ** 2 + (k_add_err / k_add) ** 2) ** 0.5
        K_err = abs(K) * rel_err_K
    else:
        K_err = None

    return K, K_err


def run(input_args: list[str]):
    """
    CLI entrypoint: parse values from command-line args or interactive prompt.
    """
    try:
        if len(input_args) < 2:
            console.print("[yellow]Usage:[/yellow] nerd calc_energy --mode singleK <k_obs> <k_add> [k_deg]")
            return

        k_obs = float(input_args[0])
        k_add = float(input_args[1])
        k_deg = float(input_args[2]) if len(input_args) > 2 else 0.0

        K, K_err = calculate_K(k_obs, k_add, k_deg)

        if K_err:
            console.print(f"[green]✓ K = {K:.4f} ± {K_err:.4f}[/green]")
        else:
            console.print(f"[green]✓ K = {K:.4f}[/green]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")