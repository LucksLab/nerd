# nerd/energy/calc_K.py

from nerd.utils.logging import get_logger

log = get_logger(__name__)

def calculate_K(k_obs, k_add, k_deg=0.0, k_obs_err=None, k_add_err=None, k_deg_err=None):
    """
    Calculate equilibrium constant K = (k_obs - k_deg) / k_add
    Optionally propagate error via basic error propagation rules.
    """
    numerator = k_obs - k_deg
    if k_add == 0:
        raise ValueError("k_add cannot be zero.")

    K = numerator / k_add

    if (k_obs_err is not None) and (k_add_err is not None) and (k_deg_err is not None):
        # Error propagation: ΔK = K * sqrt((Δnumerator/numerator)^2 + (Δkadd/kadd)^2)
        delta_num = (k_obs_err ** 2 + k_deg_err ** 2) ** 0.5
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
            log.warning("Usage: nerd calc_energy --mode singleK <k_obs> <k_add> [k_deg]")
            return

        k_obs = float(input_args[0])
        k_add = float(input_args[1])
        k_deg = float(input_args[2]) if len(input_args) > 2 else 0.0

        K, K_err = calculate_K(k_obs, k_add, k_deg)

        if K_err is not None:
            log.info("K = %.4f ± %.4f", K, K_err)
        else:
            log.info("K = %.4f", K)

    except Exception as e:
        log.exception("Error: %s", e)