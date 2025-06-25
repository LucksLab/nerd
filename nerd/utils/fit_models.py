# nerd/utils/fit_models.py

# These are all wrappers around lmfit models

from lmfit import Model
from lmfit.models import ExponentialModel, ConstantModel, LinearModel
import numpy as np
from scipy.integrate import solve_ivp
from lmfit import Parameters, minimize

# === Exponential decay (degradation) ===
def fit_exp_decay(x, y):
    """
    Fit a single exponential decay model to the data.
    y(t) = A * exp(-k * t) + C
    where:
    """

    model = ExponentialModel()
    params = model.guess(y, x = x)
    result = model.fit(y, params, x = x)
    return result

# === Linear fit (Arrhenius) ===
def fit_linear(x, y):
    """
    Fit a linear model to the data.
    y = m * x + b
    """
    model = LinearModel()
    params = model.guess(y, x=x)
    result = model.fit(y, params, x=x)
    return result

# === ODE fit (NTP adduction) ===

def fit_ODE_probe(dms_perc, peak_perc, ntp_conc):

    # for whichever has a shorter length between dms_perc and peak_perc, use those time values
    if len(dms_perc) < len(peak_perc):
        t_obs = list(dms_perc['time'].values)
    else:
        t_obs = list(peak_perc['time'].values)

    # filter both to observed time values
    peak_perc = peak_perc[peak_perc['time'].isin(t_obs)]
    dms_perc = dms_perc[dms_perc['time'].isin(t_obs)]

    RNACONC = ntp_conc
    DMSCONC = 0.01564

    # Replace these lists with your actual data
    U_data = list(peak_perc['peak'].values * RNACONC)
    S_data = list(dms_perc['peak'].values * DMSCONC)
    M_data = list((1 - peak_perc['peak']).values * RNACONC)

    # Create y_obs as a 2D array with one row for each of U, S, and M
    y_obs = np.array([U_data, S_data, M_data])

    # Define the system of ODEs
    def system(t, y, k_add, k_deg):
        U, S, M = y
        dUdt = -k_add * U * S
        dSdt = - k_add * U * S - k_deg * S
        dMdt = k_add * U * S
        return [dUdt, dSdt, dMdt]

    # Wrapper function for solve_ivp that takes time values and parameters as input
    def solve_system(params, t, data):
        k_add = params['k_add']
        k_deg = params['k_deg']
        S_factor = params['S_factor']
        # adjust S by S_factor
        data[1] = data[1] * S_factor

        # solve starting at t = 0 and ending at the last time point in t_obs
        fill_in = np.linspace(0, t[0], 51)[:-1]
        full_t = np.concatenate((fill_in, t))
        sol = solve_ivp(system, [0, t[-1]], y0, args=(k_add, k_deg), t_eval=full_t, vectorized=True)

        # take only the values at the observed time points index 50 onwards (ignore first couple data points for equilibration)
        sol_observed = sol.y[:, 50:]

        return (sol_observed.ravel() - data)

    # Initial conditions
    U0 = RNACONC
    S0 = DMSCONC
    M0 = 0.0
    y0 = [U0, S0, M0]

    # Time values
    t = t_obs

    # Observed data
    # Replace this with your actual data
    y_obs = np.array([U_data, S_data, M_data])

    # Define parameters

    k_deg_0 = 1e-3  # initial guess for k_deg
    k_add_0 = 0.003  # initial guess for k_add
    params = Parameters()
    params.add('k_add', value = k_add_0, vary=True)
    params.add('k_deg', value = k_deg_0, vary=True)
    params.add('S_factor', value = 1.0, vary=False)

    # Fit the parameters
    out = minimize(solve_system, params, args=(t, y_obs.ravel()))

    # Calculate R-squared
    ss_res = np.sum(out.residual**2)
    ss_tot = np.sum((y_obs - np.mean(y_obs))**2)
    r_squared = 1 - (ss_res / ss_tot)

    # Calculate chi-sq
    chi_sq = np.sum(out.residual**2) / (len(t) - len(out.params))

    return out.params, r_squared, chi_sq


def melt_fit(x, a, b, c, d, f, g):
    # a: slope of the unfolded state
    # b: y-intercept of the unfolded state
    # c: slope of the folded state
    # d: y-intercept of the folded state
    # f: energy of the transition state
    # g: temperature of the transition state
    
    temp = 1 / x

    R = 0.001987
    R = 0.0083145
    k_add = np.exp((f/R)*(1/(g+273.15) - 1/(temp)))
    Q1 = 1 + k_add
    fracu = 1 / Q1
    fracf = k_add / Q1
    basef = a * x + b
    baseu = c * x + d

    final = np.log(fracu) * baseu + np.log(fracf) * basef
    final = fracu * baseu + fracf * basef
    return final

# === 4. Generic fit wrapper ===
def run_fit(model, params, x, y):
    """
    Fit a model to (x, y) data and return lmfit Result.
    """
    return model.fit(y, params, x=x)