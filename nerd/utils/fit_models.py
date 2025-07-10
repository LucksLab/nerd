# nerd/utils/fit_models.py

# These are all wrappers around lmfit models

from lmfit import Model
from lmfit.models import ExponentialModel, ConstantModel, LinearModel
import numpy as np
from scipy.integrate import solve_ivp
from lmfit import Parameters, minimize
from typing import Optional

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
def fit_linear(x, y, weights = None):
    """
    Fit a linear model to the data.
    y = m * x + b
    """
    model = LinearModel()
    params = model.guess(y, x=x)
    result = model.fit(y, params, x=x, weights=weights)
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

# === Time-course HDX model (free) ===
# def get_kdeg(temp, slope, intercept):
#     return np.exp(slope / (temp + 273.15) + intercept)

def fmod_model(x, log_kappa, log_kdeg, log_fmod_0):
    """
    Free HDX-like model for probing time-course data.

    fmod(x) = 1 - exp(-kappa * (1 - exp(-kdeg * x))) + fmod_0
    
    where:
    - kappa is the observed rate constant for the probing reaction
    - kdeg is the rate constant for the degradation reaction
    - fmod_0 is the initial fraction of deuterium incorporation
    """
    
    kappa = np.exp(log_kappa)
    kdeg = np.exp(log_kdeg)
    fmod_0 = np.exp(log_fmod_0)
    return 1 - np.exp(- (kappa) * (1 - np.exp(-kdeg * x))) + fmod_0

def fit_timecourse(time_data, fmod_data, kdeg0, constrained_kdeg = None):

    # Initialize model
    model = Model(fmod_model)

    # Initial parameter estimates
    kappa0 = -np.log(1 - max(fmod_data))
    fmod_00 = max(min(fmod_data), 1e-6)  # Avoid log(0) errors
    
    # Create parameters for the model
    params = model.make_params(
        log_kappa = np.log(kappa0), 
        log_fmod_0 = np.log(fmod_00)
    )

    # constrained kdeg if specified
    if constrained_kdeg:
        params.add('log_kdeg', value = kdeg0, vary = False)
    else:
        # this value is log already
        params.add('log_kdeg', value = np.log(kdeg0), vary = True)

    # Fit the model to the data
    result = model.fit(fmod_data, params, x = time_data)
    
    outlier = None
    # OUTLIER MECHANISM Remove outlier outside of 1.5σ and refit (CURRENTLY DISABLED)
    # outlier = np.abs(result.residual) > 1000 * np.std(result.residual)
    
    # if sum(outlier) > 0:
    #     time_data = time_data[~outlier]
    #     fmod_data = fmod_data[~outlier]
    
    #     # Initial values = values from previous fit
    #     params = model.make_params(
    #         log_kappa=result.best_values['log_kappa'], 
    #         log_kdeg=result.best_values['log_kdeg'], 
    #         log_fmod_0=result.best_values['log_fmod_0']
    #     )
        
    #     result = model.fit(fmod_data, params, x=time_data)

    return result, outlier


# === Global time-course HDX model ===

import numpy as np
from lmfit import Parameters, minimize

# Create parameters from arrays
def create_params(kappa_array, kdeg_array, fmod0_array):
    fit_params = Parameters()
    # Ensure kdeg_array contains only positive values to avoid NaN
    log_kdeg_global = np.array(kdeg_array).mean()
    # calc from NMR

    for i in range(len(kappa_array)):
        log_kappa = kappa_array[i]
        log_fmod_0 = fmod0_array[i]

        fit_params.add(f'log_kappa_{i+1}', value=log_kappa)
        fit_params.add(f'log_kdeg_{i+1}', value=log_kdeg_global)
        fit_params.add(f'log_fmod0_{i+1}', value=log_fmod_0)

        if i > 0:
            fit_params[f'log_kdeg_{i+1}'].expr = 'log_kdeg_1'

    return fit_params

# Dataset creation assumes input: list of (x_values, y_values) for each site
def create_dataset(time_data_array, fmod_data_array):
    # time_data_array: list of arrays (one per site)
    # fmod_data_array: list of arrays (one per site)
    y_dataset = np.array(fmod_data_array)

    # check all time arrays are identical
    if not all(np.array_equal(time_data_array[0], time_array) for time_array in time_data_array):
        raise ValueError("All time arrays must be identical for global fitting.")

    x_data = np.array(time_data_array[0])  # assume all time arrays are identical
    return x_data, y_dataset


# Model for each dataset
def fmod_dataset(params, i, x):
    log_kappa = params[f'log_kappa_{i+1}']
    log_kdeg = params[f'log_kdeg_{i+1}']
    log_fmod0 = params[f'log_fmod0_{i+1}']
    return fmod_model(x, log_kappa, log_kdeg, log_fmod0)

# Objective function
def objective(params, x, data):
    ndata, _ = data.shape
    resid = np.zeros_like(data)
    for i in range(ndata):
        resid[i, :] = data[i, :] - fmod_dataset(params, i, x)
    return resid.flatten()

# Global fitting entry point
def global_fit_timecourse(time_data_array, fmod_data_array, kappa_array, kdeg_array, fmod0_array) -> Optional[tuple]:

    global_params = create_params(kappa_array, kdeg_array, fmod0_array)
    x_data, y_dataset = create_dataset(time_data_array, fmod_data_array)

    assert x_data.shape[0] == y_dataset.shape[1], "Mismatch between time points and fmod data"

    try:
        out = minimize(objective, global_params, args=(x_data, y_dataset))
    except Exception as e:
        print(f'Global fitting failed: {e}')
        return None

    # Compute predicted values
    ndata, npoints = y_dataset.shape
    predicted = np.zeros_like(y_dataset)
    for i in range(ndata):
        predicted[i, :] = fmod_dataset(out.params, i, x_data)

    # Flatten arrays for R² calculation
    y_true = y_dataset.flatten()
    y_pred = predicted.flatten()

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot

    return out.params['log_kdeg_1'].value, out.params['log_kdeg_1'].stderr, out.chisqr, r2


# === Tempgrad 2-state melting fit ===
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
    K1 = np.exp((f/R)*(1/(g+273.15) - 1/(temp)))
    Q1 = 1 + K1
    fracu = 1 / Q1
    fracf = K1 / Q1
    basef = a * x + b
    baseu = c * x + d

    final = np.log(fracu) * baseu + np.log(fracf) * basef
    final = fracu * baseu + fracf * basef
    return final

def fit_meltcurve(x, y, kadd_params = None):
    
    # Make sure x is sorted

    # Guess top baseline
    model = LinearModel()
    params = model.guess(y[:3], x = x[:3])
    top_fit = model.fit(y[:3], params, x = x[:3])
    init_m_top = top_fit.params['slope'].value
    init_b_top = top_fit.params['intercept'].value

    # Guess bottom baseline
    params = model.guess(y[-3:], x = x[-3:])
    bot_fit = model.fit(y[-3:], params, x = x[-3:])
    init_m_bot = bot_fit.params['slope'].value
    init_b_bot = bot_fit.params['intercept'].value
    
    init_m = (init_m_top + init_m_bot) / 2

    # Actual fit
    melt_model = Model(melt_fit)
    melt_params = melt_model.make_params(a = init_m_bot, b = init_b_bot, c = init_m_top, d = init_b_top, f = -500, g = 40)
    melt_params['g'].vary = True
    # bottom intercept needs to be lower than upper intercept
    melt_params['b'].max = init_b_top

    if kadd_params is not None:
        # lock to kadd_params
        kadd_slope, kadd_intercept = kadd_params
        melt_params['c'].value = kadd_slope
        melt_params['d'].value = kadd_intercept
        melt_params['c'].vary = False
        melt_params['d'].vary = False
    

    melt_result = melt_model.fit(y, melt_params, x = x, method = 'least_squares', verbose = True)

    return melt_result

# calculate smoothed best-fit values based on melt_result
def calc_smoothed_best_fit(melt_result):
    x = melt_result.userkws['x']
    x_data = np.linspace(min(x), max(x), 1000)
    y_data = melt_result.eval(x = x_data)
    return x_data, y_data


# === 4. Generic fit wrapper ===
def run_fit(model, params, x, y):
    """
    Fit a model to (x, y) data and return lmfit Result.
    """
    return model.fit(y, params, x=x)