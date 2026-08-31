"""
Monte Carlo portfolio weight optimisation (Dec 2025 – Feb 2026).

For each of WEIGHT_SIMS random weight vectors, simulate PER_PORT_SIM daily
GBM paths of the portfolio over SIM_LEN days from each asset's historical
log-return mean and volatility (drift corrected by sigma^2/2), and score the
portfolio by mean Sharpe, adjusted Sharpe (mean / std of the Sharpe across
paths — the project's own risk-aversion measure), and dispersion of final
returns. `find_opt` then picks the weights that maximise or minimise a
chosen metric. The 200,000-path figure in the CV is 1,000 portfolios x 200
paths.

Original file: LSE_016.4 weight optimisation.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# here we set the main parameters of the process 
WEIGHT_SIMS = 1000          # number of portfolio simulated
RISK_FREE = 0.04             # risk free rate (4%)
PER_PORT_SIM = 200          # number of paths generated per portfolios
SIM_LEN = 252                # length of the simulated paths in days
SEED = 2122025               # seed for reproducability

# annotations for this file are inside the functions as they were too numerous to keep outside.

def sharpe_daily(portfolio_sims, rf_yearly=RISK_FREE):
    # This function computes the sharpe of each simulated portfolio based on daily metrics (mean daily performance and daily std).
    # we have added a risk free rate of 4% to be conservative in tking the 3 month uk gov T-bills.
    # after having computed the sharpe based on daily data we annualize it before returning the result.
    daily_returns = (portfolio_sims[1:, :] / portfolio_sims[:-1, :]) - 1  
    rf_daily = (1 + rf_yearly)**(1/252) - 1
    excess_daily = daily_returns - rf_daily
    mean_excess = np.mean(excess_daily, axis=0)
    std_excess = np.std(excess_daily, axis=0, ddof=1)
    sharpe = np.where(std_excess == 0, np.nan, mean_excess / std_excess)
    sharpe_annual = sharpe * np.sqrt(252)

    return sharpe_annual

def find_opt(fnl_res, key, seckey=False, opt_type="max"):
    # this code simply goes through all the collected data after a simulation has ran and finds the optimal portfolios according to our inputs in the function.
    # find_opt(results, "sharpe", "adjusted_sharep", "max") will look for the portfolio weights that maximise the adjusted sharpe ratio.
    if opt_type == "max":
        if seckey:
            strt_val, fdat = fnl_res[0][key][seckey], 0
            for i in range(WEIGHT_SIMS):
                if fnl_res[i][key][seckey] > strt_val:
                    strt_val, fdat = fnl_res[i][key][seckey], i
        else :
            strt_val, fdat = fnl_res[0][key], 0
            for i in range(WEIGHT_SIMS):
                if fnl_res[i][key] > strt_val:
                    strt_val, fdat = fnl_res[i][key], i
    elif opt_type == "min":
        if seckey:
            strt_val, fdat = fnl_res[0][key][seckey], 0
            for i in range(WEIGHT_SIMS):
                if fnl_res[i][key][seckey] < strt_val:
                    strt_val, fdat = fnl_res[i][key][seckey], i
        else :
            strt_val, fdat = fnl_res[0][key], 0
            for i in range(WEIGHT_SIMS):
                if fnl_res[i][key] < strt_val:
                    strt_val, fdat = fnl_res[i][key], i
    return strt_val, fdat

def fetch_close_prices(tickers, start=None, end=None, period=None):
    # this function fetches the price data of our stocks and tries to deal with the many edgcases yfinance can produce when fetching data as such.
    # we forward fill and then backfill to limit stock dependant gaps.
    # we remove all NaN values when they are NaN across a whole row thus eliminating useless rows.
    # we also deal with tickers which may not have a "Close" but only the "Adj Close".
    # the length of the function for such a simple task is due to the importance of using the right data for the rest of our calculations we will compute on it.
    if period and (start or end):
        data = yf.download(tickers, period=period, progress=False, threads=True, group_by='ticker')
    else:
        data = yf.download(tickers, start=start, end=end, progress=False, threads=True, group_by='ticker')

    if isinstance(data.columns, pd.MultiIndex):
        if 'Close' in data.columns.get_level_values(1):
            close = data.xs('Close', axis=1, level=1)
        else:
            close = data.loc[:, data.columns.get_level_values(1).isin(['Adj Close', 'Close'])]
            close = close.xs(close.columns.levels[1][0], axis=1, level=1)
    else:
        if 'Close' in data.columns:
            close = data[['Close']].copy()
            close.columns = [tickers[0] if isinstance(tickers, (list,tuple)) else tickers]
        else:
            close = data.copy()

    close = close.dropna(how='all')
    close = close.ffill().bfill()  

    if close.ndim == 1:
        close = close.to_frame(name=tickers[0])
    try:
        close = close.loc[:, list(tickers)]
    except Exception:
        pass

    return close

def portfolio_log_return_series(close_df, weights=None):
    # This function computes returns ad drops the NaN values (whilst already done in the start we will repeat this a few more times in case certain step or error patterns create new NaN cells).
    # then the function checks if we have inputed weights and if so checks the shape of our weight array is the dame as the x dimension of the pct_change array, to check we have one weight value per asset as expected.
    # we conduct another check that the weights sum to 1 and if not we divide them by their sum, suming them all to 1 once more. 
    # we multiply the weights and their respective asset returns using the dot product function.
    # finally we convert to log units before returning in a panda series format.
    returns = close_df.pct_change().dropna(how='all')
    n = close_df.shape[1]

    if weights is None:
        weights = np.repeat(1.0 / n, n)
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.ndim != 1 or len(weights) != n:
            raise ValueError("weights must be a 1D array of length equal to number of tickers")
        if not np.isclose(weights.sum(), 1.0):
            weights = weights / weights.sum()

    port_ret = returns.dot(weights)
    port_log_ret = np.log1p(port_ret).dropna()

    return port_log_ret, weights

def simulate_gbm_paths_from_logparams(mu_log_daily, sigma_log_daily, n_steps, n_paths, s0=1.0, random_seed=None):
    # this function is at the heart of the code as it simulates the paths
    # we check the seed is set to ensure reproducability 
    # then we specify that each time step is a market day
    # then we calculate our "drift element" which represents the movement we can expect (ignoring shocks and random events) our asset to move in a market day.
    # we create Z which is drawn from a normal distribution and will create one shock per step per path adding the random element into our paths.
    # we compute our increments in which the random shocks are scaled according to the assets volatility (sigma_log_daily * Z).
    # we build the log price paths where each value is the sum of all previous (this is the point of using log so it's additive).
    # then we convert to price paths multiplying S0, the starting value by the exponent of our log perfromance series.
    if random_seed is not None:
        np.random.seed(random_seed)

    dt = 1.0  
    steps = n_steps
    drift_per_step = mu_log_daily - 0.5 * (sigma_log_daily ** 2)
    Z = np.random.normal(size=(steps, n_paths))
    increments = drift_per_step * dt + sigma_log_daily * np.sqrt(dt) * Z  
    log_paths = np.vstack([np.zeros(n_paths), np.cumsum(increments, axis=0)])  
    sim_paths = s0 * np.exp(log_paths)
    return sim_paths

def simulate_portfolio_paths(tickers, close, n_paths=PER_PORT_SIM, n_steps=252, start=None, end=None, period='3y', weights=None, s0=1.0, random_seed=SEED, return_df=True, plot=True, verbose = False):
    # given the length of the function we will annotate this chunk of code part by part;
    # a lot of error catching bocks are here from development stages as this code was made to be used in several ways (prints are optional just like we take several date input format).

    # fetches the log returns and weights and then checks wether we have enough data points to carry on.
    port_log_ret, weights = portfolio_log_return_series(close, weights=weights)
    if port_log_ret.size < 10:
        raise RuntimeError("Not enough valid portfolio return points to estimate statistics.")

    mu_log_daily = port_log_ret.mean() 
    sigma_log_daily = port_log_ret.std(ddof=1) 
    annual_mu = mu_log_daily * 252
    annual_sigma = sigma_log_daily * np.sqrt(252)

    # through the testing phases for which monte carlo process should be used, all seemed to inflate certain paths greatly over 252 market days. for this purpose we have the following chunk of code.
    # this part of the script shrinks down extreme paths so we don't obtain unrealistic extreme values (previous to this modification, certain paths would reach over +150% and equally unfeasible negative values).
    lower_bound = -0.30
    upper_bound = 0.30
    if (annual_mu < lower_bound) or (annual_mu > upper_bound):
        target_annual = 0.09
        shrink_alpha = 0.5  
        adjusted_annual = shrink_alpha * target_annual + (1 - shrink_alpha) * annual_mu
        mu_log_daily = adjusted_annual / 252.0

    max_annual_sigma = 2.0  
    if annual_sigma > max_annual_sigma:
        sigma_log_daily = max_annual_sigma / np.sqrt(252)

    # here we generate our paths through the function we described earlier.
    sim_paths = simulate_gbm_paths_from_logparams(
        mu_log_daily=float(mu_log_daily),
        sigma_log_daily=float(sigma_log_daily),
        n_steps=n_steps,
        n_paths=n_paths,
        s0=s0,
        random_seed=random_seed
    )

    # we switch from an array to a dataframe and collect the final time steps
    cols = [f'path_{i+1}' for i in range(n_paths)]
    df_paths = pd.DataFrame(sim_paths, index=range(n_steps+1), columns=cols)
    final_vals = df_paths.iloc[-1, :].values
    final_returns = final_vals / s0 - 1.0  

    # the rest of the function under is printing statements, computation of important metrics as well as plotting and data structuring.
    p05 = np.percentile(final_returns, 5)
    p95 = np.percentile(final_returns, 95)
    mean_final = np.mean(final_returns)
    std_final = np.std(final_returns, ddof=1)
    median_final = np.median(final_returns)
    shrp_vctr = sharpe_daily(sim_paths)
    mean_shrp = np.mean(shrp_vctr)
    median_shrp = np.median(shrp_vctr)
    std_shrp = np.std(shrp_vctr, ddof=1)

    ret = {
        "values" : df_paths,
        "final_returns": {
            "5%_percentile": p05,
            "95%_percentile": p95,
            "mean": mean_final,
            "median": median_final,
            "std": std_final
        }, 
        "sharpe": {
            "vctr": shrp_vctr,
            "mean": mean_shrp,
            "median": median_shrp,
            "std": std_shrp,
            "adjusted_sharep" : 1/std_shrp * mean_shrp if std_shrp !=0 else np.nan
        },
        "weights": {tickers[i]: weights[i] for i in range(len(tickers))}
    }

    def fmt_pct(x):
        return f"{100.0 * x:.2f}%"

    if verbose:
        print("=== Portfolio simulation summary ===")
        print(f"Tickers: {tickers}")
        print(f"Weights used: {np.round(weights, 4)}")
        print(f"Historical window used (end-most data from yfinance): {close.index.min().date()} to {close.index.max().date()}")
        print(f"Estimated annualized drift (log-return): {annual_mu:.4f} (≈ {100*annual_mu:.2f}%)")
        print(f"Estimated annualized volatility (log-return): {annual_sigma:.4f} (≈ {100*annual_sigma:.2f}%)")
        print("--- Final performance stats across simulated paths ---")
        print(f"5% percentile (5% VaR): {fmt_pct(p05)}")
        print(f"95% percentile: {fmt_pct(p95)}")
        print(f"Mean final return: {fmt_pct(mean_final)}")
        print(f"Std of final returns: {fmt_pct(std_final)}")
        print("Note: these are percentages of portfolio performance relative to starting value (s0).")

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(df_paths.index, df_paths.values)
        plt.xlabel("Step (days)")
        plt.ylabel("Portfolio value (normalized; S0 = {:.2f})".format(s0))
        plt.title(f"Monte Carlo simulated portfolio paths ({n_paths} paths, {n_steps} steps)")
        plt.grid(alpha=0.3)
        plt.axhline(s0, linestyle='--', linewidth=1, alpha=0.6)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 4))
        plt.hist(final_returns * 100, bins=50)
        plt.xlabel("Final return (%)")
        plt.ylabel("Frequency")
        plt.title("Distribution of final portfolio returns across simulated paths")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    if return_df:
        return ret
    else:
        return None

# both of the following functions only work to present the results a certain way and have no influence on the resutls themselves.
def describe_port (result, mxgl):

    """    ret = {
        "values" : df_paths,
        "final_returns": {
            "5%_percentile": p05,
            "95%_percentile": p95,
            "mean": mean_final,
            "median": median_final,
            "std": std_final
        }, 
        "sharpe": {
            "vctr": shrp_vctr,
            "mean": mean_shrp,
            "median": median_shrp,
            "std": std_shrp,
            "adjusted_sharep" : 1/std_shrp * mean_shrp if std_shrp !=0 else np.nan
        }
    }"""

    print(f"\n================= {mxgl} Optimal Portfolio =================")
    print(f"mean final value: {result['final_returns']['mean']}")
    print(f"median final value: {result['final_returns']['median']}")
    print(f"5% percentile final value: {result['final_returns']['5%_percentile']}")
    print(f"95% percentile final value: {result['final_returns']['95%_percentile']}")
    print(f"mean sharpe ratio: {result['sharpe']['mean']}")
    print(f"median sharpe ratio: {result['sharpe']['median']}")
    print(f"sharpe ratio std: {result['sharpe']['std']}")
    print("\nWeights:")
    #for t, weight in result['weights'].items():
    #     print(f"{t}: {weight:.3f}")
    print(result['weights'])

def result_pretty_print(results):
    exp_val, idx1 = find_opt(results, "sharpe", "mean", "max")
    describe_port(results[idx1], "Mean sharpe")

    print("\n-- Optimal Portfolio by Mean Sharpe --")
    exp_sharpe, idx2 = find_opt(results, "sharpe", "adjusted_sharep", "max")
    describe_port(results[idx2], "adjusted mean sharpe")

    print("\n-- Optimal Portfolio by Mean Sharpe --")
    exp_sharpe, idx2 = find_opt(results, "final_returns", "std", "min")
    describe_port(results[idx2], "final returns std")



# running the functions below and the actual execution of the code.

# setting tickers
tickers = ["GSK.L", "SCT.L", "BAES.L", "HILS.L", "RKH.L", "NG.L", "SGRO.L", "INCH.L", "HSBA.L", "BT-A.L"]

# intialising our result list nad fetching close prices over the past 3 years
results = []
close = fetch_close_prices(tickers, start=None, end=None, period='3y')
if close.shape[0] < 10:
    raise RuntimeError("Not enough historical data fetched. Try a longer period or check tickers.")

#simulating for the desired amount of weights 
for i in range (WEIGHT_SIMS):
    weights = np.random.random(len(tickers))
    weights = weights / np.sum(weights)
    print(f"weights: {weights}")

        # Simulate 500 paths over 252 trading days (1 year)
    df_sim = simulate_portfolio_paths(
        tickers=tickers,
        close=close,
        n_paths=PER_PORT_SIM,
        n_steps=SIM_LEN,
        period='3y',      
        weights=weights,     
        s0=1.0,
        random_seed=None,
        return_df=True,
        plot=False
    )

    results.append(df_sim)

    # df_sim is a DataFrame of shape (n_steps+1, n_paths)
    print(f"{i*100/WEIGHT_SIMS:.4f}% done", end="\r")
print("\nReturned DataFrame shape:", df_sim["values"].shape)

result_pretty_print(results)