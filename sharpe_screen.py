"""
Per-stock Sharpe ratio screen over the candidate universe (daily data, 4% risk-free).

Original file: LSE_007 maximsers individual sharpe.py
"""

import yfinance as yf
import numpy as np

RISK_FREE = 0.04 

FINAL_BASKET = ["GSK.L", "SCT.L", "BA.L", "HILS.L", "RKH.L", "NG.L", "SGRO.L", "INCH.L", "HSBA.L", "BT-A.L"]
# candidate universe screened for the report
tickers = ["GSK.L", "CTEC.L", "SGE.L", "REL.L", "SCT.L", "BAB.L", "BA.L", "HILS.L", "MGNS.L", "RKH.L", "SSE.L", "NG.L", "RIO.L", "SGRO.L", "AUTO.L", "INCH.L", "BLND.L", "HSBA.L", "TSCO.L", "SBRY.L", "BT-A.L"]


data = yf.download(tickers, start='2023-01-01', end='2025-10-01')['Close']
data = data.pct_change().dropna()

print(data)

def comp_sharpe(dt, tkrs, rf_yearly=RISK_FREE):
    daily_ret = dt[f"{tkrs}"]
    rf_daily = (1 + rf_yearly)**(1/252) - 1
    excess_daily = daily_ret - rf_daily
    mean_xs = np.mean(excess_daily, axis=0)
    std_xs = np.std(excess_daily, axis=0, ddof=1)
    sharpe = mean_xs/std_xs
    annual_sharpe = sharpe * np.sqrt(252)
    
    return annual_sharpe
    

ans_dict = {}
for i in tickers :
    #print(i)
    res = comp_sharpe(data, i)
    #print(res)
    ans_dict[f'{i}'] = res

for i in ans_dict:
    print(f"{i} : {ans_dict[f'{i}']}")


    

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
