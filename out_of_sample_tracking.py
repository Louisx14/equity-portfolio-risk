"""
Out-of-sample tracking of the submitted portfolio (from Dec 2025).

Fetches prices for the submitted weights after the submission date and
plots the portfolio path against its constituents / the index, the check
behind the out-of-sample Sharpe and drawdown figures quoted in the README.

Original file: graph making portfolio moves.py
"""

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

tickers = ["GSK.L", "SCT.L", "BAES.L", "HILS.L", "RKH.L", "NG.L", "SGRO.L", "INCH.L", "HSBA.L", "BT-A.L"]
weights = [0.10, 0.16, 0.10, 0.12, 0.10, 0.08, 0.05, 0.11, 0.08, 0.10]
#YYYY-MM-DD
START = "2025-12-30"
END = "2026-02-01"



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

res = fetch_close_prices(tickers, start=START, end=END)
res = res.pct_change().dropna()

res["prtfl"] = res[tickers].dot(weights)

print(res)

norm = (1 + res).cumprod()

plt.figure(figsize=(12, 6))

# Plot individual stocks (red, transparent)
for tkr in tickers:
    plt.plot(
        norm.index,
        norm[tkr],
        color="red",
        alpha=0.4,
        linewidth=1
    )

# Plot portfolio (blue, solid)
plt.plot(
    norm.index,
    norm["prtfl"],
    color="blue",
    alpha=1,
    linewidth=2,
    label="Portfolio"
)

plt.title("Normalized Price Paths (Starting at 1)")
plt.xlabel("Date")
plt.ylabel("Growth of $1")
plt.grid(True)
plt.legend()

plt.show()

print(norm)