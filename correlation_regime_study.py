"""
Does rising market-wide correlation predict a downturn? (Nov 2025 – Jan 2026)

Daily data for a 20-asset diversified basket, 2007-11 to 2025-11, sliced
into 50 windows of ~90 days. For each window and asset, the aggregate
correlation coefficient (sum of the asset's correlations with the rest of
the basket) is computed and drawn as a surface over time; then next-window
performance (raw, average-price and minimum-price definitions, with and
without the gold / Treasury hedges) is regressed on it.

Result: null. The hypothesis predicts a negative slope; the significant fits
(raw comparison, p < 0.001) have a positive one and the other four are not
significant. Documented in the project report.

Original file: LSE_004 3d correlation matrice.py
"""

import pandas as pd
import yfinance as yf
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from matplotlib import dates as mdates
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection on older matplotlib)
from scipy.interpolate import griddata

MULTI = 1
SEGMENTS = 50

# this function rounds values and takes as inputs a list or array of values and rounds all of them winthin the list/array at teh desired decimal place.
def round_matrix(mtrx, hhh = None):
    rounded = []
    for i in range(len(mtrx)):
        y = []
        for j in mtrx[i]:
            if hhh != None:
                y.append(round(j, hhh))
            else:
                y.append(j)
        rounded.append(y)
    return rounded

# this function splits OHLC price dataframes into smaller ones. this used to seperate the frames withing different sections for our per period analysis and computations.
def split_dataframe(df, n_parts):
    chunk_size = len(df) // n_parts
    print(f"Each part will have approximately {chunk_size} rows.")
    result = {}
    for i in range(n_parts):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < n_parts - 1 else len(df)
        result[i] = df.iloc[start:end]
    
    return result

# this functions computes the correlation matrix of each slice of our data and also transofmrs the raw OHLC data innto pct_change() form.
def aggregate_correlation(df, n_parts):
    split_dfs = split_dataframe(df, n_parts)
    resd = {}
    aggregate_correlation = {}
    for i in split_dfs:
        tranch = []
        x = split_dfs[i].pct_change().dropna()
        corr_matrix = x.corr()
        corr = corr_matrix.values.tolist()
        corr = round_matrix(corr)
        resd[i] = corr
        y = [sum(k)-1 for k in corr]
        aggregate_correlation[i] = y

    return resd, aggregate_correlation

# this function plots the 3d graph of our per stock aggregate correlation measure overtime for the whole basket of stocks. (figures 6 and 7 of the essay).
def plot_aggregate_correlation_surface(aggregate_correlation, split_dfs, tickers=None, cmap='plasma', interp_method='cubic'):
    interval_dates = {}
    for i, df_part in split_dfs.items():
        if i == 1:
            print(f"each dataframe is made of {len(df_part)} rows")
        mid_idx = len(df_part) // 2
        interval_dates[i] = pd.to_datetime(df_part.index[mid_idx])

    data = []
    for i, values in aggregate_correlation.items():
        for j, val in enumerate(values):
            data.append([interval_dates[i], j, val])
    df = pd.DataFrame(data, columns=['date', 'ticker_idx', 'value'])

    df['date_num'] = mdates.date2num(df['date'])

    x = np.linspace(df['date_num'].min(), df['date_num'].max(), 100)
    y = np.linspace(df['ticker_idx'].min(), df['ticker_idx'].max(), 100)
    X, Y = np.meshgrid(x, y)

    Z = griddata((df['date_num'], df['ticker_idx']), df['value'], (X, Y), method=interp_method)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([3, 1.5, 1]) 

    surf = ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor='none', alpha=0.95)

    ax.set_xlabel('Date')
    ax.set_ylabel('Ticker')
    ax.set_zlabel('Value')
    ax.set_title('3D Surface Plot of aggregate_correlation (with Date Axis)')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    for label in ax.xaxis.get_ticklabels():
        label.set_rotation(90)
        label.set_horizontalalignment('center')
        label.set_verticalalignment('center')

    if tickers:
        ax.set_yticks(range(len(tickers)))
        ax.set_yticklabels(tickers)

        for label in ax.yaxis.get_ticklabels():
            label.set_rotation(90)
            label.set_verticalalignment('center')

    fig.colorbar(surf, ax=ax, shrink=0.6, label='Value')

    plt.tight_layout()
    plt.show()

# this function plots our per stock aggregate correlation measure overtime for the whole basket of stocks but in 2d this time. it is only a different way to do the same thing.
def plot_aggregate_correlation_2d(aggregate_correlation, split_dfs, tickers=None):
    interval_dates = {}
    dataxe = []
    for i, df_part in split_dfs.items():
        mid_idx = len(df_part) // 2
        interval_dates[i] = pd.to_datetime(df_part.index[mid_idx])
        dataxe.append(interval_dates[i])

    y = []
    for i in range(len(aggregate_correlation[0])):
        y.append(i)

    plt.figure(figsize=(12, 6))
    for j in range(len(aggregate_correlation[0])):
        for i in range(len(aggregate_correlation)):
            plt.plot(dataxe[i], aggregate_correlation[i][j], label=f"ticker {tickers[j]}")
    plt.show()

# once again just a dfferent mean of visualising the same thing.
def plot_aggregate_correlation_lines(aggregate_correlation, split_dfs, tickers=None):
    midpoints = []
    for i in sorted(split_dfs.keys()):
        df_part = split_dfs[i]
        mid_idx = len(df_part) // 2
        midpoints.append(pd.to_datetime(df_part.index[mid_idx]))

    # Plot
    plt.figure(figsize=(12, 6))
    num_tickers = len(aggregate_correlation[0])

    for j in range(num_tickers):
        y = [aggregate_correlation[i][j] for i in range(len(aggregate_correlation))]
        x, y = zip(*sorted(zip(midpoints, y)))  # ensure chronological order
        plt.plot(x, y, marker='o', label=tickers[j] if tickers else f"Ticker {j}")

    plt.xlabel("Date (interval midpoint)")
    plt.ylabel("aggregate_correlation value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# this function plots the regression graphs of period t correlation levels agaisnt dfferent measures in period t+1 and returns the regression p value as well.
def plot_correlation_vs_future_return(resdx_dict, split_dfs, tickers, minimum=False, ave=False, plot=False):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    X_vals = []  # aggregate_correlation
    Y_vals = []  # next period returns
    labels = []  # ticker names (for coloring)

    segment_keys = sorted(split_dfs.keys())
    n_segments = len(segment_keys)

    if minimum:
        for i in range(n_segments - 1):
            df_curr = split_dfs[i]
            df_next = split_dfs[i + 1]

            price_curr = df_curr.iloc[-1]
            price_next = df_next.min()
            future_returns = (price_next - price_curr) / price_curr

            for j, ticker in enumerate(tickers):
                X_vals.append(resdx_dict[i][j])
                Y_vals.append(future_returns[ticker])
                labels.append(ticker)

    elif ave:
        for i in range(n_segments - 1):
            df_curr = split_dfs[i]
            df_next = split_dfs[i + 1]

            price_curr_low = df_curr.min()
            price_curr_high = df_curr.max()
            price_curr = (price_curr_low + price_curr_high) / 2
            price_next_low = df_next.min()
            price_next_high = df_next.max()
            price_next = (price_next_low + price_next_high) / 2
            future_returns = (price_next - price_curr) / price_curr

            for j, ticker in enumerate(tickers):
                X_vals.append(resdx_dict[i][j])
                Y_vals.append(future_returns[ticker])
                labels.append(ticker)

    else:
        for i in range(n_segments - 1):
            df_curr = split_dfs[i]
            df_next = split_dfs[i + 1]

            price_curr = df_curr.iloc[-1]
            price_next = df_next.iloc[-1]

            future_returns = (price_next - price_curr) / price_curr

            for j, ticker in enumerate(tickers):
                X_vals.append(resdx_dict[i][j])
                Y_vals.append(future_returns[ticker])
                labels.append(ticker)

    df_plot = pd.DataFrame({
        "aggregate_correlation": X_vals,
        "future_return": Y_vals,
        "ticker": labels
    })

    X = np.array(X_vals)
    Y = np.array(Y_vals)

    X_sm = sm.add_constant(X)

    model = sm.OLS(Y, X_sm).fit()

    print(model.summary())

    slope, intercept = np.polyfit(X, Y, 1)

    plt.figure(figsize=(10, 6))

    for ticker in tickers:
        df_t = df_plot[df_plot['ticker'] == ticker]
        plt.scatter(df_t['aggregate_correlation'], df_t['future_return'], label=ticker, alpha=0.7)

    x_line = np.linspace(min(X), max(X), 200)
    y_line = slope * x_line + intercept
    plt.plot(x_line, y_line, linewidth=2)

    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.xlabel("aggregate_correlation value (current segment)")
    plt.ylabel("Next-segment return")
    plt.title("aggregate_correlation vs Next-period Price Movement\n(With Linear Regression)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if plot == True:
        plt.show()

    return df_plot

# setting up our test portfolio. 
tickers = [
    # --- Tech ---
    "AAPL",   # Apple
    "MSFT",   # Microsoft
    "GOOGL",  # Alphabet
    "NVDA",   # Nvidia
    "AMZN",   # Amazon

    # --- Financials ---
    "JPM",    # JPMorgan
    "BAC",    # Bank of America

    # --- Industrials / Manufacturing ---
    "CAT",    # Caterpillar
    "BA",     # Boeing

    # --- Consumer ---
    "PG",     # Procter & Gamble
    "KO",     # Coca-Cola
    "MCD",    # McDonald's

    # --- Energy / Materials ---
    "XOM",    # ExxonMobil
    "CVX",    # Chevron
    "BHP",    # BHP Group

    # --- Healthcare ---
    "JNJ",    # Johnson & Johnson
    "PFE",    # Pfizer

    # --- Utilities ---
    "NEE",    # NextEra Energy

    # --- Hedges ---
    "GLD",    # Gold ETF
    "TLT"     # Long-term US Treasuries (interest rate hedge)
]

# fetching data
data = yf.download(tickers, start='2007-11-11', end='2025-11-10')['Close']

# pct_change transofrmation for 2d corr matrix.
returns = data.pct_change().dropna()
corr_matrix = returns.corr()
matrix = corr_matrix.values.tolist()
matrix = round_matrix(matrix)

for j in matrix:
    print(j)
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix of Asset Returns')
#plt.show()

# calling our functions to prepare the data
_, lll = aggregate_correlation(data, SEGMENTS)
segments = split_dataframe(data, SEGMENTS)

# executing the script
plot_aggregate_correlation_surface(lll, segments, tickers=tickers, cmap='plasma', interp_method='cubic')
df_rel = plot_correlation_vs_future_return(lll, segments, tickers, minimum = True, ave = False)
