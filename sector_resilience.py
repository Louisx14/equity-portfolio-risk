"""
Sector resilience: normalised performance, max drawdown and
performance-adjusted max drawdown (PAMDD = final price / MDD) for a basket of
same-sector LSE stocks, e.g. the UK banks vs their sector mean.

Original file: LSE_005 adjusted performance.py
"""

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# this function was not used in the end but it fetches different data aboit the stock and processes it all in teh following ways:
# - fetches the price and scales it so we have a constant unit and not a mic of GBX/GBp and GBP
# - fetches the EPS and P/E ratios and compaers them to the share price.
def analyze_stocks(tickers):
    results = []

    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info

            # --- Basic fields ---
            name = info.get("shortName", ticker)
            price = info.get("currentPrice", None)
            eps_trailing = info.get("epsTrailingTwelveMonths")
            eps_forward = info.get("epsForward")

            # --- Detect if this is a London-listed stock (".L") ---
            is_lse = ticker.endswith(".L")

            # --- Rescale prices if needed ---
            # Yahoo reports LSE prices in GBp, but EPS often in GBP.
            # So we divide by 100 to bring price to GBP scale.
            if is_lse and price and price > 100:
                price = price / 100  # Convert from GBp → GBP

            # --- Choose which EPS to use ---
            eps_used = None
            if eps_trailing and eps_trailing > 0:
                eps_used = eps_trailing
            elif eps_forward and eps_forward > 0:
                eps_used = eps_forward

            # --- Compute P/E ---
            pe = None
            if price and eps_used and eps_used != 0:
                pe = price / eps_used

            results.append([name, price, eps_used, pe])

        except Exception as e:
            results.append([ticker, None, None, None])
            print(f"⚠️ Failed to fetch info for {ticker}: {e}")

    # --- Plot EPS vs Price ---
    plt.figure()
    for i, (name, price, eps, pe) in enumerate(results):
        if price and eps:
            plt.scatter(price, eps)
            plt.text(price * 1.01, eps, tickers[i], fontsize=9, va='center')
    plt.title("EPS vs Price")
    plt.xlabel("Current Price (GBP)")
    plt.ylabel("EPS Used (GBP)")
    plt.show()

    plt.figure()
    for i, (name, price, eps, pe) in enumerate(results):
        if price and pe:
            plt.scatter(price, pe)
            plt.text(price * 1.01, pe, tickers[i], fontsize=9, va='center')
    plt.title("P/E Ratio vs Price")
    plt.xlabel("Current Price (GBP)")
    plt.ylabel("P/E Ratio")
    plt.show()

    return results

# this function helped sort large lists of stocks according to sectors
def info_scan(tkrs):
    for i in tkrs:
        info = yf.Ticker(i).info
        sector = info.get("sector", "")
        industry = info.get("industry", "")
        print(f"{i}: Sector: {sector}, Industry: {industry}")

# this function plots the adjusted performance graph and computes and prints the MDD and PAMDD of the stocks inputed.
def plot_adjusted_performance(tkrs, data):
    returns = data.pct_change().dropna()
    adjusted_perf = pd.DataFrame()

    # Compute individual stock adjusted performances
    for i in tkrs:
        START = 100
        rolling_performance = (1 + returns[i]).cumprod() * START
        adjusted_perf[i] = rolling_performance
        plt.plot(adjusted_perf[i], linewidth=1, alpha=0.7)

    # Industry standard (mean returns)
    industry_returns = returns.mean(axis=1)
    adjusted_perf["industry_standard"] = (1 + industry_returns).cumprod() * 100

    plt.plot(adjusted_perf["industry_standard"], color='red', linewidth=1, label='Industry Standard')

    plt.title("Adjusted Performance of the Stocks")
    plt.xlabel("Date")
    plt.ylabel("Adjusted Price")
    plt.legend(tkrs + ["industry_standard"])
    plt.gca().set_facecolor('black')
    plt.show()

    # Print final performance
    print(adjusted_perf.iloc[-1])

    # MDD function
    def max_drawdown(series):
        roll_max = series.cummax()
        drawdown = (series - roll_max) / roll_max
        return drawdown.min() * 100  # percentage
    
    print("\nMaximum Drawdown (MDD %) for each stock:")
    for col in adjusted_perf.columns:
        print(f"{col}: {max_drawdown(adjusted_perf[col]):.2f}%")
    
    # PAMDD function
    def pamdd(series):
        # final price
        final_price = series.iloc[-1]

        # compute MDD using your formula
        peak = series.expanding(min_periods=1).max()
        drawdown = (series - peak) / peak
        mdd = abs(drawdown.min()) * 100  # positive

        return final_price / mdd

    print("\nPerformance Adjusted Maximum Drawdown (PAMDD) for each stock:")
    for col in adjusted_perf.columns:
        print(f"{col}: {pamdd(adjusted_perf[col]):.5f}")

#telecom_tickers = ['VOD.L', 'AAF.L', 'HTWS.L', 'GAMA.L', "TEP.L", 'ZEG.L', "AFRN.L", "BBB.L", "CYAN.L", "MAI.L", "ORNT.L", "CLX.L", "BT-A.L"]
#financial_services_tickers = ["HSBA.L","BARC.L","LLOY.L","NWG.L","STAN.L","LSEG.L","MNG.L","RAT.L","OSB.L","N91.L","AJB.L","PHNX.L","PAG.L"]
#tkrs = ["NWG.L", "HSBA.L", "BARC.L", "LLOY.L", "STAN.L", "MNG.L", "RAT.L", "OSB.L", "N91.L"]
tkrs = ["NWG.L", "HSBA.L", "BARC.L", "LLOY.L", "STAN.L"]


# setting our tickiers, fetchig the data and feeding it to the function to run the script.
TICKER = tkrs
data = yf.download(TICKER, start='2023-11-11', end='2025-11-10')['Close']
plot_adjusted_performance(TICKER, data)




