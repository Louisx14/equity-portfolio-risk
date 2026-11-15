"""
LSE Portfolio — Out-of-Sample Comparison Report
===============================================
Compares, on data the optimiser never saw:

    * "MC-optimised"      — the raw Monte Carlo average_port weights (report p.15)
    * "Final (submitted)" — the handed-in portfolio after manual tweaks (report p.2)
    * "Equal-weight"      — naive 1/N baseline
    * "^FTSE"             — market benchmark

Windows (from LSE_016: period='3y' ending at submission):
    In-sample  : 2022-12-05 → 2025-12-05   (what the optimiser trained on)
    Out-of-sample: 2025-12-05 → today      (genuinely unseen — portfolio inception
                                            at the 5 Dec opening prices)

Run:  pip install yfinance   (if needed)
      python out_of_sample_comparison.py
"""

import sys
import numpy as np
import pandas as pd

# ════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════

TICKERS = ["GSK.L", "SCT.L", "BA.L", "HILS.L", "RKH.L",
           "NG.L", "SGRO.L", "INCH.L", "HSBA.L", "BT-A.L"]

# Pre-optimisation allocation (from fundamental analysis, BEFORE Monte Carlo)
# !! Confirm these against your notes — order matches TICKERS above.
#ORIGINAL_WEIGHTS = [0.12, 0.05, 0.08, 0.12, 0.12, 0.12, 0.05, 0.12, 0.12, 0.10]
ORIGINAL_WEIGHTS = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,]

# Raw Monte Carlo output (average_port, report p.15)
MC_WEIGHTS = [0.12848, 0.20086, 0.09362, 0.14097, 0.08660,
              0.06989, 0.03041, 0.08558, 0.08349, 0.08005]

# Final handed-in portfolio (report p.2)
FINAL_WEIGHTS = [0.10, 0.16, 0.10, 0.12, 0.10,
                 0.08, 0.05, 0.11, 0.08, 0.10]


BENCHMARK = "^FTSE"

CSV_PATH = None              # None = download via yfinance; or point at a saved
                             # wide CSV (Date index, one close column per ticker)

START_DATE = "2022-12-05"    # start of the 3y training window
END_DATE = "2026-12-31"    # future date = "up to today"
OOS_SPLIT_DATE = "2025-12-05"    # optimiser saw nothing after this

RISK_FREE_ANNUAL = 0.04      # matches RISK_FREE in LSE_016
TRADING_DAYS = 252
CVAR_LEVEL = 0.95

W = 116  # report width

# ════════════════════════════════════════════════════════════════════
# No edits needed below this line
# ════════════════════════════════════════════════════════════════════


def _extract_close(raw, tickers):
    """Handle the different shapes yfinance can return."""
    if isinstance(raw.columns, pd.MultiIndex):
        lvl_vals_0 = raw.columns.get_level_values(0)
        lvl_vals_1 = raw.columns.get_level_values(1)
        if "Close" in lvl_vals_0:
            close = raw.xs("Close", axis=1, level=0)
        elif "Close" in lvl_vals_1:
            close = raw.xs("Close", axis=1, level=1)
        else:
            sys.exit(f"Couldn't find 'Close' in downloaded columns: {raw.columns[:6]}")
    elif "Close" in raw.columns:
        close = raw[["Close"]].copy()
        close.columns = [tickers[0]]
    else:
        close = raw.copy()
    return close


def load_prices():
    if CSV_PATH:
        px = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
        missing = [t for t in TICKERS if t not in px.columns]
        if missing:
            sys.exit(f"CSV is missing columns for: {missing}")
        cols = list(TICKERS) + ([BENCHMARK] if BENCHMARK and BENCHMARK in px.columns else [])
        px = px[cols]
    else:
        try:
            import yfinance as yf
        except ImportError:
            sys.exit("yfinance not installed — `pip install yfinance` or set CSV_PATH.")
        all_tickers = list(TICKERS) + ([BENCHMARK] if BENCHMARK else [])
        raw = yf.download(all_tickers, start=START_DATE, end=END_DATE,
                          auto_adjust=True, progress=False, group_by="column")
        px = _extract_close(raw, all_tickers)

        # Retry any ticker that came back empty, one at a time
        for t in all_tickers:
            if t not in px.columns or px[t].dropna().empty:
                print(f"  [retry] {t} returned no data in batch download, retrying alone...")
                solo = yf.download(t, start=START_DATE, end=END_DATE,
                                   auto_adjust=True, progress=False)
                solo_close = _extract_close(solo, [t])
                if not solo_close.empty and not solo_close.iloc[:, 0].dropna().empty:
                    px[t] = solo_close.iloc[:, 0]

        # ── Diagnostics: show what we actually got ──
        print("\nDATA CHECK (non-missing daily closes per ticker):")
        dead = []
        for t in all_tickers:
            n_obs = int(px[t].notna().sum()) if t in px.columns else 0
            first = px[t].first_valid_index() if n_obs else None
            last = px[t].last_valid_index() if n_obs else None
            span = f"{first.date()} → {last.date()}" if n_obs else "NO DATA"
            print(f"  {t:<8} {n_obs:>5} obs   {span}")
            if n_obs == 0:
                dead.append(t)

        if dead:
            if BENCHMARK in dead and not (set(dead) & set(TICKERS)):
                print(f"  [warn] benchmark {BENCHMARK} unavailable — continuing without it.")
                px = px.drop(columns=[BENCHMARK], errors="ignore")
            else:
                sys.exit(f"\nNo data for portfolio ticker(s): {dead}. "
                         "Check the symbol(s) on finance.yahoo.com, or your yfinance "
                         "version (`pip install -U yfinance` fixes most Yahoo API breakages).")

    px = px.loc[START_DATE:END_DATE].ffill().dropna(how="any")
    if px.empty:
        sys.exit("No overlapping price data in the chosen window (see DATA CHECK above — "
                 "the row-wise overlap of the listed date spans is empty).")
    return px


def norm(w):
    w = np.asarray(w, dtype=float)
    return w / w.sum()


def portfolio_returns(asset_rets: pd.DataFrame, weights) -> pd.Series:
    return asset_rets.mul(norm(weights), axis=1).sum(axis=1)


def max_drawdown(rets: pd.Series) -> float:
    wealth = (1 + rets).cumprod()
    return (wealth / wealth.cummax() - 1).min()


def metrics(rets: pd.Series) -> dict:
    rf_daily = RISK_FREE_ANNUAL / TRADING_DAYS
    ann_ret = (1 + rets).prod() ** (TRADING_DAYS / len(rets)) - 1
    ann_vol = rets.std(ddof=1) * np.sqrt(TRADING_DAYS)
    sharpe = ((rets - rf_daily).mean() / rets.std(ddof=1)) * np.sqrt(TRADING_DAYS)
    downside = rets[rets < 0]
    dd_dev = downside.std(ddof=1) * np.sqrt(TRADING_DAYS) if len(downside) > 1 else np.nan
    var_cut = rets.quantile(1 - CVAR_LEVEL)
    cvar = rets[rets <= var_cut].mean()
    return {
        "Ann. return": ann_ret,
        "Ann. volatility": ann_vol,
        "Sharpe ratio": sharpe,
        "Max drawdown": max_drawdown(rets),
        "Downside dev.": dd_dev,
        f"CVaR {int(CVAR_LEVEL*100)}% (daily)": cvar,
    }


def fmt(name, value):
    if name == "Sharpe ratio":
        return f"{value:.2f}"
    return f"{value * 100:.1f}%"


def print_metric_table(title, results: dict):
    names = list(results.keys())
    metric_names = list(next(iter(results.values())).keys())
    col_w = max(15, max(len(n) for n in names) + 2)
    label_w = W - col_w * len(names)

    print(f"\n{title}")
    print("-" * W)
    print(" " * label_w + "".join(n.rjust(col_w) for n in names))
    for m in metric_names:
        row = m.ljust(label_w)
        row += "".join(fmt(m, results[n][m]).rjust(col_w) for n in names)
        print(row)
    print("-" * W)


def turnover_block(title, w_from, w_to):
    a, b = norm(w_from), norm(w_to)
    print(f"\n{title}")
    print("-" * W)
    for t, x, y in zip(TICKERS, a, b):
        bar_len = int(round(abs(y - x) * 100))
        bar = ("+" if y > x else "-") * min(bar_len, 40)
        print(f"  {t:<8} {x*100:6.1f}%  →  {y*100:6.1f}%   ({(y-x)*100:+5.1f} pp)  {bar}")
    turn = 0.5 * np.abs(b - a).sum()
    print(f"\n  Turnover: {turn*100:.1f}% of capital reallocated")
    print("-" * W)


def run_window(label, asset_rets, bench_rets, portfolios):
    if len(asset_rets) < 30:
        print(f"\n[skip] {label}: fewer than 30 observations.")
        return None
    results = {}
    for name, w in portfolios.items():
        results[name] = metrics(portfolio_returns(asset_rets, w))
    if bench_rets is not None:
        results["Benchmark"] = metrics(bench_rets)
    span = f"{asset_rets.index[0].date()} → {asset_rets.index[-1].date()}  ({len(asset_rets)} days)"
    print_metric_table(f"{label}   [{span}]", results)
    return results


def pair_delta(results, a, b):
    """Key deltas of portfolio a relative to portfolio b."""
    if a not in results or b not in results:
        return
    pa, pb = results[a], results[b]
    vol_red = (pb["Ann. volatility"] - pa["Ann. volatility"]) / pb["Ann. volatility"]
    dd_red = (abs(pb["Max drawdown"]) - abs(pa["Max drawdown"])) / abs(pb["Max drawdown"])
    ret_spread = pa["Ann. return"] - pb["Ann. return"]
    print(f"{a}  vs  {b}:")
    print(f"    volatility {'reduced' if vol_red >= 0 else 'INCREASED'} by "
          f"{abs(vol_red)*100:.0f}%   |   max drawdown "
          f"{'reduced' if dd_red >= 0 else 'INCREASED'} by {abs(dd_red)*100:.0f}%")
    print(f"    return spread {ret_spread*100:+.1f} pp/yr   |   Sharpe "
          f"{pb['Sharpe ratio']:.2f} → {pa['Sharpe ratio']:.2f} "
          f"({pa['Sharpe ratio'] - pb['Sharpe ratio']:+.2f})")


def main():
    print("=" * W)
    print("LSE PORTFOLIO — OUT-OF-SAMPLE COMPARISON".center(W))
    print("=" * W)

    px = load_prices()
    asset_px = px[TICKERS]
    rets = asset_px.pct_change().dropna()
    bench_rets = None
    if BENCHMARK and BENCHMARK in px.columns:
        bench_rets = px[BENCHMARK].pct_change().dropna().reindex(rets.index).dropna()
        rets = rets.loc[bench_rets.index]

    n = len(TICKERS)
    portfolios = {
        "Original": ORIGINAL_WEIGHTS,
        "MC-optimised": MC_WEIGHTS,
        "Final (submitted)": FINAL_WEIGHTS,
        "Equal-weight": [1 / n] * n,
    }

    # 1 ── Reallocation
    turnover_block("REALLOCATION A: Original → MC-optimised (what the optimiser changed)",
                   ORIGINAL_WEIGHTS, MC_WEIGHTS)
    turnover_block("REALLOCATION B: MC-optimised → Final (your manual/fundamental tweaks)",
                   MC_WEIGHTS, FINAL_WEIGHTS)

    # 2 ── In-sample vs out-of-sample
    run_window("IN-SAMPLE (optimiser training window)",
               rets.loc[:OOS_SPLIT_DATE],
               bench_rets.loc[:OOS_SPLIT_DATE] if bench_rets is not None else None,
               portfolios)
    headline = run_window("OUT-OF-SAMPLE (since portfolio inception — unseen data)",
                          rets.loc[OOS_SPLIT_DATE:],
                          bench_rets.loc[OOS_SPLIT_DATE:] if bench_rets is not None else None,
                          portfolios)

    # 3 ── Summary deltas (out-of-sample only — these are the CV numbers)
    if headline:
        print("\nOUT-OF-SAMPLE SUMMARY — CV-READY NUMBERS")
        print("-" * W)
        pair_delta(headline, "MC-optimised", "Original")
        pair_delta(headline, "Final (submitted)", "Original")
        pair_delta(headline, "MC-optimised", "Equal-weight")
        pair_delta(headline, "Final (submitted)", "MC-optimised")
        if "Benchmark" in headline:
            pair_delta(headline, "Final (submitted)", "Benchmark")
        print("-" * W)
        print("\n  Note: annualised figures from windows of under a year are noisy; the Sharpe standard error is about 1.")
        print("  Quote total return and drawdown over the whole window rather than annualised returns.")
    print()


if __name__ == "__main__":
    main()