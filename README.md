# Equity Portfolio Risk Modelling & Optimisation

Python behind my second-year University of Bath portfolio project
(coursework December 2025 – February 2026, graded First, 71%; tracked out of
sample to August 2026): a systematic UK equity
portfolio built with a Monte Carlo weight optimiser, sector and drawdown
analytics, and an 18-year test of whether rising market-wide correlation
predicts downturns. Daily data from Yahoo Finance via `yfinance`; NumPy,
pandas, statsmodels, matplotlib.

## Files

| file | what it does |
|------|--------------|
| `sector_resilience.py` | normalised performance, max drawdown and performance-adjusted MDD for same-sector baskets (e.g. the five UK banks vs their sector mean) |
| `correlation_matrix.py` | Pearson correlation heatmap of the candidate universe |
| `correlation_regime_study.py` | aggregate cross-asset correlation per asset and window, drawn as a surface over 2007–2025, then regressed against next-window performance |
| `sharpe_screen.py` | per-stock Sharpe screen of the candidate universe |
| `monte_carlo_weight_optimisation.py` | GBM Monte Carlo over random weight vectors (1,000 portfolios × 200 paths), scored by mean Sharpe, adjusted Sharpe and dispersion of outcomes |
| `gradient_descent_weights.py` | exploratory gradient-ascent alternative over the feasible weight region |
| `out_of_sample_tracking.py` | tracks the submitted weights after the submission date |

## Figures

![Normalised performance of the five UK banks against the sector mean](docs/figures/uk_banks_normalised_performance.png)

*`sector_resilience.py`: NatWest, Barclays, Lloyds, HSBC and Standard Chartered
rebased to 100, sector mean in red.*

![Aggregate correlation per asset over 50 windows, 2007–2025](docs/figures/aggregate_correlation_surface.png)

*`correlation_regime_study.py`: each asset's summed correlation with the rest
of the basket, per ~90-day window; the ridges are the windows where the whole
basket moved together.*

![Next-window return against current aggregate correlation, raw comparison](docs/figures/correlation_vs_next_period_return.png)

*The regression behind the null result: the fitted slope is positive, not the
negative one the hypothesis predicted.*

## Results

* **Weight optimisation.** Adjusted Sharpe (mean Sharpe across a portfolio's
  simulated paths divided by its standard deviation) was the metric chosen for
  the final weights, reflecting a preference for stable expected returns over
  the two-month horizon of the exercise.
* **Out of sample.** Over eight months after submission the optimised weights
  delivered a Sharpe of 1.6 against 0.9 for the FTSE 100, a 53% smaller maximum
  drawdown than the index and 31% smaller than an equal-weight baseline.
  `out_of_sample_tracking.py` is the script used for that check (its date
  constants are those of the first monthly run).
* **Correlation as a downturn predictor — null.** Across three definitions of
  next-window performance and two basket versions (with and without the gold and
  Treasury hedges), the hypothesised negative slope never appeared: the two
  significant fits (raw comparison, p < 0.001) were positive, the other four not
  significant. The likely reason is documented in the report: correlations rise
  *into* a top, so most high-correlation windows are still followed by gains.

## Running things

```
pip install -r requirements.txt
python monte_carlo_weight_optimisation.py
```

Each script downloads what it needs from Yahoo Finance at run time (an
internet connection is required; `yfinance` occasionally changes its column
layout, which `fetch_close_prices` handles). Tickers, date ranges and simulation
sizes are constants at the top of each file.

## Notes

Coursework code, cleaned up for publication: names, imports and one unfinished
loop were fixed, the methods are untouched. The Sharpe figures above are those
reported in my CV and were produced with these scripts; re-running them today
will download a different window of data.
