"""
Gradient-ascent weight optimisation over the feasible weight region (Jan 2026).

Alternative to the Monte Carlo search: enumerate the corner points of the
weight bounds, then follow the numerical gradient of the objective
(Sharpe by default) from a start portfolio until it stops moving.
Exploratory; the Monte Carlo optimiser is the one used for the report.

Original file: GDPO_001.py
"""

import yfinance as yf
import numpy as np
from math import comb
from more_itertools import distinct_permutations


bounds = [0.02, 0.2]
tickers = ["GSK.L", "SCT.L", "BA.L", "HILS.L", "RKH.L", "NG.L", "SGRO.L", "INCH.L", "HSBA.L", "BT-A.L"]

LEARNING_RATE = 0.03

def fetch_data(tickers, start_date="2018-01-01", end_date="2025-12-27"):
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    returns = data.pct_change().dropna()
    return returns

def count_corners(bounds, dims, pos=False):
    """
    Count the number of extreme points ("corners") of the space defined by:
    - dims dimensions
    - each dimension in [low, high]
    - sum of all dimensions equals 1
    - optional non-negativity constraint

    Parameters
    ----------
    bounds : list or tuple [low, high]
    dims : int
    pos : bool (default False)
        If True, negative values are not allowed

    Returns
    -------
    int : number of corners
    """

    low, high = bounds
    target_sum = 1.0

    # If positivity is required and bounds violate it, no corners exist
    if pos and low < 0:
        return 0

    total_corners = 0

    # Number of variables set to 'high'
    for k in range(dims):
        # dims - 1 variables are fixed, 1 is free
        m = (dims - 1) - k  # number at 'low'

        fixed_sum = k * high + m * low
        free_value = target_sum - fixed_sum

        # Check if free variable is valid
        if low <= free_value <= high:
            if pos and free_value < 0:
                continue

            # Count combinations:
            # choose k highs from dims
            # choose m lows from remaining dims - k
            count = comb(dims, k) * comb(dims - k, m)
            total_corners += count

    return total_corners

def half_find(bounds, dims, pos=False):
    high_rep, low_rep, free_rep, free = 0, 0, 1, 0
    for i in range(len(tickers)):
        #print(i*bounds[1]+bounds[0]*(dims-i-free_rep))
        if (i*bounds[1]+bounds[0]*(dims-i-free_rep)) >= 1:
            high_rep = i-1
            break
    free = 1 - ((high_rep*bounds[1])+(bounds[0]*(dims-high_rep-1)))
    low_rep = dims - high_rep - free_rep
    #print(f"{high_rep}, {low_rep}, {free_rep}, {free}")
    return high_rep, low_rep, free_rep, free

def find_corners(bounds, dims, pos=False):
    h, l, f, free = half_find(bounds, dims, pos=False)
    h_size = count_corners(bounds, dims, pos=False)
    #crnrs = np.empty((h_size, len(tickers)+3))
    points = [free] * f + [bounds[1]] * h + [bounds[0]] * l
    corner_prtfl = list(distinct_permutations(points))
    corner_prtfls = np.array(corner_prtfl)
    print(f"Shape of array: {corner_prtfls.shape}")
    print(corner_prtfls[:100])
    return corner_prtfls

def ports_setup(noc, gen=10000, bnds = bounds, size = len(tickers)):
    fill = (1-bnds[1])/(size-1)
    width = size + 3
    height = 2**size + 1000
    ports = np.empty((height, width))
    for i in range(noc):
        pass

    return ports

def vector_between(dims, point1, point2, msrx):
    vctr = []
    if len(point1) != len(point2) :
        raise ValueError("both points must have the same number of dimensions")
    for i in range(len(point1)):
        vctr.append(point1[i] - point2[i])

    return vctr

def comp_port(weights, method="sharpe"):
    pass

def gradient(port, h = 1e-5):
    grad = np.zeros_like(port)

    for i in range(len(port)):
        port_plus_h = port.copy()
        port_minus_h = port.copy()

        port_plus_h += h
        port_minus_h -= h

        f_plus_h = comp_port(port_plus_h, method="sharpe")
        f_minus_h = comp_port(port_minus_h, method="sharpe")

        grad[i] = (f_plus_h - f_minus_h) / (2 * h)

        return grad


def optimise(st_port, method="sharpe", ln_rt = LEARNING_RATE, max_iter=10000):
    """Gradient ascent on the portfolio weights until the objective stops improving."""
    history = []
    for i in range(max_iter):
        if accept_step(st_port, i, history):
            break
        grd = gradient(st_port, method = method)
        st_port = st_port + ln_rt * grd
    return st_port

def accept_step (new_point, i, history):
    """Stop when the last two accepted points are identical (no further progress)."""
    history.append(new_point)
    if i == 0 :
        return False
    currentval = history[-1]
    last2val = history[-3] if len(history) > 2 else None
    if last2val is not None and np.array_equal(last2val, currentval) :
        return True
    else :
        return False






    




find_corners(bounds, 10)







# Generate all unique permutations efficiently




