import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

CURVE = Path(__file__).parent / "data" / "market_curve.csv"
OUT = Path(__file__).parent / "outputs"

A = 0.10        # mean reversion, half-life ln(2)/a ~ 6.9 years
SIGMA = 0.01    # short-rate volatility
EXERCISE = 2.0
PAY_DATES = [3.0, 4.0, 5.0, 6.0, 7.0]
DELTAS = [1.0] * 5

trapezoid = getattr(np, "trapezoid", None) or np.trapz


class Curve:
    # Discount factors, log-linearly interpolated, with an optional parallel shift.

    def __init__(self, times, discounts, shift=0.0):
        self.times = np.asarray(times, float)
        self.logd = np.log(np.asarray(discounts, float))
        self.shift = shift

    def shifted(self, dr):
        return Curve(self.times, np.exp(self.logd), self.shift + dr)

    def df(self, t):
        if t <= 0:
            return 1.0
        if t >= self.times[-1]:
            z = -self.logd[-1] / self.times[-1]
            base = math.exp(-z * t)
        else:
            base = math.exp(np.interp(t, self.times, self.logd))
        return base * math.exp(-self.shift * t)

    def fwd(self, t, h=1e-4):
        lo, hi = max(t - h, 1e-8), t + h
        return -(math.log(self.df(hi)) - math.log(self.df(lo))) / (hi - lo)


def bootstrap(path):
    quotes = pd.read_csv(path)
    p = {0.0: 1.0}

    # Deposits are simple interest.
    for _, row in quotes.iterrows():
        t, r = float(row["maturity"]), float(row["rate"])
        if t <= 1.0:
            p[t] = 1.0 / (1.0 + r * t)

    # A par swap has zero value, so the fixed leg equals the floating one. Each step
    # only reuses discount factors already computed at shorter maturities, which is
    # why every annual maturity has to be quoted.
    for _, row in quotes.iterrows():
        if str(row["type"]).strip().lower() != "swap":
            continue
        n, s = int(row["maturity"]), float(row["rate"])
        p[float(n)] = (1.0 - s * sum(p[float(i)] for i in range(1, n))) / (1.0 + s)

    times = sorted(p)
    return Curve(times, [p[t] for t in times])


# P(t,T) = A(t,T) exp(-B(t,T) r(t))

def B(t, T):
    return (1.0 - math.exp(-A * (T - t))) / A


def A_term(curve, t, T):
    b = B(t, T)
    expo = b * curve.fwd(max(t, 1e-8)) - SIGMA**2 / (4*A) * (1 - math.exp(-2*A*t)) * b*b
    return curve.df(T) / curve.df(t) * math.exp(expo)


def bond(curve, t, T, r):
    return A_term(curve, t, T) * math.exp(-B(t, T) * r)


def zcb_put(curve, expiry, maturity, strike):
    sp = SIGMA * math.sqrt((1 - math.exp(-2*A*expiry)) / (2*A)) * B(expiry, maturity)
    ps, pt = curve.df(expiry), curve.df(maturity)
    if sp < 1e-12:
        return max(strike * ps - pt, 0.0)
    h = math.log(pt / (strike * ps)) / sp + 0.5 * sp
    return strike * ps * norm.cdf(-h + sp) - pt * norm.cdf(-h)


def alpha(curve, t):
    # Deterministic shift that pins the model to the market curve.
    return curve.fwd(max(t, 1e-8)) + SIGMA**2 / (2*A*A) * (1 - math.exp(-A*t))**2


def atm_strike(curve):
    num = curve.df(EXERCISE) - curve.df(PAY_DATES[-1])
    den = sum(d * curve.df(t) for d, t in zip(DELTAS, PAY_DATES))
    return num / den


def coupons(strike):
    return [strike * d for d in DELTAS[:-1]] + [1.0 + strike * DELTAS[-1]]


def jamshidian(curve, strike):
    # Bond prices are monotone in r(T0), so a unique r* makes the swap worth zero.
    # Exercise happens iff r(T0) < r*, which turns the payoff into a portfolio of
    # puts on zero-coupon bonds struck at P(T0, Ti, r*).
    c = coupons(strike)
    r_star = brentq(
        lambda r: sum(ci * bond(curve, EXERCISE, t, r) for ci, t in zip(c, PAY_DATES)) - 1.0,
        -0.5, 0.5,
    )
    price = sum(
        ci * zcb_put(curve, EXERCISE, t, bond(curve, EXERCISE, t, r_star))
        for ci, t in zip(c, PAY_DATES)
    )
    return price, r_star


def monte_carlo(curve, strike, n_paths=200_000, seed=42):
    # (x(T0), integral of x) is bivariate Gaussian with a covariance known in closed
    # form, so paths come from one Cholesky draw. No Euler stepping, no discretisation
    # bias: the only error left is sampling noise.
    grid = np.linspace(0.0, EXERCISE, 4001)
    det = float(trapezoid([alpha(curve, float(t)) for t in grid], grid))

    var_x = SIGMA**2 / (2*A) * (1 - math.exp(-2*A*EXERCISE))
    cov = SIGMA**2 / (2*A*A) * (1 - math.exp(-A*EXERCISE))**2
    var_i = SIGMA**2 / (A*A) * (
        EXERCISE - 2*(1 - math.exp(-A*EXERCISE))/A + (1 - math.exp(-2*A*EXERCISE))/(2*A)
    )

    chol = np.linalg.cholesky([[var_x, cov], [cov, var_i]])
    x, integral = chol @ np.random.default_rng(seed).standard_normal((2, n_paths))

    r = alpha(curve, EXERCISE) + x
    bonds = np.column_stack([
        A_term(curve, EXERCISE, T) * np.exp(-B(EXERCISE, T) * r) for T in PAY_DATES
    ])
    pv = np.exp(-(det + integral)) * np.maximum(1.0 - bonds @ np.array(coupons(strike)), 0.0)
    return float(pv.mean()), float(pv.std(ddof=1) / math.sqrt(n_paths))


def greeks(curve, strike, h=1e-4):
    # Centred differences on the closed form, so simulation noise stays out of them.
    p0 = jamshidian(curve, strike)[0]
    up = jamshidian(curve.shifted(h), strike)[0]
    dn = jamshidian(curve.shifted(-h), strike)[0]
    return (up - dn) / (2*h) * 1e-4, (up - 2*p0 + dn) / (h*h) * 1e-8


def scenarios(curve, strike):
    rows = []
    for bp in (-50, 0, 50, 100):
        c = curve.shifted(bp / 10000)
        semi = jamshidian(c, strike)[0]
        mc, se = monte_carlo(c, strike, n_paths=100_000, seed=123)
        rows.append((bp, atm_strike(c), semi, mc, se))
    return rows


def parameter_sweep(curve, strike):
    # A and SIGMA are module-level, so they get set and restored around the sweep
    # instead of being threaded through every pricing function.
    global A, SIGMA
    a0, s0 = A, SIGMA
    a_range = np.linspace(0.05, 0.20, 10)
    s_range = np.linspace(0.005, 0.020, 10)
    try:
        prices_a = []
        for A in a_range:
            prices_a.append(jamshidian(curve, strike)[0])
        A = a0
        prices_s = []
        for SIGMA in s_range:
            prices_s.append(jamshidian(curve, strike)[0])
    finally:
        A, SIGMA = a0, s0
    return a_range, prices_a, s_range, prices_s


def plots(curve, rows, sweep):
    OUT.mkdir(exist_ok=True)

    ts = np.linspace(0.25, 30, 200)
    plt.figure(figsize=(8, 5))
    plt.plot(ts, [-math.log(curve.df(t)) / t * 100 for t in ts], lw=2)
    plt.xlabel("Maturity (years)")
    plt.ylabel("Zero rate (%, continuously compounded)")
    plt.title("Bootstrapped zero-coupon curve")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "zero_curve.png", dpi=160)
    plt.close()

    bp = [r[0] for r in rows]
    plt.figure(figsize=(8, 5))
    plt.plot(bp, [r[2] for r in rows], lw=2, label="Jamshidian (closed form)")
    plt.errorbar(bp, [r[3] for r in rows], yerr=[1.96 * r[4] for r in rows],
                 fmt="o", capsize=3, color="crimson", label="Monte Carlo (95% CI)")
    plt.xlabel("Parallel shift (bp)")
    plt.ylabel("Swaption price")
    plt.title("Price vs parallel curve shift")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "shift_sensitivity.png", dpi=160)
    plt.close()

    a_range, prices_a, s_range, prices_s = sweep
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(a_range, prices_a, "o-", lw=2)
    ax1.set(xlabel="Mean reversion a", ylabel="Swaption price",
            title="Sensitivity to mean reversion")
    ax1.grid(alpha=0.3)
    ax2.plot(s_range, prices_s, "o-", lw=2, color="green")
    ax2.set(xlabel="Volatility sigma", ylabel="Swaption price",
            title="Sensitivity to volatility")
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "parameter_sensitivity.png", dpi=160)
    plt.close()


def main():
    curve = bootstrap(CURVE)
    k = atm_strike(curve)
    semi, r_star = jamshidian(curve, k)
    mc, se = monte_carlo(curve, k)
    delta, gamma = greeks(curve, k)
    rows = scenarios(curve, k)
    sweep = parameter_sweep(curve, k)

    print(f"2Y x 5Y European ATM payer swaption, a = {A:.0%}, sigma = {SIGMA:.0%}\n")
    print(f"ATM forward swap rate   {k:.4%}")
    print(f"Jamshidian              {semi:.6f}")
    print(f"Monte Carlo             {mc:.6f}   (200,000 paths, se {se:.6f})")
    print(f"Critical rate r*        {r_star:.4%}")
    print(f"Difference              {abs(mc - semi):.6f}   = {abs(mc - semi)/se:.2f} se\n")
    print(f"Delta  {delta:.4e} per +1bp")
    print(f"Gamma  {gamma:.4e} per 1bp squared\n")

    print(f"{'Shift':>7} {'Fwd swap':>10} {'Jamshidian':>12} {'MC':>11} {'95% CI':>10}")
    for bp, fwd, s, m, e in rows:
        print(f"{bp:>5}bp {fwd:>9.4%} {s:>12.6f} {m:>11.6f} {1.96*e:>10.6f}")

    a_range, pa, s_range, ps = sweep
    print(f"\nMean reversion {a_range[0]:.2f} to {a_range[-1]:.2f}: "
          f"price {pa[0]:.6f} to {pa[-1]:.6f}")
    print(f"Volatility {s_range[0]:.3f} to {s_range[-1]:.3f}: "
          f"price {ps[0]:.6f} to {ps[-1]:.6f}")

    plots(curve, rows, sweep)
    print(f"\nFigures written to {OUT}/")


if __name__ == "__main__":
    main()
