import argparse, json, sys, math, os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import math

# ---------- Config ----------
DEFAULT_WINDOW = 250       # trading days for vol/corr estimation (~1y)
DEFAULT_SCENARIOS = 10000  # Monte Carlo paths
DEFAULT_ALPHA = 0.95       # VaR/CVaR confidence

# ---------- Data structures ----------
@dataclass
class Position:
    symbol: str
    qty: float
    price: float  # if NaN, we'll fetch last price
    multiplier: float
# --- Backtest diagnostics: Kupiec POF, Christoffersen, Basel ---
try:
    from scipy.stats import chi2
    chi2_sf = lambda x, df: chi2.sf(x, df)   # survival function = 1 - CDF
except Exception:
    # SciPy missing? crude fallback so we still return something
    chi2_sf = lambda x, df: math.exp(-x/2)

def kupiec_pof(n, x, alpha):
    """
    Proportion-of-failures (Kupiec) test.
    n = #observations, x = #breaches, alpha = VaR confidence (e.g. 0.95)
    Returns (LR, p_value).
    """
    if n == 0: 
        return float("nan"), float("nan")
    p = 1 - alpha
    if x == 0 or x == n:
        # handle log(0) edge cases by tiny jitter
        x = max(1e-12, min(n - 1e-12, x))
    LR = -2 * (
        (n - x) * math.log(1 - p) + x * math.log(p)
        - ((n - x) * math.log(1 - x / n) + x * math.log(x / n))
    )
    return LR, chi2_sf(LR, 1)

def christoffersen_independence(hits):
    """
    hits: list/array of 0/1 where 1 = breach.
    Tests independence of exceptions (Markov test).
    Returns (LR, p_value).
    """
    if len(hits) < 2:
        return float("nan"), float("nan")
    n00 = n01 = n10 = n11 = 0
    for i in range(1, len(hits)):
        a, b = hits[i-1], hits[i]
        if a == 0 and b == 0: n00 += 1
        elif a == 0 and b == 1: n01 += 1
        elif a == 1 and b == 0: n10 += 1
        else: n11 += 1
    # transition probabilities
    denom0 = n00 + n01
    denom1 = n10 + n11
    if denom0 == 0 or denom1 == 0:
        return float("nan"), float("nan")
    pi01 = n01 / denom0
    pi11 = n11 / denom1
    pi   = (n01 + n11) / (denom0 + denom1)
    # Likelihood ratio
    def _s(x): 
        return 0 if x <= 0 else x
    LR = -2 * (
        _s(n00)*math.log(1 - pi) + _s(n01)*math.log(pi) + _s(n10)*math.log(1 - pi) + _s(n11)*math.log(pi)
        - (_s(n00)*math.log(1 - pi01) + _s(n01)*math.log(pi01) + _s(n10)*math.log(1 - pi11) + _s(n11)*math.log(pi11))
    )
    return LR, chi2_sf(LR, 1)

def basel_traffic_light_99(n, x):
    """
    Basel traffic-light zones for 99% VaR with ~250 observations.
    Returns 'green'/'yellow'/'red' or 'N/A' if not applicable.
    """
    if not (0.985 <= 0.99 <= 0.995):  # silly guard, keep readable
        return "N/A"
    # Basel’s canonical bands for 250 obs @ 99%:
    if 200 <= n <= 260:
        if x <= 4:  return "green"
        if x <= 9:  return "yellow"
        return "red"
    return "N/A"


# ---------- Helpers ----------
def cov_to_corr(cov):
    """Convert a covariance matrix to a correlation matrix."""
    stddev = np.sqrt(np.diag(cov))
    corr = cov / np.outer(stddev, stddev)
    corr[corr > 1] = 1  # avoid floating-point overflow >1
    corr[corr < -1] = -1
    return corr
def read_portfolio(csv_path: str) -> List[Position]:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "symbol" not in df.columns or "qty" not in df.columns:
        raise ValueError("portfolio.csv must have columns: symbol, qty [, price] [, multiplier]")
    if "multiplier" not in df.columns:
        df["multiplier"] = 1.0

    out: List[Position] = []
    for _, r in df.iterrows():
        symbol = str(r["symbol"]).strip()
        if not symbol:
            continue
        qty = float(r["qty"])
        price = (
            float(r["price"])
            if "price" in df.columns and pd.notna(r["price"]) and str(r["price"]).strip() != ""
            else np.nan
        )
        mult = float(r["multiplier"]) if pd.notna(r["multiplier"]) else 1.0
        out.append(Position(symbol=symbol, qty=qty, price=price, multiplier=mult))
    return out
def fetch_prices(symbols: List[str], window: int) -> pd.DataFrame:
    print(f"[2/6] Downloading prices for: {symbols}")
    # Pull more history than we need; avoid MultiIndex issues
    data = yf.download(
        symbols, period="2y", interval="1d", auto_adjust=True,
        progress=False, group_by="ticker", threads=False
    )

    # Normalize to a flat DataFrame with columns = symbols
    if isinstance(data.columns, pd.MultiIndex):
        close_cols = []
        for s in symbols:
            # Yahoo sometimes omits a ticker; guard it
            try:
                ser = data[(s, "Close")].astype(float)
                ser.name = s
                close_cols.append(ser)
            except Exception:
                pass
        prices = pd.concat(close_cols, axis=1)
    else:
        # Single symbol case comes as a Series or DataFrame with 'Close'
        if "Close" in data:
            prices = data["Close"].to_frame()
        else:
            prices = data.to_frame(name=symbols[0])

    prices = prices.dropna(how="all").ffill()
    prices = prices.tail(window + 30)      # keep last window with buffer
    prices = prices[symbols].dropna(axis=1, how="all")  # drop symbols with no data

    if prices.empty or prices.shape[1] == 0:
        print("ERROR: No price data returned. Check symbols or internet access.")
        return pd.DataFrame()
    print(f"[2/6] Prices fetched. Shape={prices.shape}")
    return prices

def last_prices(prices: pd.DataFrame) -> pd.Series:
    return prices.iloc[-1]

def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()

def ensure_pos_def(Sigma: np.ndarray, jitter: float = 1e-8) -> np.ndarray:
    # make covariance numerically positive-definite if needed
    try:
        np.linalg.cholesky(Sigma)
        return Sigma
    except np.linalg.LinAlgError:
        # eigenvalue floor
        vals, vecs = np.linalg.eigh(Sigma)
        vals[vals < jitter] = jitter
        return (vecs @ np.diag(vals) @ vecs.T)

def simulate_pnl_normal(positions: List[Position], prices_now: pd.Series,
                 mu: np.ndarray, Sigma: np.ndarray,
                 scenarios: int) -> np.ndarray:
    n = len(positions)
    L = np.linalg.cholesky(Sigma)
    Z = np.random.normal(size=(n, scenarios))       # iid standard normals
    R = (mu.reshape(-1,1) + L @ Z)                  # correlated returns
    # Compute P&L = sum_i qty * price * multiplier * r_i
    qty = np.array([p.qty for p in positions]).reshape(-1,1)
    px  = np.array([prices_now[p.symbol] for p in positions]).reshape(-1,1)
    mult= np.array([p.multiplier for p in positions]).reshape(-1,1)
    pnl = (qty * px * mult * R).sum(axis=0)         # shape (scenarios,)
    return pnl
def simulate_pnl_t(positions: List[Position], prices_now: pd.Series,
                   mu: np.ndarray, Sigma: np.ndarray,
                   scenarios: int, df: int = 7) -> np.ndarray:
    """
    Multivariate t via normal / chi-square mixture:
    R = mu + (L @ Z) / sqrt(U/df),  Z ~ N(0,I), U ~ chi2(df)
    Approx reproduces fat tails with correlation preserved by L.
    """
    n = len(positions)
    L = np.linalg.cholesky(Sigma)
    Z = np.random.normal(size=(n, scenarios))
    U = np.random.chisquare(df, size=(1, scenarios))  # shape (1, scenarios)
    scale = np.sqrt(U / df)                           # same for all assets per scenario
    R = (mu.reshape(-1,1) + (L @ Z) / scale)

    qty = np.array([p.qty for p in positions]).reshape(-1,1)
    px  = np.array([prices_now[p.symbol] for p in positions]).reshape(-1,1)
    mult= np.array([p.multiplier for p in positions]).reshape(-1,1)
    pnl = (qty * px * mult * R).sum(axis=0)
    return pnl
def historical_horizon_returns(rets_df: pd.DataFrame, h: int) -> np.ndarray:
    """
    Build empirical distribution of h-day returns by summing log-returns
    over rolling windows (no overlap drop). Returns array shape (n_assets, N)
    """
    if h <= 1:
        return rets_df.values.T  # shape (n_assets, T)
    # Rolling sum of log-returns approximates multi-day log-return
    roll = rets_df.rolling(window=h).sum().dropna()
    return roll.values.T

def simulate_pnl_historical(positions: List[Position], prices_now: pd.Series,
                            rets_df: pd.DataFrame, scenarios: int, horizon: int = 1) -> np.ndarray:
    # Build empirical h-day return vectors
    R_emp = historical_horizon_returns(rets_df, horizon)  # shape (n_assets, T_eff)
    if R_emp.size == 0:
        return np.array([])
    n_assets, T_eff = R_emp.shape
    idx = np.random.randint(0, T_eff, size=scenarios)     # bootstrap with replacement
    R = R_emp[:, idx]                                     # shape (n_assets, scenarios)

    qty = np.array([p.qty for p in positions]).reshape(-1,1)
    px  = np.array([prices_now[p.symbol] for p in positions]).reshape(-1,1)
    mult= np.array([p.multiplier for p in positions]).reshape(-1,1)
    pnl = (qty * px * mult * R).sum(axis=0)
    return pnl


def var_cvar(losses: np.ndarray, alpha: float) -> Tuple[float, float]:
    # losses = -pnl (positive is bad)
    q = np.quantile(losses, alpha)  # e.g., alpha=0.95 -> 95th percentile loss
    tail = losses[losses >= q]
    cvar = tail.mean() if tail.size > 0 else q
    return float(q), float(cvar)

def stress_table(positions: List[Position], prices_now: pd.Series,
                 vol: np.ndarray, sigmas: List[int] = [1,2,3]) -> pd.DataFrame:
    qty = np.array([p.qty for p in positions]).reshape(-1,1)
    px  = np.array([prices_now[p.symbol] for p in positions]).reshape(-1,1)
    mult= np.array([p.multiplier for p in positions]).reshape(-1,1)
    rows = []
    for k in sigmas:
        shock = (-k * vol).reshape(-1,1)            # all assets down kσ
        pnl   = (qty * px * mult * shock).sum()
        rows.append({"Shock": f"-{k}σ all assets", "P&L": float(pnl)})
    return pd.DataFrame(rows)

def plot_hist(pnl: np.ndarray, alpha_levels: List[float], out_path: str, kde: bool = False):
    losses = -pnl
    try:
        from scipy.stats import gaussian_kde
        have_scipy = True
    except Exception:
        have_scipy = False
        kde = False

    plt.figure(figsize=(9,5))
    # Histogram (density=True so KDE overlays properly)
    n, bins, _ = plt.hist(pnl, bins=80, density=True, alpha=0.5, label="P&L histogram")

    # Optional KDE
    if kde and have_scipy and pnl.size > 10:
        xs = np.linspace(pnl.min(), pnl.max(), 600)
        kde_est = gaussian_kde(pnl)
        plt.plot(xs, kde_est(xs), label="KDE")

    # VaR lines (colored)
    for a, color in zip(alpha_levels, ["tab:red", "tab:purple"]):
        q_loss = np.quantile(losses, a)
        x = -q_loss
        plt.axvline(x=x, linestyle="--", color=color, label=f"VaR {int(a*100)}% = {x:,.0f}")

    plt.title("Simulated Portfolio P&L")
    plt.xlabel("P&L")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
def scale_for_horizon(mu: np.ndarray, Sigma: np.ndarray, h: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    For Gaussian (and approx for t), scale mean and covariance for h-day horizon:
    mu_h = h * mu,  Sigma_h = h * Sigma
    """
    h = max(1, int(h))
    return mu * h, Sigma * h
def horizon_sweep(positions, px_now, mu, Sigma, rets_df,
                  mode, df, scenarios, horizons, kde=False, make_plots=False,
                  liq_loss_abs: float = 0.0):
    results = []
    for h in horizons:
        pnl = simulate_pnl_for_mode(positions, px_now, mu, Sigma, rets_df,
                                    mode=mode, df=df, scenarios=scenarios, horizon=h)
        if pnl.size == 0:
            continue
        if liq_loss_abs:
            pnl = pnl - liq_loss_abs

        losses = -pnl
        var95, cvar95 = var_cvar(losses, 0.95)
        var99, cvar99 = var_cvar(losses, 0.99)

        if make_plots:
            out_path = f"outputs/pnl_hist_{h}d_{mode}.png"
            plot_hist(pnl, [0.95, 0.99], out_path, kde=kde)

        results.append({
            "horizon_days": h,
            "VaR95": var95, "CVaR95": cvar95,
            "VaR99": var99, "CVaR99": cvar99
        })
    return results

def backtest_var(positions, prices_df, alpha=0.95, window=250):
    """
    Rolling 1-day VaR backtest (Gaussian MC).
    Returns: (hit_rate, n_obs, breaches, hits_list)
    """
    rets = log_returns(prices_df)
    n_rets = len(rets)
    if n_rets < 2:
        return 0.0, 0, 0, []

    win = min(window, n_rets - 1)
    if win < 10:
        return 0.0, 0, 0, []

    qty  = np.array([p.qty for p in positions]).reshape(-1, 1)
    mult = np.array([p.multiplier for p in positions]).reshape(-1, 1)

    hits = []

    for t in range(win, n_rets):
        rets_window = rets.iloc[t - win:t]
        Sigma   = ensure_pos_def(rets_window.cov().values)

        today_r  = rets.iloc[t].values.reshape(-1, 1)
        px_tminus1 = prices_df.iloc[t - 1].values.reshape(-1, 1)
        actual_pnl = float((qty * px_tminus1 * mult * today_r).sum())

        pnl_sim = simulate_pnl_normal(
            positions, prices_df.iloc[t - 1],
            mu=np.zeros(Sigma.shape[0]), Sigma=Sigma, scenarios=5000
        )
        var_alpha, _ = var_cvar(-pnl_sim, alpha)
        hit = 1 if actual_pnl < -var_alpha else 0
        hits.append(hit)

    n = len(hits)
    breaches = sum(hits)
    hit_rate = (breaches / n) if n > 0 else 0.0
    return hit_rate, n, breaches, hits

def var_ci_from_losses(losses: np.ndarray, alpha: float = 0.95, level: int = 90, boot: int = 400):
    """
    Bootstrap CI for VaR (on loss distribution). Returns (lo, hi) for the VaR.
    """
    n = losses.size
    if n == 0:
        return float("nan"), float("nan")
    qs = []
    for _ in range(boot):
        idx = np.random.randint(0, n, size=n)   # uses np.random.seed if set
        bs = losses[idx]
        qs.append(np.quantile(bs, alpha))
    lo = np.percentile(qs, (100 - level) / 2)
    hi = np.percentile(qs, 100 - (100 - level) / 2)
    return float(lo), float(hi)

def simulate_pnl_for_mode(positions, px_now, mu, Sigma, rets_df,
                          mode: str, df: int, scenarios: int, horizon: int):
    """Return simulated P&L array for the given mode and horizon."""
    if mode == "historical":
        return simulate_pnl_historical(positions, px_now, rets_df, scenarios, horizon=horizon)
    else:
        mu_h, Sigma_h = scale_for_horizon(mu=mu, Sigma=Sigma, h=horizon)
        if mode == "normal":
            return simulate_pnl_normal(positions, px_now, mu=mu_h, Sigma=Sigma_h, scenarios=scenarios)
        elif mode == "t":
            return simulate_pnl_t(positions, px_now, mu=mu_h, Sigma=Sigma_h, scenarios=scenarios, df=df)
        else:
            raise ValueError("Unknown mode")

def fetch_price_data(tickers, window_days, extra_buffer_days=40):
    """
    Download adjusted close prices for the given tickers and return a clean
    wide DataFrame aligned on common trading dates (last `window_days` rows).
    """
    series = []
    for t in tickers:
        # buffer a few extra days to survive holidays / missing bars
        df = yf.download(
            t,
            period=f"{window_days + extra_buffer_days}d",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if df.empty or "Close" not in df:
            raise RuntimeError(f"Could not download prices for {t}")
        s = df["Close"].astype(float)
        s.name = t
        series.append(s) 

    # Inner join on dates all tickers have
    prices = pd.concat(series, axis=1, join="inner").sort_index()

    # Keep only the last `window_days` observations
    return prices.tail(window_days)

def main():
    # -------- 1) Parse args --------
    parser = argparse.ArgumentParser(description="Monte Carlo VaR Engine")
    parser.add_argument("--portfolio", default="portfolio.csv")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--scenarios", type=int, default=DEFAULT_SCENARIOS)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--mode", choices=["normal", "t", "historical"], default="normal")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--df", type=int, default=7)
    parser.add_argument("--kde", action="store_true")
    parser.add_argument("--rho_stress", type=float, default=0.0)
    parser.add_argument("--drift", default="0")  # auto | 0 | <annual %>
    parser.add_argument("--seed", type=int, default=None,
        help="Random seed for reproducibility (e.g., 42)")
    parser.add_argument("--liq_haircut_bp", type=float, default=0.0,
        help="Liquidity/slippage haircut in basis points of portfolio notional (e.g., 20 = 0.20%)")
    parser.add_argument("--ci", type=int, default=None,
        help="Bootstrap CI level for VaR (e.g., 90 for 90% interval)")

    args = parser.parse_args()
    if args.seed is not None:
      np.random.seed(args.seed)


    os.makedirs("outputs", exist_ok=True)

    # -------- 2) Portfolio & prices --------
    positions = read_portfolio(args.portfolio)          # List[Position]
    symbols   = [p.symbol for p in positions]
    print(f"[1/6] Portfolio loaded. Symbols: {symbols}")

    prices = fetch_price_data(symbols, args.window)     # DataFrame (cols = symbols)
    if prices.empty:
        print("ERROR: no price data; check symbols/internet")
        sys.exit(1)
    returns = prices.pct_change().dropna()
    print(f"[3/6] Computed log returns. Shape={returns.shape}")

    # -------- 3) Covariance with correlation stress --------
    Sigma_hist = returns.cov().values
    R_hist  = cov_to_corr(Sigma_hist)
    R_one   = np.ones_like(R_hist)
    R_stress = (1.0 - args.rho_stress) * R_hist + args.rho_stress * R_one
    np.fill_diagonal(R_stress, 1.0)
    std = np.sqrt(np.diag(Sigma_hist))
    Sigma = ensure_pos_def(np.outer(std, std) * R_stress)

    # -------- 4) Drift (daily) --------
    if args.drift == "auto":
        mu_daily = returns.mean().values
    elif args.drift == "0":
        mu_daily = np.zeros(len(symbols))
    else:
        mu_daily = np.full(len(symbols), float(args.drift)/100.0/252.0)

    # -------- 5) Current prices & fill overrides --------
    px_now = last_prices(prices)
    for p in positions:
        if np.isnan(p.price):
            p.price = float(px_now[p.symbol])
    # ---- Portfolio notional and liquidity haircut ----
    total_value = sum(p.qty * p.price * p.multiplier for p in positions)
    liq_loss_abs = (args.liq_haircut_bp / 10000.0) * total_value  # absolute haircut in currency


    # -------- 6) Simulate for requested mode/horizon --------
    mode = args.mode
    horizon = max(1, args.horizon)
    print(f"[5/6] Mode={mode}, horizon={horizon} day(s), scenarios={args.scenarios}")

    if mode == "historical":
        pnl = simulate_pnl_historical(positions, px_now, returns,
                                      scenarios=args.scenarios, horizon=horizon)
        if pnl.size == 0:
            print("ERROR: Historical simulation failed (not enough data)."); sys.exit(1)
    else:
        # use zero mean unless you want to center by mu_daily
        mu_h, Sigma_h = scale_for_horizon(mu=mu_daily, Sigma=Sigma, h=horizon)
        if mode == "normal":
            pnl = simulate_pnl_normal(positions, px_now, mu=mu_h, Sigma=Sigma_h, scenarios=args.scenarios)
        else:  # t
            pnl = simulate_pnl_t(positions, px_now, mu=mu_h, Sigma=Sigma_h, scenarios=args.scenarios, df=args.df)

    print("[6/6] Simulation complete. Computing VaR/CVaR & saving outputs...")

    # -------- 7) Metrics, stress, outputs --------
    pnl_used = pnl - liq_loss_abs if args.liq_haircut_bp else pnl  # apply haircut once per evaluation
    losses   = -pnl_used
    var95, cvar95 = var_cvar(losses, 0.95)
    var99, cvar99 = var_cvar(losses, 0.99)

    # Optional VaR CIs
    if args.ci:
        lo95, hi95 = var_ci_from_losses(losses, 0.95, args.ci, boot=400)
        lo99, hi99 = var_ci_from_losses(losses, 0.99, args.ci, boot=400)
    vol = np.sqrt(np.diag(Sigma))
    stress = stress_table(positions, px_now, vol)

    total_value = sum(p.qty * p.price * p.multiplier for p in positions)

    main_plot_path = f"outputs/pnl_hist_{args.horizon}d_{mode}.png"
    plot_hist(pnl, [0.95, 0.99], main_plot_path, kde=args.kde)

    # Multi-horizon sweep (uses same Sigma/mu)
    horizons = [1, 5, 10, 20]
    sweep_results = horizon_sweep(
    positions, px_now, mu_daily, Sigma, returns,
    mode, args.df, args.scenarios, [1, 5, 10, 20],
    kde=args.kde, make_plots=True, liq_loss_abs=liq_loss_abs
)
    print("\n=== Multi-horizon VaR/CVaR ===")
    print(f"{'Horizon':>8} {'VaR95':>10} {'CVaR95':>10} {'VaR99':>10} {'CVaR99':>10}")
    for row in sweep_results:
        print(f"{row['horizon_days']:>8} {row['VaR95']:>10.0f} {row['CVaR95']:>10.0f} "
              f"{row['VaR99']:>10.0f} {row['CVaR99']:>10.0f}")
    # ---- Build results dict BEFORE backtest ----
    total_value = sum(p.qty * p.price * p.multiplier for p in positions)

    results = {
        "portfolio_value": total_value,
        "params": {"window_days": args.window, "scenarios": args.scenarios},
        "flags": {
            "mode": args.mode, "df": args.df, "horizon": args.horizon,
            "rho_stress": args.rho_stress, "drift": args.drift,
            "seed": args.seed, "ci_level": args.ci, "liq_haircut_bp": args.liq_haircut_bp
        },
        "VaR": {"95": var95, "99": var99},
        "CVaR": {"95": cvar95, "99": cvar99},
        "as_pct_of_portfolio": {
            "VaR95_pct": (var95 / total_value * 100) if total_value else None,
            "VaR99_pct": (var99 / total_value * 100) if total_value else None,
        },
        "current_prices": {sym: float(px_now[sym]) for sym in symbols},
        "daily_volatility": {sym: float(v) for sym, v in zip(symbols, vol)},
        "liquidity": {"haircut_bp": args.liq_haircut_bp, "haircut_abs": liq_loss_abs},
        "stress_all_down": stress.to_dict(orient="records"),
        "multi_horizon": sweep_results,
    }
    if args.ci:
        results["VaR_CI"] = {"95": [lo95, hi95], "99": [lo99, hi99]}

    print("\n=== Backtest (1-day, Gaussian) ===")
    hit_rate, n_obs, breaches, hits = backtest_var(
        positions, prices, alpha=args.alpha, window=args.window
    )
    print(f"Observed {breaches} breaches out of {n_obs} days -> hit rate = {hit_rate*100:.2f}% "
        f"(expected {(1-args.alpha)*100:.2f}%)")

    LR_pof, p_pof = kupiec_pof(n_obs, breaches, args.alpha)
    LR_ind, p_ind = christoffersen_independence(hits)
    zone = basel_traffic_light_99(n_obs, breaches) if abs(args.alpha - 0.99) < 1e-9 else "N/A (Basel uses 99%)"

    print(f"Kupiec POF: LR={LR_pof:.3f}, p-value={p_pof:.3f}")
    print(f"Christoffersen Independence: LR={LR_ind:.3f}, p-value={p_ind:.3f}")
    print(f"Basel traffic light (99% only): {zone}")

    results["backtest"] = {
        "alpha": args.alpha,
        "n_obs": n_obs,
        "breaches": breaches,
        "hit_rate_pct": hit_rate * 100,
        "kupiec": {"LR": LR_pof, "p_value": p_pof},
        "christoffersen": {"LR": LR_ind, "p_value": p_ind},
        "basel_zone_99": zone,
    }


    # Finally write JSON
    with open("outputs/results.json", "w") as f:
        json.dump(results, f, indent=2)


    # Console summary
    print("\n=== Portfolio ===")
    for p in positions:
        print(f"{p.symbol:12s} qty={p.qty:8.2f} price={p.price:10.2f} mult={p.multiplier:4.0f}")
    print(f"Total value: {total_value:,.2f}")

    print("\n=== Risk (1-day) ===")
    print(f"VaR 95% : {var95:,.0f}  ({(var95/total_value*100):.2f}% of portfolio)")
    print(f"CVaR95 : {cvar95:,.0f}")
    print(f"VaR 99% : {var99:,.0f}  ({(var99/total_value*100):.2f}% of portfolio)")
    print(f"CVaR99 : {cvar99:,.0f}")

    print("\nSaved: outputs/results.json and plots in outputs/")

if __name__ == "__main__":
    # np.random.seed(42)  # comment out for different samples each run
    main()
