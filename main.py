"""GTA VI launch-day Twitch CCU forecaster (gta6-twitch-ccu-forecast-01).

Method (chosen by measured backtest, see BUILD-LOG.md in the work folder):
per-hour OLS in log space, ln(viewers_h) ~ a_h + b_h * ln(hour-1 print),
fit on the case's own comparables — the past launches' release-day curves.
Two fits are blended in log space: 70% on the rolling-midnight pool
(launches whose hour-1 print is already >= 10% of their pre-midnight wave
peak and >= 10,000 — the regime GTA VI is in) and 30% on the full pool
(a hedge for a deep-trough final, when the early wave is small and the US
wave dominates). Quantile spreads come from leave-one-out residuals inside
each pool: per-hour empirical quantiles blended 50/50 with a normal whose
sigma is the residual IQR / 1.349, then narrowed 20% toward the median
(the LOO residuals overstate the spread). Everything is computed at runtime
from the case file; no hardcoded curves.

Local practice result (p00-p03, SullyGnome answers): mean pinball loss
0.2030 vs house baseline 0.3871 (-47.6%); per case 0.2275 / 0.2202 /
0.1709 / 0.1933. Leave-one-out over 32 rolling-like past launches: 0.2638
vs baseline 0.3554.

Python 3.13, standard library only, offline, deterministic, no network.
Reads every <id>.in in the input directory and writes <id>.out:
{"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]}.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

# Blend weight on the rolling-pool fit; spread shrink and scale (measured).
W_ROLL = 0.7
SPREAD_SHRINK = 0.5
SPREAD_SCALE = 0.8
# A pool hour with fewer than this many donors falls back to persistence.
MIN_DONORS = 8
# A launch needs at least this many scored hours to join the pool.
MIN_HOURS = 20


def round2(x):
    """x rounded to 2 decimals, half up (as the starter rounds it)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def parse_ts(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp() / 3600.0


def ols(X, y):
    """Least squares with intercept; X is a list of feature vectors."""
    n = len(X)
    p = len(X[0]) + 1
    A = [[1.0] + list(x) for x in X]
    M = [[sum(A[k][i] * A[k][j] for k in range(n)) for j in range(p)] for i in range(p)]
    b = [sum(A[k][i] * y[k] for k in range(n)) for i in range(p)]
    for i in range(p):
        M[i][i] += 1e-8
    for i in range(p):
        piv = max(range(i, p), key=lambda r: abs(M[r][i]))
        M[i], M[piv] = M[piv], M[i]
        b[i], b[piv] = b[piv], b[i]
        for r in range(i + 1, p):
            f = M[r][i] / M[i][i]
            for c in range(i, p):
                M[r][c] -= f * M[i][c]
            b[r] -= f * b[i]
    beta = [0.0] * p
    for i in range(p - 1, -1, -1):
        beta[i] = (b[i] - sum(M[i][j] * beta[j] for j in range(i + 1, p))) / M[i][i]
    return beta


def predict_ols(beta, x):
    return beta[0] + sum(b * v for b, v in zip(beta[1:], x))


def quantile(sorted_vals, tau):
    n = len(sorted_vals)
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def spread_offsets(residuals):
    """Log-space quantile offsets from LOO residuals: empirical quantiles
    blended 50/50 with a normal of the same IQR. No scaling here: the 0.8
    narrowing is applied later to the BLENDED forecast, centred on its own
    median (so the median itself is untouched by the narrowing)."""
    if len(residuals) < MIN_DONORS:
        return [z * 0.35 for z in Z]
    s = sorted(residuals)
    emp = [quantile(s, t) for t in LEVELS]
    n = len(s)
    med = s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
    q25, q75 = quantile(s, 0.25), quantile(s, 0.75)
    sig = max(0.05, (q75 - q25) / 1.349)
    out = []
    for i in range(5):
        e = emp[i]
        n_ = med + Z[i] * sig
        out.append(SPREAD_SHRINK * e + (1 - SPREAD_SHRINK) * n_)
    for i in range(1, 5):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def build_pool(comparables):
    """Each comparable becomes a donor: hour-1 print, release-day curve
    (hours 2..24), and a rolling-midnight flag from its pre-midnight wave."""
    pool = []
    for comp in comparables:
        try:
            ds = parse_ts(comp["day_start"])
        except (KeyError, ValueError):
            continue
        hourly = {}
        for row in comp.get("hourly") or []:
            try:
                v = row["viewers"]
                if v is None or v <= 0:
                    continue
                off = int(round(parse_ts(row["from"]) - ds))
                hourly[off] = float(v)
            except (KeyError, ValueError, TypeError):
                continue
        if 0 not in hourly:
            continue
        p1 = hourly[0]
        curve = {h: hourly[h - 1] for h in range(2, 25) if (h - 1) in hourly}
        if len(curve) < MIN_HOURS or p1 <= 0:
            continue
        pre = [v for o, v in hourly.items() if -24 <= o < 0]
        pre_max = max(pre) if pre else None
        rolling = bool(pre_max and p1 >= 0.10 * pre_max and p1 >= 10000)
        pool.append({"name": comp.get("name", ""), "p1": p1, "curve": curve, "rolling": rolling})
    pool.sort(key=lambda d: d["name"])
    return pool


def ols_forecast(items, lp1):
    """Per-hour OLS ln(y_h) ~ ln(p1) over the given donors; returns
    {hour: [q05, q25, q50, q75, q95]} with LOO-residual spreads."""
    out = {}
    for h in range(2, 25):
        X, y = [], []
        for it in items:
            if h in it["curve"]:
                X.append([math.log(it["p1"])])
                y.append(math.log(it["curve"][h]))
        if len(X) < MIN_DONORS:
            out[h] = [math.exp(lp1 + z * 0.35) for z in Z]
            continue
        beta = ols(X, y)
        mu = predict_ols(beta, [lp1])
        res = []
        for i in range(len(X)):
            b2 = ols(X[:i] + X[i + 1:], y[:i] + y[i + 1:])
            res.append(y[i] - predict_ols(b2, X[i]))
        spr = spread_offsets(res)
        out[h] = [math.exp(mu + s) for s in spr]
    return out


def baseline_row(p1, h):
    return [round2(max(1.0, p1) * math.exp(z * 0.15 * math.sqrt(h - 1))) for z in Z]


def forecast_case(case, name):
    first = next((k for k in case.get("known", []) if k.get("hour") == 1), None)
    if first is not None and first.get("viewers"):
        p1 = max(1.0, float(first["viewers"]))
    else:
        hist = [r for r in (case.get("history_hourly") or []) if r.get("viewers")]
        p1 = max(1.0, float(hist[-1]["viewers"])) if hist else 1000.0
    lp1 = math.log(p1)
    pool = build_pool(case.get("comparables") or [])
    rolling_pool = [it for it in pool if it["rolling"]]
    if len(rolling_pool) < MIN_DONORS:
        rolling_pool = pool
    if len(pool) < MIN_DONORS:
        return {"hours": [{"hour": h, "q": baseline_row(p1, h)} for h in range(2, 25)]}
    fit_roll = ols_forecast(rolling_pool, lp1)
    fit_all = ols_forecast(pool, lp1)
    hours = []
    for h in range(2, 25):
        blended = []
        for i in range(5):
            v = math.exp(W_ROLL * math.log(max(1.0, fit_roll[h][i]))
                         + (1 - W_ROLL) * math.log(max(1.0, fit_all[h][i])))
            blended.append(v)
        # Narrow the spread 20% toward the blended median (median untouched).
        mu = math.log(blended[2])
        qs = []
        for v in blended:
            q = math.exp(mu + (math.log(v) - mu) * SPREAD_SCALE)
            qs.append(round2(min(1e9, max(1.0, q))))
        for i in range(1, 5):
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        hours.append({"hour": h, "q": qs})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
