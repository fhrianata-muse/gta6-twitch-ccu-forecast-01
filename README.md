# GTA VI launch-day Twitch CCU forecast

[![make.fun](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/fhrianata-muse/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/fhrianata-muse/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

Entry for the [make.fun bounty gta6-twitch-ccu-forecast-01](https://make.fun/bounty/gta6-twitch-ccu-forecast-01):
forecast Grand Theft Auto VI's launch-day Twitch concurrent viewers, hour by
hour (hours 2–24), as five quantiles (5/25/50/75/95) per hour.

## Method

Per-hour OLS in log space, `ln(viewers_h) ~ a_h + b_h * ln(hour-1 print)`,
fit at runtime on the case's own comparables — the past launches' release-day
curves. Two fits are blended in log space: 70% on the rolling-midnight pool
(launches whose hour-1 print is already >= 10% of their pre-midnight wave peak
and >= 10,000 — the regime GTA VI is in) and 30% on the full pool (a hedge for
a deep-trough final, when the early wave is small and the US wave dominates).
Quantile spreads come from leave-one-out residuals inside each pool:
per-hour empirical quantiles blended 50/50 with a normal of the same IQR,
then narrowed 20% toward the blended median. Everything is computed from the
case file; no hardcoded curves.

Local practice result (p00–p03): mean pinball loss **0.2030** vs the house
persistence baseline **0.3871** (-47.6%); per case 0.2275 / 0.2202 / 0.1709 /
0.1933. Leave-one-out over 32 rolling-like past launches: 0.2638 vs baseline
0.3554.

## Run

Python 3.13, standard library only, offline, deterministic. Reads every
`<id>.in` in the input directory and writes `<id>.out`:

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Referral

Built by [@expylar](https://make.fun/r/q4engdna) — referral code `q4engdna`.
