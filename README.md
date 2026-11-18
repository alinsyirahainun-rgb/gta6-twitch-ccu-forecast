# Forecaster for GTA VI's launch-day Twitch audience (hour by hour)

An entry for make.fun's bounty **“Forecast GTA VI's launch-day Twitch viewers hour by hour”** — $250 in USDC.

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

## What it does

`main.py` reads one case file — everything Twitch recorded for the category up to the
end of release-day hour 1, that hour's count included — and writes, for each of hours
2 to 24, five quantiles (5%, 25%, 50%, 75%, 95%) of that hour's concurrent viewers.

## How the forecast is made

1. **Shape from comparable launches.** Every past big console launch shipped with the
   case is turned into a curve relative to its own hour 1, `r_h = v_h / v_1`, in
   natural-log space (the score is a pinball loss on logs, so 20% costs the same at
   10,000 as at 1,000,000 viewers). The per-hour **median** log ratio is the shape — a
   median so one hyped outlier cannot drag every case.
2. **Honest spread.** The disagreement between launches around that shape, as
   `1.4826 × MAD` in log space, sets the width of the quantiles: floored so the band is
   never falsely narrow, capped so a needlessly wide band does not pay at every hour,
   and widened slowly with the hour since the day drifts from the one print we know.
3. **Pre-release tail.** The category's own history before the unlock calibrates the
   level: a category already busy before release peaks at the unlock and decays faster,
   a quiet one builds through the day.

Standard library only, no network; the grader runs `python3 -P -s /bot/main.py`.

## Run it locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python main.py
```

## LICENSE

MIT — see `LICENSE`.
