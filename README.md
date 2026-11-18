# gta6-twitch-ccu-forecast

Forecast GTA VI's launch-day Twitch viewers, hour by hour — an entry for make.fun's
bounty **“Forecast GTA VI's launch-day Twitch viewers hour by hour”**, $250 in USDC.

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/alinsyirahainun-rgb/gta6-twitch-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/alinsyirahainun-rgb/gta6-twitch-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

## What it does

`main.py` reads one case file — everything Twitch recorded for the category up to the
end of release-day hour 1, that hour's count included — and writes, for each of hours
2 to 24, five quantiles (5%, 25%, 50%, 75%, 95%) of that hour's concurrent viewers.

## How the forecast is made

1. **Shape from comparable launches.** Every past big console launch shipped with the
   case is turned into a curve relative to its own hour 1, `r_h = v_h / v_1`, in
   natural-log space (the score is a pinball loss on logs, so a 20% miss costs the same
   at 10,000 as at 1,000,000 viewers).
2. **Size-matched neighbours.** The comparables are weighted by how close their own
   first hour is to ours (tricube kernel in log viewers over the 20 nearest): the first
   hour of a mega-launch is inflated by the unlock spike, so a launch of our size is the
   better teacher of our shape than one ten times bigger or smaller.
3. **Honest spread.** The weighted median log ratio per hour is the shape; the weighted
   median absolute deviation around it (× 1.4826), blended with the spread over every
   comparable, sets the width of the band — floored so it is never falsely narrow,
   capped so a needlessly wide band does not pay at every hour, and widened slowly with
   the hour since the day drifts away from the one print we know.
4. **Pre-release tail.** The category's own history before the unlock calibrates the
   level: a category already busy before release peaks at the unlock and decays faster,
   a quiet one builds through the day.

The neighbourhood size and the width floors were chosen by leave-one-out
cross-validation over 34 comparable launches, scored with this contest's own pinball
loss: about 0.39 mean loss, against 0.58 for a pooled-median shape with a narrow band.

Standard library only, no network; the grader runs `python3 -P -s /bot/main.py`.

## Run it locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python main.py
```

## LICENSE

MIT — see `LICENSE`.
