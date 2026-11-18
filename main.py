"""Forecast each hour of GTA VI's release day on Twitch, from the first hour's print.

Entry for make.fun competition `gta6-twitch-ccu-forecast-01`.

Approach
--------
The house baseline assumes every hour draws what hour 1 drew, with a spread that
opens like a random walk (sigma 0.15 / sqrt hour). Release days do not look like
that: in the 38 comparable launches shipped with every case, the category dips
right after the first hour (many viewers arrive at the unlock, then sleep), falls
to its floor around hours 3-8, climbs through the afternoon and peaks in prime
time at 1.8-2.1x the first hour before easing toward the end of the UTC day.

So the forecast is built from the comparables rather than assumed:

  1. every comparable is turned into a curve relative to its own hour 1,
     r_h = v_h / v_1, in natural-log space (a 20% miss costs the same at
     10,000 viewers as at 1,000,000, so logs are the scale that is scored);
  2. the median log ratio per hour is the shape (a median, not a mean: one
     hyped outlier such as Elden Ring must not drag every case up);
  3. the spread of the comparables around that shape, as 1.4826 x MAD, is the
     width of the quantiles: the honest uncertainty between launches. It is
     floored (never narrower than a typical launch disagreement), capped (a
     too-wide band pays at every single hour) and widened with the hour, as the
     day moves further from the one print we know;
  4. a small calibration term uses the pre-release tail in `history_hourly`:
     a category already busy before the unlock peaks at the unlock and falls
     faster afterwards, while a quiet one builds through the day.

Reads every <id>.in in the input directory, writes <id>.out: 23 rows, hours
2..24, five non-decreasing quantiles at levels 0.05, 0.25, 0.5, 0.75, 0.95.

Standard library only; the grader runs `python3 -P -s /bot/main.py` in
python:3.13-slim with no network.
"""

import datetime as dt
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard normal quantiles at those levels.
Z = [-1.6449, -0.6745, 0.0, 0.6745, 1.6449]

# Width floors and caps, in natural-log units.
SIGMA_FLOOR = 0.16
SIGMA_CEIL = 1.15
# How much wider each hour past the first makes the band, per sqrt hour.
SIGMA_GROWTH = 0.035

# Typical release-day shape, used only if a case ships no usable comparable.
FALLBACK_RATIO = [
    0.92, 0.78, 0.72, 0.72, 0.76, 0.82, 0.90, 1.00, 1.02, 1.00, 1.06, 1.14,
    1.22, 1.30, 1.42, 1.58, 1.68, 1.85, 1.95, 1.95, 1.88, 1.75, 1.60,
]
FALLBACK_SIGMA = 0.60
MIN_HOUR1 = 25


def _hourly_map(hourly):
    """from(ISO hour) -> viewers, whole hours only."""
    out = {}
    for point in hourly or []:
        start = point.get("from") or point.get("at")
        if not start or start[14:19] != "00:00":
            continue
        out[start] = point.get("viewers") or 0
    return out


def day_sequence(item):
    """The 24 hourly viewer counts from day_start (hour 1 at index 0); None where unsampled."""
    hourly = _hourly_map(item.get("hourly") or item.get("history_hourly"))
    try:
        start = dt.datetime.strptime(item["day_start"], "%Y-%m-%dT%H:%M:%SZ")
    except (KeyError, ValueError):
        return [None] * 24
    return [hourly.get((start + dt.timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:%SZ"))
            for h in range(24)]


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    if not n:
        return None
    return ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2


def scaled_mad(values):
    """Robust spread in log space: 1.4826 x median absolute deviation."""
    middle = median(values)
    if middle is None:
        return None
    return 1.4826 * (median([abs(v - middle) for v in values]) or 0.0)


def collect_curves(comparables):
    """Log ratios per forecast hour, pooled over every usable comparable."""
    per_hour = [[] for _ in range(23)]
    for comp in comparables or []:
        sequence = day_sequence(comp)
        first = sequence[0]
        if first is None or first < MIN_HOUR1:
            continue
        ratios = []
        for viewers in sequence[1:]:
            if viewers is None or viewers <= 0:
                ratios.append(None)
            else:
                ratios.append(math.log(viewers / first))
        if sum(1 for r in ratios if r is not None) < 20:
            continue
        for index, ratio in enumerate(ratios):
            if ratio is not None and -4.0 < ratio < 4.0:
                per_hour[index].append(ratio)
    return per_hour


def shape(per_hour):
    """(median log ratio, spread) per forecast hour."""
    mu, sigma = [], []
    for index in range(23):
        values = per_hour[index]
        if len(values) >= 4:
            mu.append(median(values))
            spread = scaled_mad(values) or SIGMA_FLOOR
            sigma.append(min(SIGMA_CEIL, max(SIGMA_FLOOR, spread)))
        elif values:
            mu.append(median(values))
            sigma.append(0.75)
        else:
            mu.append(math.log(FALLBACK_RATIO[index]))
            sigma.append(FALLBACK_SIGMA)
    return mu, sigma


def pre_release_pull(case):
    """Shift the shape down when the category was already busy before the unlock.

    A category carrying a big audience into the first hour peaks at the unlock and
    loses it afterwards; one that starts quiet climbs through the day.
    """
    try:
        start = dt.datetime.strptime(case["day_start"], "%Y-%m-%dT%H:%M:%SZ")
        hourly = _hourly_map(case.get("history_hourly"))
        first = float(case["known"][0]["viewers"])
        if first <= 0:
            return 0.0
        tail = []
        for hours_before in range(2, 14):
            key = (start - dt.timedelta(hours=hours_before)).strftime("%Y-%m-%dT%H:%M:%SZ")
            if key in hourly:
                tail.append(hourly[key])
        if not tail:
            return 0.0
        ratio = (sum(tail) / len(tail)) / first
        if ratio <= 0:
            return 0.0
        return max(-0.22, min(0.10, -0.42 * math.log10(max(ratio, 1e-6)) - 0.42))
    except (KeyError, IndexError, ValueError, TypeError):
        return 0.0


def round2(value):
    """Round to 2 decimals, half up, as the rules' examples write them."""
    scaled = value * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    out = whole / 100
    return int(out) if out.is_integer() else out


def forecast_case(case):
    if case.get("levels") != LEVELS:
        raise ValueError(f"case asks for levels {case.get('levels')}, not {LEVELS}")
    first = next((row for row in case["known"] if row["hour"] == 1), None)
    if first is None:
        raise ValueError("case has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))

    mu, sigma = shape(collect_curves(case.get("comparables")))
    pull = pre_release_pull(case)

    hours = []
    for index in range(23):
        hour = index + 2
        centre = mu[index] + pull * min(1.0, (index + 1) / 8.0)
        width = min(SIGMA_CEIL, sigma[index] + SIGMA_GROWTH * math.sqrt(index + 1))
        quantiles = []
        for z in Z:
            value = p1 * math.exp(centre + z * width)
            quantiles.append(round2(max(0.5, min(1e9, value))))
        for k in range(1, 5):
            if quantiles[k] < quantiles[k - 1]:
                quantiles[k] = quantiles[k - 1]
        hours.append({"hour": hour, "q": quantiles})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        body = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(body, encoding="utf-8")


if __name__ == "__main__":
    main()
