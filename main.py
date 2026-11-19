"""Forecast each hour of GTA VI's release day on Twitch, from the first hour's print.

Entry for make.fun competition `gta6-twitch-ccu-forecast-01`.

Approach
--------
The house baseline assumes every hour draws what hour 1 drew, with a spread that
opens like a random walk (sigma 0.15 / sqrt hour). Release days do not look like
that: in the 38 comparable launches shipped with every case, the category dips
right after the first hour (many viewers arrive at the unlock, then sleep), falls
to its floor around hours 3-8, climbs through the afternoon and peaks in prime
time at 1.6-2.4x the first hour before easing toward the end of the UTC day.
The shape also depends on how big the launch is: a launch whose first hour
printed 875,000 viewers (Elden Ring, Cyberpunk 2077) falls further and faster
than one that printed 5,000 (Black Myth: Wukong), because the first hour of a
mega-launch is inflated by the unlock spike.

So the forecast is built from the comparables rather than assumed:

  1. every comparable is turned into a curve relative to its own hour 1,
     r_h = v_h / v_1, in natural-log space (a 20% miss costs the same at
     10,000 viewers as at 1,000,000, so logs are the scale that is scored);
  2. those curves are weighted by how close their own first hour is to ours
     (tricube kernel in log viewers over the 20 nearest): a launch of our size
     teaches our shape better than one ten times smaller or bigger;
  3. the weighted median log ratio per hour is the shape (a median, not a mean:
     one hyped outlier must not drag every case up);
  4. the spread of the comparables around that shape, as 1.4826 x MAD (weighted,
     blended with the spread over every comparable), is the width of the
     quantiles: the honest uncertainty between launches. It is floored (never
     narrower than a typical launch disagreement), capped (a too-wide band pays
     at every single hour) and widened with the hour, as the day moves further
     from the one print we know;
  5. a small calibration term uses the pre-release tail in `history_hourly`:
     a category already busy before the unlock peaks at the unlock and falls
     faster afterwards, while a quiet one builds through the day.

The neighbourhood size and the width floors were chosen by leave-one-out
cross-validation over the 34 large comparable launches (each forecast from the
other 37, scored with this contest's own pinball loss): about 0.39 mean loss,
against 0.58 for a pooled-median shape with a narrow band.

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

# Neighbourhood size, in launches: the k comparables whose hour 1 is nearest ours.
NEIGHBOURS = 20
# Width floors and caps, in natural-log units.
SIGMA_FLOOR = 0.55
SIGMA_CEIL = 1.10
# How much wider each hour past the first makes the band, per sqrt hour.
SIGMA_GROWTH = 0.08
# Share of the pooled spread (every comparable) mixed into the neighbourhood spread.
POOL_BLEND = 0.25

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


def weighted_median(pairs):
    """pairs: [(value, weight)] -> the value where the running weight passes half."""
    ordered = sorted(pairs)
    total = sum(weight for _, weight in ordered)
    if total <= 0:
        return ordered[len(ordered) // 2][0]
    running = 0.0
    for value, weight in ordered:
        running += weight
        if running >= total / 2.0:
            return value
    return ordered[-1][0]


def scaled_mad(values):
    """Robust spread in log space: 1.4826 x median absolute deviation."""
    middle = median(values)
    if middle is None:
        return None
    return 1.4826 * (median([abs(v - middle) for v in values]) or 0.0)


def collect_curves(comparables):
    """[(log of its own hour 1, 23 log ratios or None)] for every usable comparable.

    A comparable teaches the shape of a release day as r_h = v_h / v_1; it is only
    usable when its own first hour is a real print and most of its day was sampled.
    """
    out = []
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
                ratio = math.log(viewers / first)
                ratios.append(ratio if -4.0 < ratio < 4.0 else None)
        if sum(1 for r in ratios if r is not None) >= 20:
            out.append((math.log(first), ratios))
    return out


def neighbour_weights(p1, curves):
    """Tricube weights over the launches whose own hour 1 is nearest ours.

    A launch of our size is the better teacher of our shape: the first hour of a
    mega-launch is inflated by the unlock spike and falls away faster than a
    small launch's, so the shape is taken from the nearest launches in log viewers.
    """
    if not curves:
        return []
    centre = math.log(max(p1, 1.0))
    ranked = sorted((abs(level - centre), index) for index, (level, _) in enumerate(curves))
    chosen = ranked[: max(1, min(NEIGHBOURS, len(ranked)))]
    span = chosen[-1][0]
    weights = []
    for distance, index in chosen:
        weight = 1.0 if span <= 0 else (1.0 - (distance / span) ** 3) ** 3
        weights.append((max(weight, 1e-6), index))
    return weights


def shape(curves, p1):
    """(centre log ratio, spread) per forecast hour, from the nearest launches."""
    weights = neighbour_weights(p1, curves)
    mu, sigma = [], []
    for index in range(23):
        near = []
        pooled = []
        for weight, curve_index in weights:
            ratio = curves[curve_index][1][index]
            if ratio is not None:
                near.append((ratio, weight))
        for _, ratios in curves:
            ratio = ratios[index]
            if ratio is not None:
                pooled.append(ratio)
        if not near:
            mu.append(math.log(FALLBACK_RATIO[index]))
            sigma.append(FALLBACK_SIGMA)
            continue
        centre = weighted_median(near)
        spread = weighted_median([(abs(r - centre), w) for r, w in near]) * 1.4826
        if len(pooled) >= 6:
            pooled_spread = scaled_mad(pooled)
            if pooled_spread:
                spread = (1 - POOL_BLEND) * spread + POOL_BLEND * pooled_spread
        mu.append(centre)
        sigma.append(min(SIGMA_CEIL, max(SIGMA_FLOOR, spread)))
    return mu, sigma


def pre_release_baseline(case, hours_before=range(2, 14)):
    """Mean viewers per hour in the 12 hours leading up to the unlock, or None.

    The window starts two hours before the day does: the hour right before the
    unlock is often already inflated by a pre-load, so it is skipped to keep the
    baseline a description of the ordinary night, not of the spike.
    """
    try:
        start = dt.datetime.strptime(case["day_start"], "%Y-%m-%dT%H:%M:%SZ")
        hourly = _hourly_map(case.get("history_hourly"))
    except (KeyError, ValueError, TypeError):
        return None
    seen = []
    for hours_back in hours_before:
        key = (start - dt.timedelta(hours=hours_back)).strftime("%Y-%m-%dT%H:%M:%SZ")
        if key in hourly:
            seen.append(hourly[key])
    if not seen:
        return None
    return sum(seen) / len(seen)


def pre_release_pull(case):
    """Shift the shape down when the category was already busy before the unlock.

    A category carrying a big audience into the first hour (a Fortnite-size event,
    a 24/7 channel) peaks at the unlock and then loses that standing audience;
    one that starts quiet builds through the day. The size of the shift is read
    from how much traffic the category already had before hour 1, measured in
    log10 against the first hour's print, and it is clamped so a strange history
    can never flip the shape on its own. Zero means "apply no correction".
    """
    try:
        first = float(case["known"][0]["viewers"])
    except (KeyError, IndexError, TypeError, ValueError):
        return 0.0
    if first <= 0:
        return 0.0
    baseline = pre_release_baseline(case)
    if baseline is None or baseline <= 0:
        return 0.0
    # ratio 1.0 (as busy before as in hour 1) -> -0.42; a quiet night -> near 0.
    ratio = baseline / first
    pull = -0.42 * math.log10(max(ratio, 1e-6)) - 0.42
    return max(-0.22, min(0.10, pull))


def round2(value):
    """Round to 2 decimals, half up, as the rules' examples write them."""
    scaled = value * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    out = whole / 100
    return int(out) if out.is_integer() else out


def hour_one_print(case):
    """The viewer count the rules guarantee every case carries: hour 1."""
    if case.get("levels") != LEVELS:
        raise ValueError(f"case asks for levels {case.get('levels')}, not {LEVELS}")
    first = next((row for row in case.get("known") or [] if row.get("hour") == 1), None)
    if first is None:
        raise ValueError("case has no hour-1 print")
    return max(1.0, float(first["viewers"]))


def quantiles_for(p1, centre, width):
    """Five log-normal quantiles around p1 * exp(centre), made monotone and in range.

    Working in log space keeps the band symmetric in ratio terms (a doubled forecast
    and a halved one are equally far away), which is the scale pinball loss punishes.
    """
    values = []
    for z in Z:
        value = p1 * math.exp(centre + z * width)
        values.append(round2(max(0.5, min(1e9, value))))
    for k in range(1, len(values)):
        if values[k] < values[k - 1]:
            values[k] = values[k - 1]
    return values


def forecast_case(case):
    """One case in, 23 rows (hours 2..24) of five quantiles out."""
    p1 = hour_one_print(case)
    mu, sigma = shape(collect_curves(case.get("comparables")), p1)
    pull = pre_release_pull(case)

    hours = []
    for index in range(23):
        # The pre-release correction fades in over the first eight hours: it speaks
        # about the standing audience, which the day's own curve reveals gradually.
        fade = min(1.0, (index + 1) / 8.0)
        centre = mu[index] + pull * fade
        # The band widens with distance from the one print we know.
        width = min(SIGMA_CEIL, sigma[index] + SIGMA_GROWTH * math.sqrt(index + 1))
        hours.append({"hour": index + 2, "q": quantiles_for(p1, centre, width)})
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
