#!/usr/bin/env python3
"""
Rain Monitor viewer
===================

Opens the CSV logs written by the rain monitor (RAINnnnn.CSV, and the older
WETnnnn.CSV files) and turns them into plots and tables you can read at a glance.

    python rain_monitor_viewer.py                      open the window, then File > Open
    python rain_monitor_viewer.py RAIN0003.CSV         open a log straight away
    python rain_monitor_viewer.py RAIN0003.CSV --png out/    no window: save every plot as a PNG

Needs: Python 3.9+, pandas, numpy, matplotlib (pip install pandas matplotlib).
tkinter ships with most Python installs (on Debian/Ubuntu: sudo apt install python3-tk).
"""

import argparse
import os
import sys
import textwrap
import time

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.dates as mdates
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

# ---------------------------------------------------------------------------
#  Look
# ---------------------------------------------------------------------------

# Color-blind-safe palette (Okabe-Ito)
BLUE, ORANGE, GREEN, RED, PURPLE, SKY, GRAY = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#6b7280",
)

# One fill color per state, used by every plot and by the color key in the window.
STATE_FILL = {"DRY": "#f1ede4", "WET": "#d6eaf8", "DEW": "#bfe8e3", "RAIN": "#8fbce6"}

matplotlib.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "legend.frameon": False,
        "legend.fontsize": 8,
    }
)

# ---------------------------------------------------------------------------
#  Loading and deriving columns
# ---------------------------------------------------------------------------

# Same defaults as the firmware (rain_monitor.ino).
DEFAULTS = dict(
    rise_mv=4.0, activity_mv=0.3, zigzag_samples=6, rise_samples=60, hold_min=30.0
)


def zigzag(x, w):
    """Distance traveled over the last w samples minus the net change, per step.
    A smooth rise or fall scores 0; drops landing and running off score high."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    out = np.zeros(n)
    if n < w or w < 2:
        return out
    cs = np.cumsum(np.abs(np.diff(x, prepend=x[0])))
    traveled = cs[w - 1 :] - cs[: n - w + 1]
    net = np.abs(x[w - 1 :] - x[: n - w + 1])
    out[w - 1 :] = (traveled - net) / (w - 1)
    return out


def rise_over(x, r):
    """Reading minus the reading r samples earlier (the first r samples compare with the first)."""
    x = np.asarray(x, dtype=float)
    out = x - x[0]
    if len(x) > r:
        out[r:] = x[r:] - x[:-r]
    return out


def load_log(path, utc_offset_h=0.0):
    """Read one log into a DataFrame with tidy, derived columns. Works on logs from
    every firmware version that has a leaf reading (mv, or volts in the oldest)."""
    d = pd.read_csv(path, skipinitialspace=True)
    d.columns = [c.strip() for c in d.columns]
    if "mv" not in d and "volts" in d:
        d["mv"] = d["volts"] * 1000.0
    if "mv" not in d:
        raise ValueError("no 'mv' column: this does not look like a rain monitor log")
    if "state" not in d and "class" in d:
        d["state"] = d["class"]
    for c in d.columns:
        if c not in ("date", "time", "state", "class", "source"):
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[d["mv"].notna()].reset_index(drop=True)

    if "uptime_s" not in d:
        d["uptime_s"] = (np.arange(len(d)) + 1) * 10.0
    d.attrs["dt"] = float(np.median(np.diff(d["uptime_s"]))) if len(d) > 1 else 10.0
    d.attrs["name"] = os.path.basename(path)

    # Real time: stamp every row from the first valid GPS stamp plus uptime, so rows
    # logged before a GPS fix (blank date/time in old logs) still get a time.
    if "time" in d:
        date = d["date"].astype(str) if "date" in d else "2000-01-01"
        stamp = pd.to_datetime(date + " " + d["time"].astype(str), errors="coerce")
    else:
        stamp = pd.Series(pd.NaT, index=d.index)
    ok = stamp.notna()
    d.attrs["has_clock"] = bool(ok.any()) and "date" in d
    if ok.any():
        zero = (stamp[ok] - pd.to_timedelta(d.loc[ok, "uptime_s"], unit="s")).iloc[0]
    else:
        zero = pd.Timestamp("2000-01-01")
    d["t_utc"] = zero + pd.to_timedelta(d["uptime_s"], unit="s")
    d["t"] = d["t_utc"] + pd.to_timedelta(utc_offset_h, unit="h")

    for col, default in (
        ("tips", 0),
        ("temp_c", np.nan),
        ("rh_pct", np.nan),
        ("dewpt_c", np.nan),
    ):
        if col not in d:
            d[col] = default
    d["tips"] = d["tips"].fillna(0)
    d["rain_mm_cum"] = d["tips"].cumsum() * 0.1
    d.attrs["has_gauge"] = bool(d["tips"].sum() > 0 or "tips_total" in d)
    d.attrs["has_air"] = bool(d["temp_c"].notna().any())

    d["state"] = (
        d["state"].astype(str).str.upper().where(d["state"].notna(), "")
        if "state" in d
        else ""
    )
    d.attrs["has_state"] = bool((d["state"] != "").any())

    # Re-derived with fixed definitions so every firmware version is comparable.
    d["rise10"] = rise_over(d["mv"], int(round(600 / d.attrs["dt"])))
    d["zig"] = zigzag(d["mv"], max(2, int(round(60 / d.attrs["dt"])) + 1))
    if "baseline_mv" not in d:
        d["baseline_mv"] = np.nan
    d["above"] = d["mv"] - d["baseline_mv"]

    # Degree-days (base 0 degC): use the logged column if present, otherwise integrate temp_c.
    if "degree_days" in d and d["degree_days"].notna().any():
        d["dd"] = d["degree_days"]
        d.attrs["dd_source"] = "logged"
    elif d.attrs["has_air"]:
        step = np.maximum(d["temp_c"].fillna(0), 0) * d.attrs["dt"] / 86400.0
        d["dd"] = step.cumsum()
        d.attrs["dd_source"] = "computed from temp_c"
    else:
        d["dd"] = np.nan
        d.attrs["dd_source"] = "none"

    # Rain signal: logged by v8+ firmware; otherwise replayed with the firmware defaults.
    if "rain_sig" in d and d["rain_sig"].notna().any():
        d["sig"] = d["rain_sig"].fillna(0) > 0
        d.attrs["sig_source"] = "logged"
    else:
        d["sig"] = replay_detector(d, **DEFAULTS)["leaf_sig"] | (d["tips"].values > 0)
        d.attrs["sig_source"] = "replayed (log has no rain_sig)"
    return d


def replay_detector(d, rise_mv, activity_mv, zigzag_samples, rise_samples, hold_min):
    """Re-run the firmware's rain test on the leaf reading alone (gauge tips excluded)
    so thresholds can be tuned offline. Returns per-sample arrays."""
    mv = d["mv"].values
    rise = rise_over(mv, int(rise_samples))
    zig = zigzag(mv, int(zigzag_samples))
    leaf_sig = (rise >= rise_mv) | (zig >= activity_mv)
    t = d["uptime_s"].values.astype(float)
    last = pd.Series(np.where(leaf_sig, t, np.nan)).ffill().values
    active = np.nan_to_num(t - last, nan=1e12) < hold_min * 60
    return dict(rise=rise, zig=zig, leaf_sig=leaf_sig, active=active)


# ---------------------------------------------------------------------------
#  Analysis
# ---------------------------------------------------------------------------


def find_events(d, gap_min=30.0):
    """Group rain signals (leaf or gauge) into events separated by at least gap_min of quiet."""
    idx = np.flatnonzero(d["sig"].values)
    if len(idx) == 0:
        return pd.DataFrame()
    t = d["uptime_s"].values
    breaks = np.flatnonzero(np.diff(t[idx]) > gap_min * 60)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    rows = []
    for a, b in zip(starts, ends):
        lo = max(0, a - int(300 / d.attrs["dt"]))  # tips just before the leaf noticed
        seg = d.iloc[lo : b + 1]
        tips = int(seg["tips"].sum())
        tip_rows = seg[seg["tips"] > 0]
        lead = (
            ((tip_rows["uptime_s"].iloc[0] - d["uptime_s"].iloc[a]) / 60)
            if tips
            else np.nan
        )
        after = d.iloc[a : min(len(d), b + 1 + int(1800 / d.attrs["dt"]))]
        base = d["baseline_mv"].iloc[a]
        ref = base if pd.notna(base) else d["mv"].iloc[max(0, a - 1)]
        rows.append(
            dict(
                start=d["t"].iloc[a],
                end=d["t"].iloc[b],
                minutes=(d["uptime_s"].iloc[b] - d["uptime_s"].iloc[a]) / 60,
                tips=tips,
                mm=tips * 0.1,
                lead_min=lead,
                peak_rise_mv=after["mv"].max() - ref,
                max_zig=d["zig"].iloc[a : b + 1].max(),
                kind=("leaf only" if tips == 0 else "gauge + leaf"),
            )
        )
    return pd.DataFrame(rows)


def daily_table(d):
    day = d["t"].dt.date
    dt_h = d.attrs["dt"] / 3600
    g = pd.DataFrame(
        {
            "day": day,
            "rain_mm": d["tips"] * 0.1,
            "rain_h": (d["state"] == "RAIN") * dt_h,
            "wet_h": d["state"].isin(["RAIN", "WET", "DEW"]) * dt_h,
            "hours": dt_h,
            "tmin": d["temp_c"],
            "tmax": d["temp_c"],
            "dd": d["dd"].diff().fillna(d["dd"].iloc[0] if len(d) else 0),
        }
    )
    out = g.groupby("day").agg(
        rain_mm=("rain_mm", "sum"),
        rain_h=("rain_h", "sum"),
        wet_h=("wet_h", "sum"),
        hours=("hours", "sum"),
        tmin=("tmin", "min"),
        tmax=("tmax", "max"),
        dd=("dd", "sum"),
    )
    return out.reset_index()


def state_runs(d):
    """Yield (start, end, state) for each stretch of one state."""
    s = d["state"].values
    t = d["t"].values
    if len(s) == 0:
        return
    start = 0
    for i in range(1, len(s) + 1):
        if i == len(s) or s[i] != s[start]:
            if s[start] in STATE_FILL:
                yield t[start], t[min(i, len(s) - 1)], s[start]
            start = i


def summary_text(d, events):
    dt = d.attrs["dt"]
    hours = len(d) * dt / 3600
    L = []
    L.append(f"File            {d.attrs['name']}")
    L.append(f"Rows            {len(d):,}  ({dt:.0f} s apart, {hours:.1f} h)")
    clock = (
        "GPS time"
        if d.attrs["has_clock"]
        else "no clock in this log (times counted from start)"
    )
    L.append(f"From            {d['t'].iloc[0]:%Y-%m-%d %H:%M}   ({clock})")
    L.append(f"To              {d['t'].iloc[-1]:%Y-%m-%d %H:%M}")
    gaps = int((np.diff(d["uptime_s"]) > 2.5 * dt).sum())
    L.append(f"Gaps in data    {gaps}")
    L.append("")
    L.append("LEAF SENSOR")
    L.append(
        f"  Reading       {d['mv'].min():.0f} to {d['mv'].max():.0f} mV   (median {d['mv'].median():.0f})"
    )
    if d["baseline_mv"].notna().any():
        L.append(f"  Dry baseline  {d['baseline_mv'].iloc[0]:.0f} mV at start")
    if d.attrs["has_state"]:
        wet = d["state"].isin(["RAIN", "WET", "DEW"])
        rain = d["state"] == "RAIN"
        L.append(
            f"  Leaf wet      {wet.mean() * 100:.0f}% of the time ({wet.sum() * dt / 3600:.1f} h)"
        )
        L.append(f"  State RAIN    {rain.sum() * dt / 3600:.1f} h")
        n_dew = int(((d["state"] == "DEW") & (d["state"].shift() != "DEW")).sum())
        L.append(f"  Dew periods   {n_dew}")
    L.append("")
    L.append("RAIN GAUGE")
    if d.attrs["has_gauge"]:
        tips = int(d["tips"].sum())
        L.append(f"  Tips          {tips}   ({tips * 0.1:.1f} mm)")
        if tips:
            wet_h = d["tips"].groupby(d["t"].dt.floor("h")).sum().max() * 0.1
            L.append(f"  Heaviest hour {wet_h:.1f} mm")
    else:
        L.append("  none in this log")
    L.append("")
    L.append(
        f"RAIN EVENTS     {len(events)}   (signals less than 30 min apart merged; rain signal: {d.attrs['sig_source']})"
    )
    if len(events):
        both = int((events["tips"] > 0).sum())
        L.append(f"  with gauge    {both}")
        L.append(
            f"  leaf only     {len(events) - both}   (light rain or drizzle the gauge missed, or false alarms)"
        )
    L.append("")
    L.append("AIR (the sensor box, not the open air)")
    if d.attrs["has_air"]:
        L.append(
            f"  Temperature   {d['temp_c'].min():.1f} to {d['temp_c'].max():.1f} degC"
        )
        L.append(
            f"  Humidity      {d['rh_pct'].min():.0f} to {d['rh_pct'].max():.0f} %"
        )
        if d["dd"].notna().any():
            L.append(
                f"  Degree-days   {d['dd'].iloc[-1] - d['dd'].iloc[0]:.2f}  above 0 degC  ({d.attrs['dd_source']})"
            )
    else:
        L.append("  none in this log")
    return "\n".join(L)


def narrative(d, events):
    """The log in a few plain sentences, for someone who has never seen one."""
    dt = d.attrs["dt"]
    hours = len(d) * dt / 3600
    out = [
        f"This log covers {hours:.1f} hours, from {d['t'].iloc[0]:%b %d %H:%M} to {d['t'].iloc[-1]:%b %d %H:%M}."
    ]
    if not d.attrs["has_clock"]:
        out.append(
            "It has no real clock time, so times are counted from when the logger started."
        )
    if d.attrs["has_gauge"]:
        tips = int(d["tips"].sum())
        out.append(
            f"The rain gauge caught {tips * 0.1:.1f} mm ({tips} tip{'s' if tips != 1 else ''})."
            if tips
            else "The rain gauge recorded no rain."
        )
    else:
        out.append(
            "This log has no rain gauge data, so rain is judged from the leaf sensor alone."
        )
    if len(events):
        n = len(events)
        leaf_only = int((events["tips"] == 0).sum())
        s = f"The logger found {n} rain event{'s' if n != 1 else ''}"
        if not d.attrs["has_gauge"]:
            s += "."
        elif leaf_only:
            s += f", {leaf_only} of them seen only by the leaf sensor (light rain or drizzle too fine for the gauge, or a false alarm)."
        else:
            s += ", all confirmed by the gauge."
        out.append(s)
        lead = events["lead_min"].dropna()
        if len(lead):
            m = float(lead.median())
            if abs(m) < 1:
                out.append(
                    "The leaf sensor noticed rain at about the same time as the gauge's first tip."
                )
            else:
                out.append(
                    f"The leaf sensor noticed rain about {abs(m):.0f} minutes {'before' if m >= 0 else 'after'} "
                    "the gauge's first tip (median over the events)."
                )
    else:
        out.append("No rain was detected.")
    if d.attrs["has_state"]:
        wet = d["state"].isin(["RAIN", "WET", "DEW"])
        out.append(
            f"The leaf was wet for {wet.sum() * dt / 3600:.1f} hours ({wet.mean() * 100:.0f}% of the log)."
        )
        if len(events):
            after = d[d["t"] > events["end"].iloc[-1]]
            dry = after[after["state"] == "DRY"]
            if len(after) and len(dry):
                out.append(
                    f"After the last rain signal it stayed wet for another "
                    f"{(dry['t'].iloc[0] - events['end'].iloc[-1]).total_seconds() / 3600:.1f} hours."
                )
            elif len(after):
                out.append("It was still wet when the log ended.")
        n_dew = int(((d["state"] == "DEW") & (d["state"].shift() != "DEW")).sum())
        if n_dew:
            out.append(
                f"{n_dew} wet period{'s' if n_dew != 1 else ''} started with no rain at all: dew or fog."
            )
    if d.attrs["has_air"]:
        out.append(
            f"The air in the sensor box ranged from {d['temp_c'].min():.1f} to {d['temp_c'].max():.1f} degC "
            f"({d['dd'].iloc[-1] - d['dd'].iloc[0]:.1f} degree-days above 0 degC)."
        )
    gaps = int((np.diff(d["uptime_s"]) > 2.5 * dt).sum())
    if gaps:
        out.append(
            f"Warning: the data has {gaps} gap{'s' if gaps != 1 else ''} (power loss or SD trouble)."
        )
    return " ".join(out)


# ---------------------------------------------------------------------------
#  Help text. Shown in the Help tab; the short captions sit at the top of each tab.
# ---------------------------------------------------------------------------

CAPTIONS = {
    "Overview": "The log in plain words, then the numbers. The table on the right is one row per day (your local time).",
    "Timeline": "Everything on one time axis. Top: the leaf sensor reading, shaded by what the logger decided "
    "(color key at the top of the window); orange ticks along the bottom are rain gauge tips. Below: rain "
    "per 10 minutes with the running total, the evidence the logger uses to spot rain, and the air. "
    "Use the toolbar under the plot to zoom and pan.",
    "Rain events": "Each row is one stretch of rain, made by joining rain signals that are close together. "
    "'Leaf leads gauge' is how many minutes the leaf noticed the rain before the first gauge tip "
    "(blank: the gauge never tipped). Click a row to zoom in on it.",
    "Rain intensity": "Does the leaf tell you how hard it is raining? Left: each dot is 10 minutes. Higher = more rain "
    "at the gauge, further right = a more jumpy leaf reading. Right: the same thing hour by hour.",
    "Day by hour": "One row per day, one box per hour (your local time). Sand = dry, light blue = wet, blue = raining, "
    "white = no data. Orange dots are gauge tips (bigger = more). The bars show each day's rainfall.",
    "Temperature and degree-days": "Top: air temperature and dew point inside the sensor box; when the two lines meet, "
    "dew is possible. Bottom: degree-days above 0 degC, added up over the log.",
    "Tune the detector": "For anyone adjusting the firmware. Move a slider and the leaf-only rain test is replayed on "
    "this log. Blue shading = the leaf says it is raining; compare it with the orange gauge ticks.",
}

# Each section: (key, title, [paragraphs]). A paragraph is a string, ("swatch", state, text),
# ("term", name, text) or ("bullet", text).
HELP = [
    (
        "start",
        "What am I looking at?",
        [
            "The rain monitor sits in the vineyard canopy and writes one row to its SD card every 10 seconds. "
            "It senses rain in two different ways, and this viewer lines them up so you can see what each one noticed.",
            (
                "term",
                "Rain gauge (tipping bucket)",
                "Every 0.1 mm of rain tips a small bucket and the tip is logged. Accurate amounts, but it cannot see "
                "drizzle too light to fill a bucket, and it only reports after the water has arrived.",
            ),
            (
                "term",
                "Leaf wetness sensor (PHYTOS31)",
                "A flat sensor that reads higher the more water is on its surface. It reacts within seconds to the lightest "
                "rain and shows how long the leaf stays wet afterwards, but it has no units of rain: it reads in millivolts "
                "(mV), not millimeters.",
            ),
            "From the two, the logger decides at every sample whether the leaf is dry, wet or being rained on, and counts "
            "the time since the last rain. When it rained, how much, and how long leaves stayed wet are the inputs that "
            "vineyard disease-risk models need.",
            (
                "bullet",
                "Open a log with 'Open CSV...' in the top bar. It reads RAINnnnn.CSV and the older WETnnnn.CSV files.",
            ),
            (
                "bullet",
                "Times in the logs are UTC. The box in the top bar shifts everything to your local time.",
            ),
            (
                "bullet",
                "Start with the Overview tab, then the Timeline. Every tab has a short note at the top, and "
                "'More help' (or F1) opens the matching section here.",
            ),
        ],
    ),
    (
        "colors",
        "Color key",
        [
            ("swatch", "DRY", "DRY: the leaf is at its dry level."),
            (
                "swatch",
                "WET",
                "WET: the leaf is wet, but there has been no rain signal for 30 minutes. Rain has stopped "
                "and the leaf is still drying. Also used when the logger starts on an already wet leaf.",
            ),
            (
                "swatch",
                "DEW",
                "DEW: the leaf is wet and there has been no rain at all in this wet period: dew or fog.",
            ),
            (
                "swatch",
                "RAIN",
                "RAIN: the leaf is wet and the logger saw a rain signal (leaf or gauge) in the last 30 minutes.",
            ),
            "In the plots:",
            (
                "bullet",
                "Dark line: the leaf sensor reading in mV. Dashed gray line: the dry baseline it learned.",
            ),
            (
                "bullet",
                "Orange ticks or dots: a rain gauge tip (0.1 mm). Orange line: the running rainfall total, or the temperature.",
            ),
            (
                "bullet",
                "Blue line in the 'rain evidence' panel: how far the reading rose in 10 minutes. Red: how much it zig-zagged "
                "in the last minute. Dotted lines: the thresholds the firmware uses.",
            ),
            (
                "bullet",
                "Dashed light blue line: dew point. Light blue line on the right-hand axis: relative humidity.",
            ),
        ],
    ),
    (
        "terms",
        "Glossary",
        [
            (
                "term",
                "mV (millivolts)",
                "The leaf sensor's raw reading. In these logs a dry sensor reads about 350-365 mV and "
                "a soaked one 550-680 mV. It is a relative signal, not a calibrated wetness percentage.",
            ),
            (
                "term",
                "Dry baseline",
                "The reading when the sensor is dry. The logger learns it by itself and ignores slow drift "
                "while it is dry. 'Wet' means 20 mV or more above it.",
            ),
            (
                "term",
                "Rain signal",
                "The moment the logger decides water is arriving. It happens when the reading rises 4 mV "
                "or more in 10 minutes, or zig-zags 0.3 mV or more per sample over the last minute, or the "
                "gauge tips.",
            ),
            (
                "term",
                "Rise over 10 minutes",
                "The reading now minus the reading 10 minutes ago. Rain pushes it up; dew and "
                "temperature drift move it far more slowly.",
            ),
            (
                "term",
                "Zig-zag",
                "How much the reading jumps up and down: drops landing and running off. A smooth rise or fall "
                "(like drying) scores zero however fast it moves. Heavier rain zig-zags more.",
            ),
            (
                "term",
                "Hold (30 minutes)",
                "Rain stays 'active' for 30 minutes after the last rain signal, so a short pause "
                "does not split one storm in two. 'Time since last rain' counts from the last signal.",
            ),
            (
                "term",
                "Rain event",
                "One stretch of rain: rain signals less than 30 minutes apart (you can change this on the "
                "Rain events tab) joined together.",
            ),
            (
                "term",
                "Leaf only event",
                "An event with no gauge tip. Either light rain or drizzle under the gauge's 0.1 mm "
                "step, or a false alarm. The gauge cannot tell you which.",
            ),
            ("term", "Tip", "One tip of the rain gauge's bucket: 0.1 mm of rain."),
            (
                "term",
                "Degree-days",
                "Heat added up over time: 1 degree-day is 1 degC above 0 degC for 24 hours. It counts from "
                "when the logger was switched on.",
            ),
            (
                "term",
                "Dew point and its depression",
                "The temperature at which the air would start to form dew. The closer the "
                "air temperature is to it (small depression), the more likely dew is.",
            ),
            ("term", "RH", "Relative humidity of the air, in percent."),
        ],
    ),
    (
        "Overview",
        "Overview tab",
        [
            "The first paragraph is the whole log in plain words. Under it are the details, and on the right a table with one row per day.",
            (
                "bullet",
                "'Rain mm' comes from the gauge only. 'Rain h' is hours in the RAIN state; 'wet h' is hours the leaf was wet.",
            ),
            (
                "bullet",
                "Gaps in the data mean the logger lost power or the SD card stopped writing.",
            ),
        ],
    ),
    (
        "Timeline",
        "Timeline tab",
        [
            "What to look for in the top panel:",
            (
                "bullet",
                "A steep climb of the dark line: rain starting. A staircase of small steps: light rain, each step a drop landing.",
            ),
            (
                "bullet",
                "A sudden drop: water running off the sensor. A slow fall over hours: drying.",
            ),
            (
                "bullet",
                "A line stuck high for hours with no gauge tips: the leaf is still wet after the rain.",
            ),
            (
                "bullet",
                "Orange ticks with no line movement around them are rare. A climb with no ticks is light rain the gauge missed.",
            ),
            "Under it: rain per 10 minutes with the running total, then the two signals the logger uses to spot rain, then the air.",
        ],
    ),
    (
        "Rain events",
        "Rain events tab",
        [
            "Each row is an event: when it started and ended, how long, tips and millimeters, how many minutes before the first "
            "tip the leaf noticed it, and how far the leaf reading climbed.",
            (
                "bullet",
                "A bigger 'merge' number joins nearby events into one; a smaller number splits them.",
            ),
            (
                "bullet",
                "'leaf only' rows had no gauge tip. A tip count of zero with a clear rise on the plot is most likely drizzle.",
            ),
            (
                "bullet",
                "The plot shows the event with 45 minutes either side. Red dots are the moments the logger saw rain.",
            ),
        ],
    ),
    (
        "Rain intensity",
        "Rain intensity tab",
        [
            "Left: each dot is 10 minutes. If dots climb to the right, a more jumpy leaf goes with harder rain, which would let "
            "the leaf estimate rain rate in steps finer than the gauge's 0.1 mm. The orange line is the average per band and "
            "the number is the correlation (1 = perfect).",
            "Right: gauge rainfall per hour (bars) with the leaf zig-zag (line). They should rise and fall together.",
            "With few tips there is nothing to conclude. This needs a log with real rain.",
        ],
    ),
    (
        "Day by hour",
        "Day by hour tab",
        [
            "A quick picture of many days. Long light-blue stretches after a dark blue patch are the hours the leaves stayed wet "
            "after rain, which matters most for disease. White boxes are hours with no data.",
        ],
    ),
    (
        "Temperature and degree-days",
        "Temperature and degree-days tab",
        [
            "The temperature is the air inside the sealed sensor box, so it reads warmer than the open air in sun and lags the "
            "weather. Treat it, and the degree-days built from it, as a rough guide until a shielded outdoor sensor is fitted.",
            "Degree-days restart at zero every time the logger is switched on. The small bar chart shows degree-days per day.",
        ],
    ),
    (
        "Tune the detector",
        "Tune the detector tab",
        [
            "This replays the leaf-only rain test of the firmware on the open log with the thresholds you choose. The gauge is "
            "used only to score it.",
            (
                "bullet",
                "Lower thresholds find lighter rain but risk false alarms from drift or noise.",
            ),
            (
                "bullet",
                "'Rain periods with no tip' grows when you lower a threshold. Those may be real drizzle or false alarms; "
                "only a second record of the weather can tell you which.",
            ),
            (
                "bullet",
                "Sliders here do not change the device. To use new values, edit the constants of the same names in rain_monitor.ino.",
            ),
        ],
    ),
    (
        "caveats",
        "Things to keep in mind",
        [
            (
                "bullet",
                "The gauge only reports rain in 0.1 mm steps, and a dry funnel absorbs the first little bit. Drizzle often never tips it.",
            ),
            (
                "bullet",
                "The leaf sensor can stay wet for hours after rain, especially in cool, cloudy, humid weather. That is real wetness.",
            ),
            (
                "bullet",
                "Moving the sensor, spraying it, or wiping it makes the reading jump about and can look like rain.",
            ),
            (
                "bullet",
                "The air temperature, humidity and dew point come from a sensor in a sealed box and are only a rough guide.",
            ),
            (
                "bullet",
                "Thresholds were set from a few logs from one site. Check them against your own gauge data on the tuning tab.",
            ),
        ],
    ),
]


# ---------------------------------------------------------------------------
#  Plots. Each takes an empty Figure and a loaded log, and draws on it.
# ---------------------------------------------------------------------------


def _time_axis(ax):
    loc = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


def _note(ax, text):
    ax.text(
        0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", color=GRAY
    )
    ax.set_xticks([])
    ax.set_yticks([])


def _tip_ticks(ax, d, y):
    tips = d[d["tips"] > 0]
    if len(tips):
        ax.plot(
            tips["t"],
            [y] * len(tips),
            "|",
            color=ORANGE,
            markersize=12,
            markeredgewidth=1.6,
            label="rain gauge tip",
            clip_on=False,
        )


def plot_timeline(fig, d):
    """The main picture: leaf reading shaded by state, rain, the rain evidence, and air."""
    fig.clear()
    ratios = [3.2, 1.3, 1.3, 1.3]
    axs = fig.subplots(
        4, 1, sharex=True, gridspec_kw=dict(height_ratios=ratios, hspace=0.12)
    )
    a1, a2, a3, a4 = axs

    for s, e, st in state_runs(d):
        a1.axvspan(s, e, color=STATE_FILL[st], lw=0)
    a1.plot(d["t"], d["mv"], color="#1f2937", lw=0.9, label="leaf reading")
    if d["baseline_mv"].notna().any():
        a1.plot(
            d["t"], d["baseline_mv"], color=GRAY, lw=1, ls="--", label="dry baseline"
        )
    _tip_ticks(a1, d, d["mv"].min() - 0.02 * (d["mv"].max() - d["mv"].min()))
    a1.set_ylabel("leaf, mV")
    a1.set_title(f"{d.attrs['name']}  -  leaf wetness, rain and air")
    handles, labels = a1.get_legend_handles_labels()
    if d.attrs["has_state"]:
        from matplotlib.patches import Patch

        for st, c in STATE_FILL.items():
            if (d["state"] == st).any():
                handles.append(Patch(facecolor=c, edgecolor=GRAY, linewidth=0.6))
                labels.append(st)
    a1.legend(handles, labels, ncol=6, loc="upper left")

    if d.attrs["has_gauge"]:
        step = max(1, int(round(600 / d.attrs["dt"])))
        per10 = d["tips"].groupby(np.arange(len(d)) // step).sum() * 0.1
        tt = d["t"].iloc[::step].reset_index(drop=True)[: len(per10)]
        a2.bar(
            tt,
            per10.values,
            width=step * d.attrs["dt"] / 86400,
            align="edge",
            color=BLUE,
            alpha=0.8,
        )
        a2.set_ylabel("mm / 10 min")
        b2 = a2.twinx()
        b2.plot(d["t"], d["rain_mm_cum"], color=ORANGE, lw=1.6)
        b2.set_ylabel("total mm", color=ORANGE)
        b2.grid(False)
        b2.spines["right"].set_visible(True)
    else:
        _note(a2, "no rain gauge data in this log")

    a3.plot(d["t"], d["rise10"], color=BLUE, lw=0.8, label="rise over 10 min (mV)")
    a3.axhline(DEFAULTS["rise_mv"], color=BLUE, lw=0.8, ls=":")
    a3.set_ylabel("rise, mV", color=BLUE)
    b3 = a3.twinx()
    b3.plot(
        d["t"], d["zig"], color=RED, lw=0.7, alpha=0.8, label="zig-zag over 1 min (mV)"
    )
    b3.axhline(DEFAULTS["activity_mv"], color=RED, lw=0.8, ls=":")
    b3.set_ylabel("zig-zag, mV", color=RED)
    b3.grid(False)
    b3.spines["right"].set_visible(True)
    a3.text(
        0.5,
        0.97,
        "rain evidence from the leaf (dotted lines: firmware thresholds)",
        transform=a3.transAxes,
        ha="center",
        va="top",
        fontsize=8,
        color=GRAY,
    )

    if d.attrs["has_air"]:
        a4.plot(d["t"], d["temp_c"], color=ORANGE, lw=1, label="temperature")
        a4.plot(
            d["t"],
            d["dewpt_c"],
            color=ORANGE,
            lw=1,
            ls="--",
            alpha=0.7,
            label="dew point",
        )
        a4.set_ylabel("degC")
        a4.legend(loc="center right", ncol=2)
        b4 = a4.twinx()
        b4.plot(d["t"], d["rh_pct"], color=SKY, lw=1)
        b4.set_ylabel("RH %", color=SKY)
        b4.grid(False)
        b4.spines["right"].set_visible(True)
    else:
        _note(a4, "no air data in this log")
    _time_axis(a4)
    fig.subplots_adjust(left=0.07, right=0.93, top=0.95, bottom=0.07)


def plot_event(fig, d, events, i, margin_min=45):
    """Zoom on one rain event: leaf reading and the gauge, with the moment each noticed it."""
    fig.clear()
    if events.empty or i is None or i >= len(events):
        _note(fig.subplots(), "no rain events to show")
        return
    ev = events.iloc[i]
    lo = ev["start"] - pd.Timedelta(minutes=margin_min)
    hi = ev["end"] + pd.Timedelta(minutes=margin_min)
    seg = d[(d["t"] >= lo) & (d["t"] <= hi)]
    a1, a2 = fig.subplots(
        2, 1, sharex=True, gridspec_kw=dict(height_ratios=[3, 1.4], hspace=0.1)
    )
    for s, e, st in state_runs(seg.reset_index(drop=True)):
        a1.axvspan(s, e, color=STATE_FILL[st], lw=0)
    a1.plot(seg["t"], seg["mv"], color="#1f2937", lw=1.1)
    sig = seg[seg["sig"]]
    a1.plot(sig["t"], sig["mv"], ".", color=RED, ms=3, label="rain signal")
    a1.axvline(ev["start"], color=RED, lw=1, ls="--", label="first rain signal")
    _tip_ticks(a1, seg, seg["mv"].min())
    a1.set_ylabel("leaf, mV")
    a1.set_title(
        f"Event {i + 1}: {ev['start']:%b %d %H:%M} to {ev['end']:%H:%M}   "
        f"{ev['minutes']:.0f} min, {ev['tips']:.0f} tips ({ev['mm']:.1f} mm)"
    )
    a1.legend(loc="upper left", ncol=3)
    a2.plot(seg["t"], seg["zig"], color=RED, lw=0.8, label="zig-zag")
    a2.plot(
        seg["t"],
        seg["rise10"].clip(lower=0) / 10,
        color=BLUE,
        lw=0.8,
        label="rise10 / 10",
    )
    a2.set_ylabel("mV")
    a2.legend(loc="upper left", ncol=2)
    _time_axis(a2)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.09)


def plot_intensity(fig, d):
    """Does the leaf's jitter tell you how hard it is raining? Compare with the gauge."""
    fig.clear()
    a1, a2 = fig.subplots(1, 2, gridspec_kw=dict(wspace=0.28))
    if not d.attrs["has_gauge"] or d["tips"].sum() == 0:
        _note(a1, "needs a log with rain gauge tips")
        _note(a2, "")
        return
    step = max(1, int(round(600 / d.attrs["dt"])))
    grp = np.arange(len(d)) // step
    w = (
        pd.DataFrame(
            {"tips": d["tips"].values, "zig": d["zig"].values, "mv": d["mv"].values}
        )
        .groupby(grp)
        .agg(tips=("tips", "sum"), zig=("zig", "mean"), mv=("mv", "mean"))
    )
    w = w[w["tips"].index >= 1]
    rate = w["tips"] * 0.1 * 6
    rng = np.random.default_rng(1)
    a1.scatter(
        w["zig"], rate + rng.normal(0, 0.05, len(w)), s=14, color=BLUE, alpha=0.5, lw=0
    )
    bins = [0, 0.1, 0.3, 0.6, 1, 2, 4, 100]
    cut = pd.cut(w["zig"], bins)
    m = (
        pd.DataFrame({"zig": w["zig"], "rate": rate})
        .groupby(cut, observed=True)
        .agg(zig=("zig", "mean"), rate=("rate", "mean"), n=("rate", "size"))
    )
    a1.plot(
        m["zig"],
        m["rate"],
        "o-",
        color=ORANGE,
        lw=1.6,
        ms=5,
        label="average per zig-zag band",
    )
    a1.set_xlabel("leaf zig-zag, mV per sample (10-minute average)")
    a1.set_ylabel("gauge rate, mm / hour")
    a1.set_title("leaf jitter vs. rain rate")
    a1.legend(loc="upper left")
    r = (
        np.corrcoef(w["zig"], w["tips"])[0, 1]
        if len(w) > 2 and w["tips"].std() > 0
        else np.nan
    )
    a1.text(
        0.98,
        0.04,
        f"correlation with tips: {r:.2f}",
        transform=a1.transAxes,
        ha="right",
        color=GRAY,
    )

    hourly = d.set_index("t")["tips"].resample("1h").sum() * 0.1
    zh = d.set_index("t")["zig"].resample("1h").mean()
    a2.bar(
        hourly.index,
        hourly.values,
        width=1 / 24 * 0.9,
        align="edge",
        color=BLUE,
        alpha=0.85,
    )
    a2.set_ylabel("gauge, mm per hour")
    b = a2.twinx()
    b.plot(
        zh.index + pd.Timedelta(minutes=30),
        zh.values,
        color=RED,
        lw=1.5,
        marker="o",
        ms=3,
    )
    b.set_ylabel("leaf zig-zag", color=RED)
    b.grid(False)
    b.spines["right"].set_visible(True)
    a2.set_title("hour by hour")
    _time_axis(a2)
    fig.subplots_adjust(left=0.07, right=0.93, top=0.92, bottom=0.14)


def plot_barcode(fig, d):
    """One row per day, one cell per hour: how the days looked, plus gauge tips and daily totals."""
    fig.clear()
    if not d.attrs["has_state"]:
        _note(fig.subplots(), "this log has no wetness state column")
        return
    day = d["t"].dt.date
    hour = d["t"].dt.hour
    frame = pd.DataFrame(
        {
            "day": day,
            "hour": hour,
            "rain": (d["state"] == "RAIN").astype(float),
            "wet": d["state"].isin(["RAIN", "WET", "DEW"]).astype(float),
            "tips": d["tips"],
        }
    )
    g = frame.groupby(["day", "hour"]).mean()
    tips = frame.groupby(["day", "hour"])["tips"].sum()
    days = sorted(frame["day"].unique())
    grid = np.full((len(days), 24), np.nan)
    dot = np.zeros((len(days), 24))
    for (dy, hr), row in g.iterrows():
        v = 2 if row["rain"] >= 0.1 else (1 if row["wet"] >= 0.5 else 0)
        grid[days.index(dy), hr] = v
        dot[days.index(dy), hr] = tips.loc[(dy, hr)]
    cmap = ListedColormap([STATE_FILL["DRY"], STATE_FILL["WET"], STATE_FILL["RAIN"]])
    cmap.set_bad("white")
    a, b = fig.subplots(1, 2, gridspec_kw=dict(width_ratios=[3.2, 1], wspace=0.08))
    a.grid(False)
    a.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
    yy, xx = np.nonzero(dot)
    a.scatter(
        xx,
        yy,
        s=18 + dot[yy, xx] * 6,
        color=ORANGE,
        edgecolor="white",
        lw=0.6,
        zorder=3,
        label="gauge tips (bigger = more)",
    )
    a.set_xticks(range(0, 24, 3))
    a.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 3)])
    a.set_yticks(range(len(days)))
    a.set_yticklabels([str(x) for x in days])
    a.set_title(
        "each hour of each day:  dry (sand)   wet (light blue, includes dew)   raining (blue)"
    )
    a.legend(loc="lower right")
    a.spines[:].set_visible(False)
    t = daily_table(d)
    b.barh(range(len(days)), t["rain_mm"], color=ORANGE)
    for i, v in enumerate(t["rain_mm"]):
        b.text(v, i, f" {v:.1f} mm", va="center", fontsize=8)
    b.set_ylim(len(days) - 0.5, -0.5)
    b.set_yticks([])
    b.set_xlim(0, max(1, t["rain_mm"].max()) * 1.4)
    b.set_title("gauge")
    fig.subplots_adjust(left=0.1, right=0.97, top=0.9, bottom=0.12)


def plot_temperature(fig, d):
    fig.clear()
    if not d.attrs["has_air"]:
        _note(fig.subplots(), "no air data in this log")
        return
    a1, a2 = fig.subplots(
        2, 1, sharex=True, gridspec_kw=dict(height_ratios=[1.3, 1], hspace=0.12)
    )
    a1.axhline(0, color=GRAY, lw=0.8)
    a1.plot(d["t"], d["temp_c"], color=ORANGE, lw=1.1, label="temperature (in the box)")
    a1.plot(d["t"], d["dewpt_c"], color=SKY, lw=1, ls="--", label="dew point")
    a1.fill_between(
        d["t"],
        d["temp_c"],
        d["dewpt_c"],
        color=SKY,
        alpha=0.12,
        label="dew point depression",
    )
    a1.set_ylabel("degC")
    a1.set_title(
        "air temperature (the BME680 inside the sealed box, so a rough guide only)"
    )
    a1.legend(loc="upper left", ncol=3)
    a2.plot(d["t"], d["dd"], color=GREEN, lw=1.8)
    a2.set_ylabel("degree-days above 0 degC")
    a2.set_title(f"cumulative degree-days  ({d.attrs['dd_source']})", fontsize=9)
    _time_axis(a2)
    t = daily_table(d)
    if len(t) > 1:
        ins = a2.inset_axes([0.02, 0.45, 0.28, 0.5])
        ins.bar(range(len(t)), t["dd"], color=GREEN, alpha=0.7)
        ins.set_xticks(range(len(t)))
        ins.set_xticklabels([str(x)[5:] for x in t["day"]])
        ins.set_title("per day", fontsize=8)
        ins.tick_params(labelsize=7)
        ins.grid(False)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.93, bottom=0.08)


def plot_tuning(fig, d, params):
    """Replay the firmware's leaf-only rain test with your own thresholds and score it against the gauge."""
    fig.clear()
    a1, a2 = fig.subplots(
        2, 1, sharex=True, gridspec_kw=dict(height_ratios=[3, 1.4], hspace=0.1)
    )
    r = replay_detector(d, **params)
    active = r["active"]
    a1.plot(d["t"], d["mv"], color="#1f2937", lw=0.9)
    lo, hi = d["mv"].min(), d["mv"].max()
    a1.fill_between(
        d["t"],
        lo,
        hi,
        where=active,
        color=BLUE,
        alpha=0.18,
        lw=0,
        label="rain active (replay)",
    )
    _tip_ticks(a1, d, lo - 0.02 * (hi - lo))
    a1.set_ylabel("leaf, mV")
    a1.set_title(
        "leaf-only rain detection with your thresholds (gauge tips shown for reference)"
    )
    a1.legend(loc="upper left", ncol=2)
    a2.plot(d["t"], r["zig"], color=RED, lw=0.7, label="zig-zag")
    a2.axhline(params["activity_mv"], color=RED, ls=":", lw=1)
    a2.plot(
        d["t"], np.clip(r["rise"], 0, None) / 10, color=BLUE, lw=0.7, label="rise / 10"
    )
    a2.axhline(params["rise_mv"] / 10, color=BLUE, ls=":", lw=1)
    a2.set_ylabel("mV")
    a2.set_ylim(0, max(2, min(8, np.nanpercentile(r["zig"], 99.5) * 1.5)))
    a2.legend(loc="upper left", ncol=2)
    _time_axis(a2)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.93, bottom=0.08)
    return score_detector(d, r, params["hold_min"])


def score_detector(d, r, hold_min):
    """Plain-language scorecard for a replayed detector."""
    t = d["uptime_s"].values.astype(float)
    sig = r["leaf_sig"]
    out = []
    tip_t = t[d["tips"].values > 0]
    if len(tip_t):
        covered = sum(
            bool(r["active"][np.searchsorted(t, x)])
            if np.searchsorted(t, x) < len(t)
            else False
            for x in tip_t
        )
        out.append(
            f"Gauge tips seen by the leaf: {covered} of {len(tip_t)}  ({100 * covered / len(tip_t):.0f}%)"
        )
    idx = np.flatnonzero(sig)
    if len(idx):
        breaks = np.flatnonzero(np.diff(t[idx]) > hold_min * 60)
        starts = np.r_[t[idx][0], t[idx][breaks + 1]]
        ends = np.r_[t[idx][breaks], t[idx][-1]]
        leads, unconfirmed = [], 0
        for a, b in zip(starts, ends):
            near = tip_t[(tip_t >= a - 600) & (tip_t <= b + 600)]
            if len(near):
                leads.append((near[0] - a) / 60)
            else:
                unconfirmed += 1
        out.append(f"Rain periods found: {len(starts)}")
        out.append(f"  with a gauge tip nearby: {len(starts) - unconfirmed}")
        out.append(f"  with no tip (drizzle or false alarm): {unconfirmed}")
        if leads:
            out.append(
                f"Leaf notices rain {np.median(leads):.0f} min before the first tip (median)"
            )
    else:
        out.append("No rain signals with these thresholds.")
    out.append(f"Rain 'active' {100 * r['active'].mean():.0f}% of the time")
    return "\n".join(out)


# ---------------------------------------------------------------------------
#  Command-line export (no window needed)
# ---------------------------------------------------------------------------


def export_png(d, outdir):
    os.makedirs(outdir, exist_ok=True)
    events = find_events(d)
    jobs = [
        ("timeline", lambda f: plot_timeline(f, d), (13, 9)),
        ("intensity", lambda f: plot_intensity(f, d), (13, 5)),
        (
            "days",
            lambda f: plot_barcode(f, d),
            (13, 4 + 0.5 * d["t"].dt.date.nunique()),
        ),
        ("temperature", lambda f: plot_temperature(f, d), (13, 7)),
        ("tuning", lambda f: plot_tuning(f, d, DEFAULTS), (13, 7)),
    ]
    if len(events):
        jobs.append(("event1", lambda f: plot_event(f, d, events, 0), (11, 6)))
    for name, fn, size in jobs:
        fig = Figure(figsize=size, dpi=110)
        fn(fig)
        path = os.path.join(
            outdir, f"{os.path.splitext(d.attrs['name'])[0]}_{name}.png"
        )
        fig.savefig(path)
        print("wrote", path)
    with open(
        os.path.join(outdir, os.path.splitext(d.attrs["name"])[0] + "_summary.txt"), "w"
    ) as f:
        f.write(narrative(d, events) + "\n\n" + summary_text(d, events) + "\n")


# ---------------------------------------------------------------------------
#  The window
# ---------------------------------------------------------------------------


def run_gui(paths, utc_offset):
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from tkinter import font as tkfont
    from matplotlib.backends.backend_tkagg import (
        FigureCanvasTkAgg,
        NavigationToolbar2Tk,
    )

    class Tip:
        """A small hover note for a widget."""

        def __init__(self, widget, text):
            self.widget, self.text, self.win = widget, text, None
            widget.bind("<Enter>", self.show)
            widget.bind("<Leave>", self.hide)

        def show(self, event=None):
            if self.win:
                return
            x = self.widget.winfo_rootx() + 12
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
            self.win = tk.Toplevel(self.widget)
            self.win.wm_overrideredirect(True)
            self.win.wm_geometry(f"+{x}+{y}")
            tk.Label(
                self.win,
                text=self.text,
                justify="left",
                background="#fffbe6",
                relief="solid",
                borderwidth=1,
                wraplength=320,
                padx=6,
                pady=4,
            ).pack()

        def hide(self, event=None):
            if self.win:
                self.win.destroy()
                self.win = None

    class PlotPane(ttk.Frame):
        """A matplotlib figure with its toolbar (zoom, pan, save as PNG), under a short note on what it shows."""

        def __init__(self, parent, app=None, tab=None, size=(9, 6)):
            super().__init__(parent)
            if app is not None:
                app.add_caption(self, tab)
            self.fig = Figure(figsize=size, dpi=100)
            self.canvas = FigureCanvasTkAgg(self.fig, master=self)
            NavigationToolbar2Tk(self.canvas, self).update()
            self.canvas.get_tk_widget().pack(fill="both", expand=True)

        def redraw(self):
            self.canvas.draw_idle()

    class App(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title("Rain Monitor viewer")
            self.geometry("1280x820")
            self.logs = {}  # file name -> path
            self.d = None
            self.events = pd.DataFrame()
            self.dirty = set()
            self.offset = tk.DoubleVar(value=utc_offset)
            self.gap = tk.DoubleVar(value=30.0)
            self.tune = {k: tk.DoubleVar(value=v) for k, v in DEFAULTS.items()}
            self._tune_job = None
            self._build()
            for p in paths:
                self.open_file(p)
            if not paths:
                self.tabs.select(self.t_help)  # first launch: start on the guide
            else:
                self.tabs.select(self.t_over)

        # ----- layout -----
        def _build(self):
            bar = ttk.Frame(self, padding=6)
            bar.pack(fill="x")
            ttk.Button(bar, text="Open CSV...", command=self.choose_files).pack(
                side="left"
            )
            ttk.Label(bar, text="  Log:").pack(side="left")
            self.picker = ttk.Combobox(bar, state="readonly", width=34)
            self.picker.pack(side="left")
            self.picker.bind("<<ComboboxSelected>>", lambda e: self.pick())
            ttk.Label(bar, text="   Show times as UTC +").pack(side="left")
            sp = ttk.Spinbox(
                bar,
                from_=-12,
                to=14,
                increment=1,
                width=5,
                textvariable=self.offset,
                command=self.reload_current,
            )
            sp.pack(side="left")
            sp.bind("<Return>", lambda e: self.reload_current())
            ttk.Label(bar, text="h  (log times are UTC)", foreground="#6b7280").pack(
                side="left"
            )
            Tip(self.picker, "Switch between the logs you have opened.")
            Tip(
                sp,
                "The logger writes UTC. Enter your hours from UTC (Pacific standard time is -8) "
                "and every plot and table is shown in local time.",
            )

            # Color key that stays visible on every tab
            key = ttk.Frame(bar)
            key.pack(side="right")
            ttk.Label(key, text="Key: ").pack(side="left")
            for state, note in (
                ("DRY", "leaf dry"),
                ("WET", "wet, rain stopped"),
                ("DEW", "wet, no rain: dew or fog"),
                ("RAIN", "wet and raining"),
            ):
                lab = tk.Label(
                    key,
                    text=f" {state} ",
                    background=STATE_FILL[state],
                    relief="solid",
                    borderwidth=1,
                )
                lab.pack(side="left", padx=2)
                Tip(lab, note)
            gauge = tk.Label(key, text=" | gauge tip ", foreground=ORANGE)
            gauge.pack(side="left", padx=2)
            Tip(
                gauge,
                "Orange ticks and dots in the plots are rain gauge tips: 0.1 mm each.",
            )

            menubar = tk.Menu(self)
            helpmenu = tk.Menu(menubar, tearoff=0)
            helpmenu.add_command(
                label="Guide to this tab   (F1)", command=self.help_for_current
            )
            helpmenu.add_command(
                label="What am I looking at?", command=lambda: self.show_help("start")
            )
            helpmenu.add_command(
                label="Color key", command=lambda: self.show_help("colors")
            )
            helpmenu.add_command(
                label="Glossary", command=lambda: self.show_help("terms")
            )
            helpmenu.add_command(
                label="Things to keep in mind",
                command=lambda: self.show_help("caveats"),
            )
            menubar.add_cascade(label="Help", menu=helpmenu)
            self.config(menu=menubar)
            self.bind("<F1>", lambda e: self.help_for_current())

            self.tabs = ttk.Notebook(self)
            self.tabs.pack(fill="both", expand=True)
            self.tabs.bind("<<NotebookTabChanged>>", lambda e: self.draw_visible())

            # Help
            self.t_help = ttk.Frame(self.tabs)
            self.tabs.add(self.t_help, text="Help")
            scroll = ttk.Scrollbar(self.t_help)
            scroll.pack(side="right", fill="y")
            self.help_text = tk.Text(
                self.t_help,
                wrap="word",
                padx=16,
                pady=12,
                yscrollcommand=scroll.set,
                font=("TkDefaultFont", 11),
                spacing3=6,
                cursor="arrow",
            )
            self.help_text.pack(fill="both", expand=True)
            scroll.config(command=self.help_text.yview)
            self.render_help()

            # Overview
            self.t_over = ttk.Frame(self.tabs)
            self.tabs.add(self.t_over, text="Overview")
            self.add_caption(self.t_over, "Overview")
            paned = ttk.PanedWindow(self.t_over, orient="horizontal")
            paned.pack(fill="both", expand=True)
            body_font, mono_font = self.pick_fonts()
            self.text = tk.Text(
                paned,
                font=body_font,
                wrap="word",
                width=70,
                padx=12,
                pady=10,
                spacing1=2,
                spacing3=2,
                cursor="arrow",
            )
            self.text.tag_configure(
                "head",
                font=(body_font[0], body_font[1], "bold"),
                spacing1=10,
                spacing3=4,
            )
            self.text.tag_configure("mono", font=mono_font)
            paned.add(self.text, weight=3)
            right = ttk.Frame(paned)
            paned.add(right, weight=2)
            ttk.Label(
                right, text="Day by day", font=("TkDefaultFont", 10, "bold")
            ).pack(anchor="w", padx=6, pady=4)
            cols = ("day", "rain mm", "rain h", "wet h", "T min", "T max", "deg-days")
            self.day_tree = ttk.Treeview(
                right, columns=cols, show="headings", height=14
            )
            for c in cols:
                self.day_tree.heading(c, text=c)
                self.day_tree.column(c, width=70, anchor="e")
            self.day_tree.column("day", width=90, anchor="w")
            self.day_tree.pack(fill="both", expand=True, padx=6)

            # Timeline
            self.timeline = PlotPane(self.tabs, self, "Timeline")
            self.tabs.add(self.timeline, text="Timeline")

            # Events
            self.t_events = ttk.Frame(self.tabs)
            self.tabs.add(self.t_events, text="Rain events")
            self.add_caption(self.t_events, "Rain events")
            top = ttk.Frame(self.t_events, padding=6)
            top.pack(fill="x")
            ttk.Label(top, text="Merge signals closer than").pack(side="left")
            g = ttk.Spinbox(
                top,
                from_=5,
                to=180,
                increment=5,
                width=5,
                textvariable=self.gap,
                command=self.refresh_events,
            )
            g.pack(side="left", padx=4)
            g.bind("<Return>", lambda e: self.refresh_events())
            Tip(
                g,
                "Rain signals closer together than this are one event. A storm with a short pause stays one event; "
                "lower the number to split it.",
            )
            ttk.Label(
                top, text="minutes into one event.  Click an event to zoom in."
            ).pack(side="left")
            ecols = (
                "#",
                "start",
                "end",
                "min",
                "tips",
                "mm",
                "leaf leads gauge (min)",
                "peak rise mV",
                "kind",
            )
            self.ev_tree = ttk.Treeview(
                self.t_events, columns=ecols, show="headings", height=7
            )
            for c in ecols:
                self.ev_tree.heading(c, text=c)
                self.ev_tree.column(c, width=90, anchor="e")
            self.ev_tree.column("start", width=130, anchor="w")
            self.ev_tree.column("end", width=130, anchor="w")
            self.ev_tree.column("kind", width=100, anchor="w")
            self.ev_tree.column("#", width=40)
            self.ev_tree.pack(fill="x", padx=6)
            self.ev_tree.bind("<<TreeviewSelect>>", lambda e: self.draw_event())
            self.event_pane = PlotPane(self.t_events)
            self.event_pane.pack(fill="both", expand=True)

            # Other plots
            self.intensity = PlotPane(self.tabs, self, "Rain intensity")
            self.tabs.add(self.intensity, text="Rain intensity")
            self.barcode = PlotPane(self.tabs, self, "Day by hour")
            self.tabs.add(self.barcode, text="Day by hour")
            self.temp = PlotPane(self.tabs, self, "Temperature and degree-days")
            self.tabs.add(self.temp, text="Temperature and degree-days")

            # Tuning
            self.t_tune = ttk.Frame(self.tabs)
            self.tabs.add(self.t_tune, text="Tune the detector")
            self.add_caption(self.t_tune, "Tune the detector")
            panel = ttk.Frame(self.t_tune, padding=10)
            panel.pack(side="left", fill="y")
            ttk.Label(
                panel,
                text="Replay the firmware's leaf rain test\nwith your own thresholds.\n"
                "The gauge is used only to score it.",
                justify="left",
            ).pack(anchor="w", pady=(0, 10))
            self.tune_labels = {}
            specs = [
                (
                    "rise_mv",
                    "RAIN_RISE_MV: rise in 10 min (mV)",
                    1,
                    20,
                    "How far the reading must climb within the rise window to count as water arriving. Lower finds "
                    "lighter rain but risks false alarms from drift. Firmware: 4 mV.",
                ),
                (
                    "activity_mv",
                    "RAIN_ACTIVITY_MV: zig-zag (mV)",
                    0.05,
                    2,
                    "How much the reading must jump up and down per sample over the last minute (drops landing and "
                    "running off). Firmware: 0.3 mV.",
                ),
                (
                    "zigzag_samples",
                    "ZIGZAG_SAMPLES (x 10 s)",
                    3,
                    18,
                    "Length of the zig-zag window in 10-second samples. Shorter = the signal stops sooner after the "
                    "rain does. Firmware: 6 (one minute).",
                ),
                (
                    "rise_samples",
                    "RISE_WINDOW_SAMPLES (x 10 s)",
                    12,
                    120,
                    "Length of the rise window in 10-second samples. Firmware: 60 (ten minutes).",
                ),
                (
                    "hold_min",
                    "RAIN_HOLD (minutes)",
                    2,
                    90,
                    "How long rain stays 'active' after the last signal. Firmware: 30. The longest quiet gap inside "
                    "one real storm in the sample logs was 27 minutes.",
                ),
            ]
            for key, label, lo, hi, tip in specs:
                lab = ttk.Label(panel, text=label)
                lab.pack(anchor="w")
                Tip(lab, tip)
                row = ttk.Frame(panel)
                row.pack(fill="x", pady=(0, 8))
                ttk.Scale(
                    row,
                    from_=lo,
                    to=hi,
                    variable=self.tune[key],
                    length=210,
                    command=lambda v, k=key: self.tune_changed(k),
                ).pack(side="left")
                self.tune_labels[key] = ttk.Label(row, width=7)
                self.tune_labels[key].pack(side="left", padx=6)
            ttk.Button(
                panel, text="Reset to firmware defaults", command=self.tune_reset
            ).pack(anchor="w", pady=6)
            self.score = tk.Label(
                panel,
                justify="left",
                anchor="w",
                font=("Courier", 9),
                bg="#f3f4f6",
                padx=8,
                pady=8,
            )
            self.score.pack(fill="x", pady=10)
            self.tune_pane = PlotPane(self.t_tune)
            self.tune_pane.pack(side="left", fill="both", expand=True)
            self.update_tune_labels()

            self.status = ttk.Label(
                self,
                text="Open a RAINnnnn.CSV (or older WETnnnn.CSV) log to begin.",
                padding=4,
            )
            self.status.pack(fill="x")

        # ----- help -----
        def add_caption(self, parent, tab):
            """A one-or-two-line note at the top of a tab, with a link to the full help."""
            bar = ttk.Frame(parent, padding=(8, 6))
            bar.pack(side="top", fill="x")
            ttk.Button(bar, text="More help", command=lambda: self.show_help(tab)).pack(
                side="right", padx=(8, 0)
            )
            ttk.Label(
                bar,
                text=CAPTIONS[tab],
                wraplength=1080,
                justify="left",
                foreground="#374151",
            ).pack(side="left", fill="x", expand=True)

        def render_help(self):
            t = self.help_text
            t.tag_configure(
                "h1", font=("TkDefaultFont", 15, "bold"), spacing1=14, spacing3=6
            )
            t.tag_configure("term", font=("TkDefaultFont", 11, "bold"))
            t.tag_configure("bullet", lmargin1=24, lmargin2=40)
            for state, color in STATE_FILL.items():
                t.tag_configure(
                    "sw_" + state, background=color, borderwidth=1, relief="solid"
                )
            for key, title, items in HELP:
                t.mark_set("sec_" + key, "end-1c")
                t.mark_gravity("sec_" + key, "left")
                t.insert("end", title + "\n", "h1")
                for item in items:
                    if isinstance(item, str):
                        t.insert("end", item + "\n")
                    elif item[0] == "swatch":
                        t.insert("end", "   " + item[1] + "   ", "sw_" + item[1])
                        t.insert("end", "  " + item[2] + "\n")
                    elif item[0] == "term":
                        t.insert("end", item[1] + ".  ", "term")
                        t.insert("end", item[2] + "\n")
                    elif item[0] == "bullet":
                        t.insert("end", "\u2022  " + item[1] + "\n", "bullet")
            t.config(state="disabled")

        def show_help(self, key):
            """Jump to a section of the Help tab: a section key, or a tab name."""
            keys = {k for k, _, _ in HELP}
            if key not in keys:
                key = "start"
            self.tabs.select(self.t_help)
            self.help_text.yview("sec_" + key)

        def help_for_current(self):
            name = self.tabs.tab(self.tabs.select(), "text")
            self.show_help(name)

        # ----- loading -----
        def choose_files(self):
            files = filedialog.askopenfilenames(
                title="Open rain monitor logs",
                filetypes=[("CSV logs", "*.csv *.CSV"), ("All files", "*.*")],
            )
            for f in files:
                self.open_file(f)

        def pick_fonts(self):
            """A readable proportional font for prose and a clear monospaced one for aligned columns."""
            families = set(tkfont.families())

            def first(names, fallback):
                return next((n for n in names if n in families), fallback)

            body = first(
                [
                    "Segoe UI",
                    "SF Pro Text",
                    "Helvetica Neue",
                    "Ubuntu",
                    "DejaVu Sans",
                    "Arial",
                ],
                "TkDefaultFont",
            )
            mono = first(
                [
                    "Consolas",
                    "Menlo",
                    "DejaVu Sans Mono",
                    "Ubuntu Mono",
                    "Liberation Mono",
                ],
                "TkFixedFont",
            )
            return (body, 12), (mono, 11)

        def open_file(self, path):
            try:
                d = load_log(path, self.offset.get())
            except Exception as exc:  # a bad file should not end the app
                messagebox.showerror(
                    "Could not open log", f"{os.path.basename(path)}\n\n{exc}"
                )
                return
            name = os.path.basename(path)
            self.logs[name] = path
            self.picker["values"] = list(self.logs)
            self.picker.set(name)
            self.set_data(d)
            self.tabs.select(self.t_over)  # a newly opened log starts on the Overview

        def pick(self):
            self.reload_current()

        def reload_current(self):
            name = self.picker.get()
            if name in self.logs:
                try:
                    self.set_data(load_log(self.logs[name], self.offset.get()))
                except Exception as exc:
                    messagebox.showerror("Could not open log", str(exc))

        def set_data(self, d):
            self.d = d
            self.refresh_events()
            self.text.delete("1.0", "end")
            self.text.insert("end", "IN PLAIN WORDS\n", "head")
            self.text.insert("end", narrative(d, self.events) + "\n\n")
            self.text.insert("end", "THE DETAILS\n", "head")
            self.text.insert("end", summary_text(d, self.events), "mono")
            for row in self.day_tree.get_children():
                self.day_tree.delete(row)
            for _, r in daily_table(d).iterrows():
                self.day_tree.insert(
                    "",
                    "end",
                    values=(
                        str(r["day"]),
                        f"{r['rain_mm']:.1f}",
                        f"{r['rain_h']:.1f}",
                        f"{r['wet_h']:.1f}",
                        "" if pd.isna(r["tmin"]) else f"{r['tmin']:.1f}",
                        "" if pd.isna(r["tmax"]) else f"{r['tmax']:.1f}",
                        "" if pd.isna(r["dd"]) else f"{r['dd']:.2f}",
                    ),
                )
            self.dirty = {"timeline", "intensity", "barcode", "temp", "tune"}
            self.status.config(
                text=f"{d.attrs['name']}:  {len(d):,} rows,  {d['t'].iloc[0]:%Y-%m-%d %H:%M} to "
                f"{d['t'].iloc[-1]:%Y-%m-%d %H:%M}   (local offset UTC{self.offset.get():+g})"
            )
            self.draw_visible()

        # ----- events -----
        def refresh_events(self):
            if self.d is None:
                return
            self.events = find_events(self.d, self.gap.get())
            for row in self.ev_tree.get_children():
                self.ev_tree.delete(row)
            for i, e in self.events.iterrows():
                self.ev_tree.insert(
                    "",
                    "end",
                    iid=str(i),
                    values=(
                        i + 1,
                        f"{e['start']:%m-%d %H:%M}",
                        f"{e['end']:%m-%d %H:%M}",
                        f"{e['minutes']:.0f}",
                        int(e["tips"]),
                        f"{e['mm']:.1f}",
                        "" if pd.isna(e["lead_min"]) else f"{e['lead_min']:.0f}",
                        f"{e['peak_rise_mv']:.0f}",
                        e["kind"],
                    ),
                )
            if len(self.events):
                self.ev_tree.selection_set("0")
            self.dirty.add("event")
            self.draw_event()

        def draw_event(self):
            if self.d is None:
                return
            sel = self.ev_tree.selection()
            plot_event(
                self.event_pane.fig, self.d, self.events, int(sel[0]) if sel else None
            )
            self.event_pane.redraw()

        # ----- tuning -----
        def params(self):
            p = {k: v.get() for k, v in self.tune.items()}
            for k in ("zigzag_samples", "rise_samples"):
                p[k] = int(round(p[k]))
            return p

        def update_tune_labels(self):
            for k, lab in self.tune_labels.items():
                v = self.params()[k]
                lab.config(text=f"{v:g}" if isinstance(v, int) else f"{v:.2f}")

        def tune_changed(self, key):
            self.update_tune_labels()
            if self._tune_job:
                self.after_cancel(self._tune_job)
            self._tune_job = self.after(
                150, self.draw_tune
            )  # wait until the slider settles

        def tune_reset(self):
            for k, v in DEFAULTS.items():
                self.tune[k].set(v)
            self.update_tune_labels()
            self.draw_tune()

        def draw_tune(self):
            if self.d is None:
                return
            self.score.config(
                text=plot_tuning(self.tune_pane.fig, self.d, self.params())
            )
            self.tune_pane.redraw()

        # ----- drawing only what you can see -----
        def draw_visible(self):
            if self.d is None:
                return
            current = self.tabs.nametowidget(self.tabs.select())
            jobs = {
                self.timeline: ("timeline", plot_timeline),
                self.intensity: ("intensity", plot_intensity),
                self.barcode: ("barcode", plot_barcode),
                self.temp: ("temp", plot_temperature),
            }
            if current in jobs and jobs[current][0] in self.dirty:
                key, fn = jobs[current]
                fn(current.fig, self.d)
                current.redraw()
                self.dirty.discard(key)
            elif current is self.t_tune and "tune" in self.dirty:
                self.draw_tune()
                self.dirty.discard("tune")

    App().mainloop()


def main():
    ap = argparse.ArgumentParser(description="View rain monitor CSV logs.")
    ap.add_argument("files", nargs="*", help="log files to open")
    ap.add_argument(
        "--png", metavar="DIR", help="do not open a window: write every plot to DIR"
    )
    ap.add_argument(
        "--utc-offset",
        type=float,
        default=None,
        help="hours to add to the UTC log times (default: this computer's offset)",
    )
    a = ap.parse_args()
    offset = (
        a.utc_offset if a.utc_offset is not None else time.localtime().tm_gmtoff / 3600
    )
    if a.png:
        if not a.files:
            ap.error("--png needs at least one log file")
        for f in a.files:
            export_png(load_log(f, offset), a.png)
    else:
        run_gui(a.files, offset)


if __name__ == "__main__":
    main()
