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

# Colour-blind-safe palette (Okabe-Ito)
BLUE, ORANGE, GREEN, RED, PURPLE, SKY, GREY = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#6b7280",
)

STATE_COLOURS = {
    "RAIN": "#2b6cb0",
    "WET": "#90cdf4",
    "DEW": "#38b2ac",
}  # DRY is left unshaded

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
    """Distance travelled over the last w samples minus the net change, per step.
    A smooth rise or fall scores 0; drops landing and running off score high."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    out = np.zeros(n)
    if n < w or w < 2:
        return out
    cs = np.cumsum(np.abs(np.diff(x, prepend=x[0])))
    travelled = cs[w - 1 :] - cs[: n - w + 1]
    net = np.abs(x[w - 1 :] - x[: n - w + 1])
    out[w - 1 :] = (travelled - net) / (w - 1)
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
    """Yield (start, end, state) for each stretch of one non-DRY state."""
    s = d["state"].values
    t = d["t"].values
    if len(s) == 0:
        return
    start = 0
    for i in range(1, len(s) + 1):
        if i == len(s) or s[i] != s[start]:
            if s[start] in STATE_COLOURS:
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


# ---------------------------------------------------------------------------
#  Plots. Each takes an empty Figure and a loaded log, and draws on it.
# ---------------------------------------------------------------------------


def _time_axis(ax):
    loc = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


def _note(ax, text):
    ax.text(
        0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", color=GREY
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
        a1.axvspan(s, e, color=STATE_COLOURS[st], alpha=0.22, lw=0)
    a1.plot(d["t"], d["mv"], color="#1f2937", lw=0.9, label="leaf reading")
    if d["baseline_mv"].notna().any():
        a1.plot(
            d["t"], d["baseline_mv"], color=GREY, lw=1, ls="--", label="dry baseline"
        )
    _tip_ticks(a1, d, d["mv"].min() - 0.02 * (d["mv"].max() - d["mv"].min()))
    a1.set_ylabel("leaf, mV")
    a1.set_title(f"{d.attrs['name']}  -  leaf wetness, rain and air")
    handles, labels = a1.get_legend_handles_labels()
    if d.attrs["has_state"]:
        from matplotlib.patches import Patch

        for st, c in STATE_COLOURS.items():
            if (d["state"] == st).any():
                handles.append(Patch(color=c, alpha=0.4))
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
        color=GREY,
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
        a1.axvspan(s, e, color=STATE_COLOURS[st], alpha=0.22, lw=0)
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
        color=GREY,
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
    cmap = ListedColormap(["#f1ede4", "#9ecae1", "#08519c"])
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
        "each hour of each day:  dry (sand)   wet (light blue)   raining (dark blue)"
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
    a1.axhline(0, color=GREY, lw=0.8)
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
        f.write(summary_text(d, events) + "\n")


# ---------------------------------------------------------------------------
#  The window
# ---------------------------------------------------------------------------


def run_gui(paths, utc_offset):
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from matplotlib.backends.backend_tkagg import (
        FigureCanvasTkAgg,
        NavigationToolbar2Tk,
    )

    class PlotPane(ttk.Frame):
        """A matplotlib figure with its toolbar (zoom, pan, save as PNG)."""

        def __init__(self, parent, size=(9, 6)):
            super().__init__(parent)
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

            self.tabs = ttk.Notebook(self)
            self.tabs.pack(fill="both", expand=True)
            self.tabs.bind("<<NotebookTabChanged>>", lambda e: self.draw_visible())

            # Overview
            self.t_over = ttk.Frame(self.tabs)
            self.tabs.add(self.t_over, text="Overview")
            paned = ttk.PanedWindow(self.t_over, orient="horizontal")
            paned.pack(fill="both", expand=True)
            self.text = tk.Text(paned, font=("Courier", 10), wrap="none", width=70)
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
            self.timeline = PlotPane(self.tabs)
            self.tabs.add(self.timeline, text="Timeline")

            # Events
            self.t_events = ttk.Frame(self.tabs)
            self.tabs.add(self.t_events, text="Rain events")
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
            self.intensity = PlotPane(self.tabs)
            self.tabs.add(self.intensity, text="Rain intensity")
            self.barcode = PlotPane(self.tabs)
            self.tabs.add(self.barcode, text="Day by hour")
            self.temp = PlotPane(self.tabs)
            self.tabs.add(self.temp, text="Temperature and degree-days")

            # Tuning
            self.t_tune = ttk.Frame(self.tabs)
            self.tabs.add(self.t_tune, text="Tune the detector")
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
                ("rise_mv", "RAIN_RISE_MV: rise in 10 min (mV)", 1, 20),
                ("activity_mv", "RAIN_ACTIVITY_MV: zig-zag (mV)", 0.05, 2),
                ("zigzag_samples", "ZIGZAG_SAMPLES (x 10 s)", 3, 18),
                ("rise_samples", "RISE_WINDOW_SAMPLES (x 10 s)", 12, 120),
                ("hold_min", "RAIN_HOLD (minutes)", 2, 90),
            ]
            for key, label, lo, hi in specs:
                lab = ttk.Label(panel, text=label)
                lab.pack(anchor="w")
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

        # ----- loading -----
        def choose_files(self):
            files = filedialog.askopenfilenames(
                title="Open rain monitor logs",
                filetypes=[("CSV logs", "*.csv *.CSV"), ("All files", "*.*")],
            )
            for f in files:
                self.open_file(f)

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
            self.text.insert("1.0", summary_text(d, self.events))
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
