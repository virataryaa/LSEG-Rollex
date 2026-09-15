"""
rollex_builder_lseg.py — Roll-Adjusted Time Series built from LSEG API
=========================================================================
LSEG-API replacement for ICEBREAKER/Rollex/Code/rollex_builder.py (icepython-
based). All contract-calendar / roll-regime / back-adjustment math is
UNCHANGED, byte-for-byte, from the ICE source — it is 100% vendor-agnostic
pandas/numpy operating on whatever c1/c2 OHLC gets fetched. The only thing
that changes is fetch_ohlc() (icepython -> lseg.data) and the c1/c2 RIC map.

Usage:
    python rollex_builder_lseg.py           # incremental update
    python rollex_builder_lseg.py --full     # full rebuild from START_DATE
    python rollex_builder_lseg.py --commodity KC
"""

import argparse
import calendar as cal_module
import numpy as np
import pandas as pd
pd.set_option("future.no_silent_downcasting", True)  # silences a harmless lseg.data internal FutureWarning
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List
from pandas.tseries.holiday import (
    AbstractHolidayCalendar, Holiday, nearest_workday,
    USMartinLutherKingJr, USPresidentsDay, GoodFriday, EasterMonday,
    USMemorialDay, USLaborDay, USThanksgivingDay,
)
from pandas.tseries.offsets import CustomBusinessDay

# ── PATHS ─────────────────────────────────────────────────────────────────────
CODE_DIR = Path(__file__).resolve().parent
DB_DIR   = CODE_DIR.parent / "Database"
DB_DIR.mkdir(exist_ok=True)

# ── CONFIG ────────────────────────────────────────────────────────────────────
OFFSET     = 20           # trading days before LTD to switch c1 -> c2
START_DATE = "2010-01-01"
START_YEAR = 2009

# LSEG continuation RICs. RC (Robusta) and LCC/LSU are ICE Futures Europe
# (LIFFE) contracts on LSEG's own "LRC"/"LCC"/"LSU" ticker roots — confirmed
# working RICs from the COT migration probe, same tickers apply here.
COMMODITIES = {
    "KC":  {"c1": "KCc1",  "c2": "KCc2",  "engine_key": "KC"},
    "RC":  {"c1": "LRCc1", "c2": "LRCc2", "engine_key": "RC"},
    "CC":  {"c1": "CCc1",  "c2": "CCc2",  "engine_key": "CC"},
    "LCC": {"c1": "LCCc1", "c2": "LCCc2", "engine_key": "LCC"},
    "SB":  {"c1": "SBc1",  "c2": "SBc2",  "engine_key": "SB"},
    "CT":  {"c1": "CTc1",  "c2": "CTc2",  "engine_key": "CT"},
    "LSU": {"c1": "LSUc1", "c2": "LSUc2", "engine_key": "LSU"},
}

# ── HOLIDAY CALENDARS (unchanged from ICE source) ───────────────────────────

class USExchangeHolidayCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday("NewYearsDay",     month=1,  day=1,  observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth",      month=6,  day=19, observance=nearest_workday),
        Holiday("IndependenceDay", month=7,  day=4,  observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas",       month=12, day=25, observance=nearest_workday),
    ]


def _build_uk_holidays(start_year: int = 2004, end_year: int = 2040) -> pd.DatetimeIndex:
    base = AbstractHolidayCalendar(rules=[
        Holiday("NewYearsDay", month=1,  day=1,  observance=nearest_workday),
        GoodFriday,
        EasterMonday,
        Holiday("Christmas",   month=12, day=25, observance=nearest_workday),
        Holiday("BoxingDay",   month=12, day=26, observance=nearest_workday),
    ])
    hols = list(base.holidays(
        start=pd.Timestamp(f"{start_year}-01-01"),
        end=pd.Timestamp(f"{end_year}-12-31"),
    ))
    for year in range(start_year, end_year + 1):
        d = pd.Timestamp(year, 5, 1)
        while d.dayofweek != 0:
            d += pd.Timedelta(days=1)
        hols.append(d)
        d = pd.Timestamp(year, 5, cal_module.monthrange(year, 5)[1])
        while d.dayofweek != 0:
            d -= pd.Timedelta(days=1)
        hols.append(d)
        d = pd.Timestamp(year, 8, cal_module.monthrange(year, 8)[1])
        while d.dayofweek != 0:
            d -= pd.Timedelta(days=1)
        hols.append(d)
    return pd.DatetimeIndex(sorted(set(hols)))


US_BDAY  = CustomBusinessDay(calendar=USExchangeHolidayCalendar())
UK_BDAY  = CustomBusinessDay(holidays=_build_uk_holidays())
BDAY_CAL = {"US": US_BDAY, "UK": UK_BDAY}

# ── DATE HELPERS (unchanged) ────────────────────────────────────────────────

def first_bd(year, month, bday):
    d = pd.Timestamp(year=year, month=month, day=1).normalize()
    while d != (d + 0 * bday):
        d += pd.Timedelta(days=1)
    return d

def last_bd(year, month, bday):
    last_day = cal_module.monthrange(year, month)[1]
    d = pd.Timestamp(year=year, month=month, day=last_day).normalize()
    while d != (d + 0 * bday):
        d -= pd.Timedelta(days=1)
    return d

def nth_bd(year, month, n, bday):
    d = first_bd(year, month, bday)
    for _ in range(n - 1):
        d = (d + 1 * bday).normalize()
    return d

def preceding_month(year, month):
    return (year - 1, 12) if month == 1 else (year, month - 1)

# ── FND / LTD RULE FACTORIES (unchanged) ────────────────────────────────────

def fnd_first_bd_minus(n):
    def calc(year, month, bday):
        return (first_bd(year, month, bday) - n * bday).normalize()
    return calc

def fnd_nth_bd_minus(nth, n):
    def calc(year, month, bday):
        return (nth_bd(year, month, nth, bday) - n * bday).normalize()
    return calc

def fnd_first_bd():
    def calc(year, month, bday):
        return first_bd(year, month, bday)
    return calc

def ltd_last_bd_minus(n):
    def calc(year, month, bday):
        return (last_bd(year, month, bday) - n * bday).normalize()
    return calc

def ltd_last_bd_preceding_month():
    def calc(year, month, bday):
        py, pm = preceding_month(year, month)
        return last_bd(py, pm, bday)
    return calc

def ltd_calendar_days_before_month_start(n, roll="preceding"):
    def calc(year, month, bday):
        d = (pd.Timestamp(year, month, 1) - pd.Timedelta(days=n)).normalize()
        if roll == "preceding":
            while d != (d + 0 * bday):
                d -= pd.Timedelta(days=1)
        else:
            while d != (d + 0 * bday):
                d += pd.Timedelta(days=1)
        return d
    return calc

# ── COMMODITY CONFIG (unchanged) ────────────────────────────────────────────

@dataclass
class CommodityConfig:
    months:    List[str]
    month_num: Dict[str, int]
    calendar:  str
    fnd_rule:  object
    ltd_rule:  Callable

COMMODITY_CONFIG = {
    "KC": CommodityConfig(
        months=["H","K","N","U","Z"], month_num={"H":3,"K":5,"N":7,"U":9,"Z":12},
        calendar="US", fnd_rule=fnd_first_bd_minus(7), ltd_rule=ltd_last_bd_minus(8),
    ),
    "CC": CommodityConfig(
        months=["H","K","N","U","Z"], month_num={"H":3,"K":5,"N":7,"U":9,"Z":12},
        calendar="US", fnd_rule=fnd_nth_bd_minus(nth=6, n=10), ltd_rule=ltd_last_bd_minus(11),
    ),
    # October (V) excluded: too illiquid — c1/c2 continuation rolling into it
    # produces noisy, non-representative prints. Roll goes Jul -> Dec directly.
    # See fetch_explicit_chain(), used only for CT in main(), which sources
    # c1/c2 from real per-contract RICs (never LSEG's generic CTc1/CTc2) so
    # the excluded month never enters the price series either.
    "CT": CommodityConfig(
        months=["H","K","N","Z"], month_num={"H":3,"K":5,"N":7,"Z":12},
        calendar="US", fnd_rule=fnd_first_bd_minus(5), ltd_rule=ltd_last_bd_minus(17),
    ),
    "SB": CommodityConfig(
        months=["H","K","N","V"], month_num={"H":3,"K":5,"N":7,"V":10},
        calendar="US", fnd_rule="after_ltd", ltd_rule=ltd_last_bd_preceding_month(),
    ),
    "OJ": CommodityConfig(
        months=["F","H","K","N","U","X"], month_num={"F":1,"H":3,"K":5,"N":7,"U":9,"X":11},
        calendar="US", fnd_rule=fnd_first_bd(), ltd_rule=ltd_last_bd_minus(14),
    ),
    "RC": CommodityConfig(
        months=["F","H","K","N","U","X"], month_num={"F":1,"H":3,"K":5,"N":7,"U":9,"X":11},
        calendar="UK", fnd_rule=fnd_first_bd_minus(4), ltd_rule=ltd_last_bd_minus(4),
    ),
    "LCC": CommodityConfig(
        months=["H","K","N","U","Z"], month_num={"H":3,"K":5,"N":7,"U":9,"Z":12},
        calendar="UK", fnd_rule="after_ltd", ltd_rule=ltd_last_bd_minus(11),
    ),
    "LSU": CommodityConfig(
        months=["H","K","Q","V","Z"], month_num={"H":3,"K":5,"Q":8,"V":10,"Z":12},
        calendar="UK",
        fnd_rule=ltd_calendar_days_before_month_start(15, roll="following"),
        ltd_rule=ltd_calendar_days_before_month_start(16, roll="preceding"),
    ),
}

# ── CONTRACT TABLE (unchanged) ──────────────────────────────────────────────

def generate_contract_table(commodity, start_year, end_year):
    cfg  = COMMODITY_CONFIG[commodity]
    bday = BDAY_CAL[cfg.calendar]
    rows = []
    for year in range(start_year, end_year + 1):
        for month_code in cfg.months:
            delivery_month = cfg.month_num[month_code]
            ltd = cfg.ltd_rule(year, delivery_month, bday)
            fnd = (
                (ltd + 1 * bday).normalize()
                if cfg.fnd_rule == "after_ltd"
                else cfg.fnd_rule(year, delivery_month, bday)
            )
            rows.append({"month": month_code, "year": year, "FND": fnd, "LTD": ltd})
    return pd.DataFrame(rows).sort_values("LTD").reset_index(drop=True)

# ── LSEG FETCH (replaces icepython fetch_ohlc) ──────────────────────────────

def fetch_ohlc(symbol, start, end, ld, bday=None):
    """Fetch OHLC from LSEG. LSEG's continuation RICs have real gaps on days
    the exchange was open (confirmed by spot-checking against the ICE feed —
    e.g. KCc1 is missing both 2025-09-17 and 2025-09-18 entirely, not just on
    a batch-fetch quirk). Since those are internal holes in an otherwise-daily
    series (not exchange closures), we reindex onto the commodity's own
    business-day calendar and linearly interpolate — but only between two
    real prints (limit_area='inside'), never extrapolating at either edge.
    This prevents a multi-day data gap from being compressed into a single
    freakishly large daily return, particularly damaging when a gap lands on
    or near a roll date."""
    try:
        # SETTLE, not TRDPRC_1 — TRDPRC_1 is a last-traded-price snapshot that
        # keeps moving/correcting after a pull (confirmed against Eikon: a
        # KCZ6 pull showed 289.95 while Eikon's own settled TRDPRC_1 table
        # later read 291.30 for the same date). SETTLE is the official
        # exchange settlement price, fixed once published — same fix already
        # applied to LCC/RC option ingest (see Options/Code/lcc_ingest_lseg.py
        # / lrc_ingest_lseg.py, which also found SETTLE more reliable than
        # TRDPRC_1 on these continuation RICs).
        df = ld.get_history(universe=[symbol], fields=["OPEN_PRC", "HIGH_1", "LOW_1", "SETTLE"],
                             start=start, end=end, interval="daily", count=10000)
        if df is None or df.empty:
            return pd.DataFrame(columns=["Open", "High", "Low", "settlement"])
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df.rename(columns={"OPEN_PRC": "Open", "HIGH_1": "High", "LOW_1": "Low", "SETTLE": "settlement"})
        df.index = pd.to_datetime(df.index).normalize()
        df.index.name = "Date"
        df = df[~df.index.duplicated(keep="last")].sort_index()
        for col in ["Open", "High", "Low", "settlement"]:
            if col not in df.columns:
                df[col] = np.nan
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[df["settlement"].notna() & (df["settlement"] > 0)]

        if bday is not None and not df.empty:
            full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq=bday)
            gaps = full_idx.difference(df.index)
            df = df.reindex(full_idx)
            for col in ["Open", "High", "Low", "settlement"]:
                df[col] = df[col].interpolate(method="linear", limit_area="inside")
            df.index.name = "Date"
            if len(gaps):
                print(f"    interpolated {len(gaps)} missing business day(s) for {symbol}")
            df = df[df["settlement"].notna() & (df["settlement"] > 0)]

        return df
    except Exception as e:
        print(f"  ERROR fetching {symbol}: {e}")
        return pd.DataFrame(columns=["Open", "High", "Low", "settlement"])

# ── EXPLICIT CONTRACT CHAIN (skips excluded months, e.g. CT's October) ──────

def resolve_explicit_contract(ric, fetch_start, end_date, ld, bday):
    """
    Try the bare RIC then ^1/^2/^3 in turn — LSEG disambiguates repeat
    occurrences of the same month-code+year-digit pair across decades with a
    "^N" suffix that has to be discovered against the live fetch, not derived
    by formula (same finding as futures_builder_lseg.py's resolve_and_fetch,
    confirmed again here: e.g. CTZ5 = Dec 2025 only resolves as "CTZ5^2").

    Uses SETTLE. An earlier version of this function used TRDPRC_1 after a
    test of SETTLE came back empty — that test used the WRONG ^N candidate
    (^1 instead of the correct ^2), so the empty result was a resolution
    miss, not a real field gap. Retested against the correct candidate:
    SETTLE is consistently MORE complete than TRDPRC_1 on every contract
    checked (e.g. CTH7: 617 rows on SETTLE vs 242 on TRDPRC_1, full
    coverage through expiry vs cutting off ~2 weeks early). TRDPRC_1 had
    been silently dropping the final ~1-2 weeks of trading before several
    contracts' expiries, which meant those days were missing from the
    stitched c1/c2 series entirely (no source data for either leg on
    those dates) rather than merely imprecise.
    """
    for cand in (ric, f"{ric}^1", f"{ric}^2", f"{ric}^3"):
        try:
            raw = ld.get_history(universe=[cand],
                                  fields=["OPEN_PRC", "HIGH_1", "LOW_1", "SETTLE"],
                                  start=fetch_start, end=end_date, interval="daily", count=10000)
        except Exception:
            continue
        if raw is None or raw.empty:
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0] for c in raw.columns]
        raw = raw.rename(columns={"OPEN_PRC": "Open", "HIGH_1": "High",
                                   "LOW_1": "Low", "SETTLE": "settlement"})
        raw.index = pd.to_datetime(raw.index).normalize()
        raw = raw[~raw.index.duplicated(keep="last")].sort_index()
        for col in ["Open", "High", "Low", "settlement"]:
            if col not in raw.columns:
                raw[col] = np.nan
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
        raw = raw[raw["settlement"].notna() & (raw["settlement"] > 0)]
        if raw.empty:
            continue
        full_idx = pd.date_range(start=raw.index.min(), end=raw.index.max(), freq=bday)
        raw = raw.reindex(full_idx)
        for col in ["Open", "High", "Low", "settlement"]:
            raw[col] = raw[col].interpolate(method="linear", limit_area="inside")
        raw.index.name = "Date"
        raw = raw[raw["settlement"].notna() & (raw["settlement"] > 0)]
        if not raw.empty:
            return cand, raw
    return None, None


def fetch_explicit_chain(engine_key, root_ric, fetch_start, end_date, ld, bday):
    """
    Build true c1/c2 OHLC frames by stitching together each individual listed
    contract's OWN price history (never a generic continuation RIC), so any
    month excluded from COMMODITY_CONFIG[engine_key].months (e.g. CT's 'V')
    never appears as front or second month.

    For each business day, c1 = the not-yet-expired contract with the
    nearest LTD, c2 = the next one — same definition load_expiry_dates/
    generate_contract_table already use for regime windows, just applied
    per-contract instead of trusting the vendor's own c1/c2 roll.
    """
    end_year = pd.Timestamp.today().year + 1
    ct = generate_contract_table(engine_key, START_YEAR, end_year)
    ct["LTD"] = pd.to_datetime(ct["LTD"]).dt.normalize()
    ct = ct.sort_values("LTD").reset_index(drop=True)

    fetch_start_ts = pd.Timestamp(fetch_start)
    end_ts         = pd.Timestamp(end_date)
    # A contract can only ever be c1/c2 on day d if its LTD >= d — so nothing
    # with LTD < fetch_start_ts is ever usable and doesn't need fetching. This
    # is what makes incremental runs cheap for free: fetch_start is recent
    # (last ~5 days) in incremental mode, so this alone collapses the chain
    # down to just the currently-live contract(s), no separate "which
    # contracts changed" tracking needed. Pad the upper bound one contract's
    # worth so the final pre-expiry window still has a c2 to look ahead to.
    relevant = ct[(ct["LTD"] >= fetch_start_ts) &
                  (ct["LTD"] <= end_ts + pd.DateOffset(years=1))]

    contract_frames = {}
    for _, row in relevant.iterrows():
        base_ric = f"{root_ric}{row['month']}{row['year'] % 10}"
        # Bound the query tightly around THIS contract's own LTD (not the
        # whole fetch_start..end_date range) — otherwise a repeat decade
        # digit (e.g. "H9" = both 2009 and 2019) lets the ^N candidate
        # search silently match the wrong decade's contract, since its data
        # also happens to fall inside an overly wide window. Same bound as
        # futures_builder_lseg.py's resolve_and_fetch (~1400 days back).
        contract_start = max(fetch_start_ts, row["LTD"] - pd.Timedelta(days=1400))
        contract_end   = min(end_ts, row["LTD"] + pd.Timedelta(days=5))
        if contract_start > contract_end:
            continue
        resolved, df = resolve_explicit_contract(
            base_ric, contract_start.strftime("%Y-%m-%d"), contract_end.strftime("%Y-%m-%d"), ld, bday)
        if resolved:
            contract_frames[row["LTD"]] = df
            print(f"    {base_ric} -> {resolved}: {len(df)} rows "
                  f"({df.index.min().date()} -> {df.index.max().date()})")
        else:
            print(f"    (no data for {base_ric}, LTD {row['LTD'].date()})")

    if not contract_frames:
        return pd.DataFrame(columns=["Open","High","Low","settlement"]), \
               pd.DataFrame(columns=["Open","High","Low","settlement"])

    idx = pd.date_range(start=fetch_start, end=end_date, freq=bday)
    c1_recs, c2_recs = [], []
    sorted_ltds = sorted(contract_frames.keys())
    for d in idx:
        active_ltds = [ltd for ltd in sorted_ltds if ltd >= d]
        if len(active_ltds) < 2:
            continue
        f1, f2 = contract_frames[active_ltds[0]], contract_frames[active_ltds[1]]
        if d in f1.index and d in f2.index:
            c1_recs.append((d, *f1.loc[d, ["Open","High","Low","settlement"]]))
            c2_recs.append((d, *f2.loc[d, ["Open","High","Low","settlement"]]))

    cols = ["Date","Open","High","Low","settlement"]
    c1_df = pd.DataFrame(c1_recs, columns=cols).set_index("Date")
    c2_df = pd.DataFrame(c2_recs, columns=cols).set_index("Date")
    return c1_df, c2_df


# Commodities whose c1/c2 must be built from explicit per-contract RICs
# instead of the vendor's generic continuation series (see COMMODITY_CONFIG
# comment on CT for why).
EXPLICIT_CHAIN_COMMODITIES = {"CT"}


# ── EXPIRY DATES (unchanged) ────────────────────────────────────────────────

def get_contract_windows(engine_key):
    """Regime B windows and switch dates. Switch = LTD (same convention as the
    ICE source — this is our own computed roll, independent of whatever roll
    convention the vendor's own c1/c2 continuation series uses internally)."""
    end_year = pd.Timestamp.today().year + 1
    ct   = generate_contract_table(engine_key, START_YEAR, end_year)
    ct["LTD"] = pd.to_datetime(ct["LTD"]).dt.normalize()
    ct["FND"] = pd.to_datetime(ct["FND"]).dt.normalize()
    cfg  = COMMODITY_CONFIG[engine_key]
    bday = BDAY_CAL[cfg.calendar]

    switch_dates, windows = [], []
    for _, row in ct.iterrows():
        ltd = row["LTD"]
        windows.append(((ltd - (OFFSET - 1) * bday).normalize(), ltd))
        switch_dates.append(ltd)

    switch_idx = pd.DatetimeIndex(sorted(set(switch_dates)))
    return windows, switch_idx

# ── ROLL LOGIC (unchanged) ──────────────────────────────────────────────────

MONTH_NAMES = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

def build_tags(dates, regime_a, engine_key):
    end_year = pd.Timestamp.today().year + 1
    ct  = generate_contract_table(engine_key, START_YEAR, end_year)
    ct["LTD"] = pd.to_datetime(ct["LTD"]).dt.normalize()
    ct["FND"] = pd.to_datetime(ct["FND"]).dt.normalize()
    ct  = ct.sort_values("LTD").reset_index(drop=True)
    month_num = COMMODITY_CONFIG[engine_key].month_num

    switch_dates = [row["LTD"] for _, row in ct.iterrows()]
    switch_arr = np.array([np.datetime64(s) for s in switch_dates])

    rows = []
    for d, is_A in zip(dates, regime_a):
        d_np = np.datetime64(pd.Timestamp(d).normalize())
        c1_idx = int(np.searchsorted(switch_arr, d_np, side="right"))
        target = c1_idx if is_A else c1_idx + 1
        if target >= len(ct):
            target = len(ct) - 1
        contract = ct.iloc[target]
        delivery_month = month_num[contract["month"]]
        label = f"{MONTH_NAMES[delivery_month]}'{str(contract['year'])[2:]}"
        rows.append({
            "active_label": label,
            "active_fnd":   contract["FND"],
            "active_ltd":   contract["LTD"],
        })
    return pd.DataFrame(rows, index=pd.DatetimeIndex(dates))


def build_rollex(comm, c1_df, c2_df, expiry_dates, regime_windows):
    df = pd.concat([
        c1_df["settlement"].rename("c1"),
        c2_df["settlement"].rename("c2"),
        c1_df["Open"].rename("c1_open"),
        c1_df["High"].rename("c1_high"),
        c1_df["Low"].rename("c1_low"),
        c2_df["Open"].rename("c2_open"),
        c2_df["High"].rename("c2_high"),
        c2_df["Low"].rename("c2_low"),
    ], axis=1).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.loc[START_DATE:].dropna(subset=["c1","c2"])
    df = df[(df["c1"] > 0) & (df["c2"] > 0)]

    if df.empty:
        print(f"  [{comm}] WARNING: no data after cleaning — skipped")
        return pd.DataFrame()

    dates   = df.index.tolist()
    exp_set = set(expiry_dates)
    n       = len(dates)

    switch_flag, regime_A, regime_B = [], [], []
    for i, d in enumerate(dates):
        d_ts   = pd.Timestamp(d)
        is_exp = d in exp_set
        if not is_exp and i > 0:
            prev   = dates[i - 1]
            is_exp = any(prev < e <= d for e in exp_set)
        switch_flag.append(1 if is_exp else 0)

        in_regime_b = any(start <= d_ts <= end for start, end in regime_windows)
        regime_A.append(0 if in_regime_b else 1)
        regime_B.append(1 if in_regime_b else 0)

    df["switch"] = switch_flag
    df["A"]      = regime_A
    df["B"]      = regime_B
    df["c1_ret"] = df["c1"].pct_change()
    df["c2_ret"] = df["c2"].pct_change()

    rollex_ret = [np.nan]
    for i in range(1, n):
        if switch_flag[i - 1] == 1:
            ret = df["c1"].iat[i] / df["c2"].iat[i - 1] - 1
        elif regime_A[i] == 1:
            ret = df["c1_ret"].iat[i]
        else:
            ret = df["c2_ret"].iat[i]
        rollex_ret.append(ret)

    df["rollex_ret"] = rollex_ret

    anchor = df["c1"].iat[-1] if regime_A[-1] == 1 else df["c2"].iat[-1]
    px = [1.0]
    for i in range(1, n):
        r = df["rollex_ret"].iat[i]
        px.append(px[-1] * (1 + r) if pd.notna(r) else px[-1])
    scale           = anchor / px[-1]
    df["rollex_px"] = [p * scale for p in px]

    regime_a = np.array(regime_A, dtype=bool)

    c1_h_c = df["c1_high"] / df["c1"]
    c1_l_c = df["c1_low"]  / df["c1"]
    c1_o_c = df["c1_open"] / df["c1"]

    c2_h_c = df["c2_high"] / df["c2"]
    c2_l_c = df["c2_low"]  / df["c2"]
    c2_o_c = df["c2_open"] / df["c2"]

    h_c = np.where(regime_a, c1_h_c, c2_h_c)
    l_c = np.where(regime_a, c1_l_c, c2_l_c)
    o_c = np.where(regime_a, c1_o_c, c2_o_c)

    df["rollex_high"] = df["rollex_px"] * h_c
    df["rollex_low"]  = df["rollex_px"] * l_c
    df["rollex_open"] = df["rollex_px"] * o_c

    return df

# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--commodity", type=str, default=None, help="Run single commodity e.g. KC")
    args        = parser.parse_args()
    INCREMENTAL = not args.full

    if args.commodity:
        COMMODITIES = {k: v for k, v in COMMODITIES.items() if k == args.commodity.upper()}

    import lseg.data as ld
    ld.open_session()

    END_DATE = pd.Timestamp.today().strftime("%Y-%m-%d")
    results  = {}

    try:
        for comm, cfg in COMMODITIES.items():
            print(f"\n{'='*55}")
            print(f"  {comm}  ({cfg['c1']} / {cfg['c2']})")
            print(f"{'='*55}")

            regime_windows, expiries = get_contract_windows(cfg["engine_key"])
            print(f"  Expiries: {len(expiries)}  ({expiries.min().date()} -> {expiries.max().date()})")

            out_path = DB_DIR / f"rollex_{comm}.parquet"
            if INCREMENTAL and out_path.exists():
                existing    = pd.read_parquet(out_path)
                latest      = existing.index.max()
                fetch_start = (latest - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
                print(f"  Mode: INCREMENTAL from {fetch_start}")
            else:
                existing    = None
                fetch_start = START_DATE
                print(f"  Mode: FULL from {fetch_start}")

            bday = BDAY_CAL[COMMODITY_CONFIG[cfg["engine_key"]].calendar]

            if comm in EXPLICIT_CHAIN_COMMODITIES:
                # Build from real per-contract RICs, skipping excluded
                # months (e.g. CT's October) — see fetch_explicit_chain().
                root_ric = cfg["c1"].replace("c1", "")
                c1_new, c2_new = fetch_explicit_chain(
                    cfg["engine_key"], root_ric, fetch_start, END_DATE, ld, bday)
                c1_df, c2_df = c1_new, c2_new
                for label, new_df in [("c1", c1_new), ("c2", c2_new)]:
                    ohlc_cols = [f"{label}_open", f"{label}_high", f"{label}_low"]
                    if existing is not None and all(c in existing.columns for c in ohlc_cols):
                        hist = existing[[label] + ohlc_cols].copy()
                        hist.columns = ["settlement", "Open", "High", "Low"]
                        hist = hist[hist["settlement"] > 0]
                        merged = pd.concat([hist, new_df])
                        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
                        if label == "c1":
                            c1_df = merged
                        else:
                            c2_df = merged
                final_rows = {"c1": len(c1_df), "c2": len(c2_df)}
                print(f"  {root_ric} explicit chain (V excluded): "
                      f"c1={final_rows['c1']} rows, c2={final_rows['c2']} rows")
                if c1_df is None or c2_df is None or c1_df.empty or c2_df.empty:
                    print(f"  Skipping {comm} — incomplete explicit-chain price data")
                    continue
                df_out = build_rollex(comm, c1_df, c2_df, expiries, regime_windows)
                if df_out.empty:
                    continue
                print(f"  Tagging active contracts...")
                tags   = build_tags(df_out.index.tolist(), df_out["A"].tolist(), cfg["engine_key"])
                df_out = pd.concat([df_out, tags], axis=1)
                df_out.index.name = "Date"
                df_out.to_parquet(out_path)
                results[comm] = df_out
                print(f"  Saved -> {out_path.name}  |  {len(df_out)} rows  |  "
                      f"Rollex Px: {df_out['rollex_px'].iat[-1]:.4f}  |  "
                      f"Active: {df_out['active_label'].iat[-1]}")
                continue

            c1_df = c2_df = None
            for label, sym in [("c1", cfg["c1"]), ("c2", cfg["c2"])]:
                ohlc_cols = [f"{label}_open", f"{label}_high", f"{label}_low"]
                try:
                    new_df = fetch_ohlc(sym, fetch_start, END_DATE, ld, bday=bday)
                    if existing is not None:
                        if all(c in existing.columns for c in ohlc_cols):
                            hist = existing[[label] + ohlc_cols].copy()
                            hist.columns = ["settlement", "Open", "High", "Low"]
                        else:
                            hist = existing[[label]].rename(columns={label: "settlement"})
                            for col in ["Open", "High", "Low"]:
                                hist[col] = np.nan
                        hist = hist[hist["settlement"] > 0]
                        new_df = pd.concat([hist, new_df])
                        new_df = new_df[~new_df.index.duplicated(keep="last")].sort_index()
                    print(f"  {sym}: {len(new_df)} rows  ({new_df.index.min().date()} -> {new_df.index.max().date()})")
                    if label == "c1":
                        c1_df = new_df
                    else:
                        c2_df = new_df
                except Exception as e:
                    print(f"  ERROR fetching {sym}: {e}")
                    if existing is not None:
                        if all(c in existing.columns for c in ohlc_cols):
                            hist = existing[[label] + ohlc_cols].copy()
                            hist.columns = ["settlement", "Open", "High", "Low"]
                        else:
                            hist = existing[[label]].rename(columns={label: "settlement"})
                            for col in ["Open", "High", "Low"]:
                                hist[col] = np.nan
                        hist = hist[hist["settlement"] > 0]
                        if not hist.empty:
                            print(f"  Falling back to existing {label} ({len(hist)} rows)")
                            if label == "c1":
                                c1_df = hist
                            else:
                                c2_df = hist

            if c1_df is None or c2_df is None or c1_df.empty or c2_df.empty:
                print(f"  Skipping {comm} — incomplete price data")
                continue

            df_out = build_rollex(comm, c1_df, c2_df, expiries, regime_windows)
            if df_out.empty:
                continue

            print(f"  Tagging active contracts...")
            tags   = build_tags(df_out.index.tolist(), df_out["A"].tolist(), cfg["engine_key"])
            df_out = pd.concat([df_out, tags], axis=1)

            df_out.index.name = "Date"
            df_out.to_parquet(out_path)
            results[comm] = df_out
            print(f"  Saved -> {out_path.name}  |  {len(df_out)} rows  |  "
                  f"Rollex Px: {df_out['rollex_px'].iat[-1]:.4f}  |  "
                  f"Active: {df_out['active_label'].iat[-1]}")
    finally:
        ld.close_session()

    print(f"\n{'='*55}")
    print(f"  DONE — {len(results)}/{len(COMMODITIES)} commodities built")
    print(f"{'='*55}")
    for comm, df in results.items():
        print(f"  {comm:5s}  rows={len(df):5d}  "
              f"from={df.index.min().date()}  to={df.index.max().date()}  "
              f"rollex_px={df['rollex_px'].iat[-1]:.2f}  "
              f"active={df['active_label'].iat[-1]}")
