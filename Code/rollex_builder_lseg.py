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
    "CT": CommodityConfig(
        months=["H","K","N","V","Z"], month_num={"H":3,"K":5,"N":7,"V":10,"Z":12},
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
        df = ld.get_history(universe=[symbol], fields=["OPEN_PRC", "HIGH_1", "LOW_1", "TRDPRC_1"],
                             start=start, end=end, interval="daily", count=10000)
        if df is None or df.empty:
            return pd.DataFrame(columns=["Open", "High", "Low", "settlement"])
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df.rename(columns={"OPEN_PRC": "Open", "HIGH_1": "High", "LOW_1": "Low", "TRDPRC_1": "settlement"})
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

            c1_df = c2_df = None
            for label, sym in [("c1", cfg["c1"]), ("c2", cfg["c2"])]:
                ohlc_cols = [f"{label}_open", f"{label}_high", f"{label}_low"]
                try:
                    bday = BDAY_CAL[COMMODITY_CONFIG[cfg["engine_key"]].calendar]
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
