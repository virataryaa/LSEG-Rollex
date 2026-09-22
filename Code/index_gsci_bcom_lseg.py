"""
index_gsci_bcom_lseg.py — Headline S&P GSCI / Bloomberg Commodity Index
=========================================================================
Pulls the two broad commodity benchmark indices (not the single-commodity
GSCI sub-indices used elsewhere in the repo, e.g. LSEG/Index or LSEG/CTA —
these are the headline composite series) so the Rollex Correlation tab can
show how each soft sits against the broad commodity complex.

RICs (verified live against LSEG Workspace 2026-09-22):
    .SPGSCI  — S&P GSCI (headline, spot/excess-return level)
    .BCOM    — Bloomberg Commodity Index (headline level)

Output: Rollex/Database/index_gsci_bcom.parquet, indexed by Date, columns
    GSCI, GSCI_ret, BCOM, BCOM_ret

Usage:
    python index_gsci_bcom_lseg.py           # incremental update
    python index_gsci_bcom_lseg.py --full     # full rebuild from START_DATE
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("future.no_silent_downcasting", True)

START_DATE = "2005-01-01"
DB_DIR     = Path(__file__).resolve().parent.parent / "Database"
OUT_FILE   = DB_DIR / "index_gsci_bcom.parquet"

RICS = {"GSCI": ".SPGSCI", "BCOM": ".BCOM"}


def fetch(ld, start: str, end: str) -> pd.DataFrame:
    df = ld.get_history(universe=list(RICS.values()), fields=["TRDPRC_1"],
                        start=start, end=end, interval="daily")
    # get_history returns a single-field frame with the RICs as columns
    # (no field sub-level) when only one field is requested.
    df = df.rename(columns={ric: name for name, ric in RICS.items()})
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df[list(RICS.keys())].apply(pd.to_numeric, errors="coerce").sort_index()


def build(full: bool) -> pd.DataFrame:
    import lseg.data as ld
    ld.open_session()

    end = pd.Timestamp.today().strftime("%Y-%m-%d")

    if not full and OUT_FILE.exists():
        old = pd.read_parquet(OUT_FILE)
        last = old.index.max()
        start = (last - pd.Timedelta(days=10)).strftime("%Y-%m-%d")  # small overlap
        new = fetch(ld, start, end)
        combined = pd.concat([old[["GSCI", "BCOM"]], new])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = fetch(ld, START_DATE, end)

    combined = combined.ffill()  # holiday-mismatch gaps between the two indices
    combined["GSCI_ret"] = combined["GSCI"].pct_change()
    combined["BCOM_ret"] = combined["BCOM"].pct_change()
    return combined


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="full rebuild from START_DATE")
    args = ap.parse_args()

    combined = build(args.full)
    DB_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUT_FILE)
    print(f"Saved -> {OUT_FILE.name} | {len(combined)} rows | "
          f"{combined.index.min().date()} -> {combined.index.max().date()}")


if __name__ == "__main__":
    main()
