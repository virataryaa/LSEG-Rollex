# Rollex — Interim Migration (LSEG)

Interim replacement for `ICEBREAKER/Rollex`, rebuilt against the **LSEG Data
API** (`lseg.data`) instead of ICE Connect (`icepython`), for the period while
ICE API access is unavailable. Produces continuous, roll-adjusted OHLC price
series for 7 ICE softs commodities (KC, RC, CC, LCC, SB, CT, LSU).

## What's here

- **`Code/rollex_builder_lseg.py`** — the builder. All contract-calendar /
  roll-regime / back-adjustment math (holiday calendars, FND/LTD contract
  tables, regime logic, return-splicing back-adjustment, OHLC reconstruction)
  is carried over **unchanged** from the ICE source — it's 100% vendor-
  agnostic pandas/numpy. Only the price fetch (`icepython` → `lseg.data`) and
  the c1/c2 RIC map were rewritten.
- **`Database/`** — `rollex_{KC,RC,CC,SB,CT,LCC,LSU}.parquet`, full history
  from 2010, also synced into the sibling `Interim_Migration/COT_ALL` project's
  `Database/Rollex/` for its Dashboard's roll-adjusted-price tabs.
- **`Dashboard/ICE_Rollex.py`** — copied verbatim from the ICE source (pure
  parquet consumer, no ICE dependency). Seasonality, Correlation, Indexed
  Performance, Price & Vol, and Return Distribution tabs.
- **`Automator/`** — `run.bat` (incremental rebuild + git push + email),
  `notify.py`.

## A data-completeness note, not a logic bug

LSEG's `KCc1`/`KCc2`-style continuation RICs have real gaps on days the
exchange was open — confirmed by spot-checking against the ICE feed (e.g.
`KCc1` is missing both 2025-09-17 and 2025-09-18 entirely). The builder
reindexes onto each commodity's own exchange business-day calendar and
linearly interpolates strictly-internal gaps (never extrapolating at the
edges) before computing returns, which meaningfully tightened the match to
ICE but doesn't fully close it — CC in particular still carries the most
residual noise of the seven. The roll/switch mechanism itself is verified
exact against the ICE source, including its "off-by-one active_label on the
switch day" quirk.

## Running it

```bash
python Code/rollex_builder_lseg.py --full          # full rebuild from 2010
python Code/rollex_builder_lseg.py                 # incremental update
python Code/rollex_builder_lseg.py --commodity KC   # single commodity
streamlit run Dashboard/ICE_Rollex.py
```

Requires an authenticated LSEG Workspace/Eikon session on the host running
the builder.
