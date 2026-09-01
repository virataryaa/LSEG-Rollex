"""
notify.py — Rollex (LSEG) Automator email summary
Usage: python notify.py <status> <git_status>
  status     : ok | error
  git_status : pushed | skipped | failed
"""

import sys
import datetime
import pandas as pd
from pathlib import Path

TO_EMAIL = "virat.arya@etgworld.com"
DB_DIR   = Path(r"C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\LSEG\Rollex\Database")
COMMS    = ["KC", "RC", "CC", "SB", "CT", "LCC", "LSU"]

status     = sys.argv[1] if len(sys.argv) > 1 else "ok"
git_status = sys.argv[2] if len(sys.argv) > 2 else "unknown"
run_dt     = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
today      = datetime.date.today().strftime("%Y-%m-%d")


def parquet_summary() -> str:
    lines = []
    for comm in COMMS:
        path = DB_DIR / f"rollex_{comm}.parquet"
        if not path.exists():
            lines.append(f"  {comm:<6}  FILE NOT FOUND")
            continue
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        latest_px = df["rollex_px"].iloc[-1] if "rollex_px" in df.columns else float("nan")
        active    = df["active_label"].iloc[-1] if "active_label" in df.columns else "—"
        lines.append(
            f"  {comm:<6}  {len(df):>5} rows   "
            f"{df.index.min().date()} -> {df.index.max().date()}   "
            f"px={latest_px:>9.2f}   active={active}"
        )
    return "\n".join(lines)


def send_outlook_email(subject: str, body: str):
    try:
        import win32com.client
        outlook      = win32com.client.Dispatch("Outlook.Application")
        mail         = outlook.CreateItem(0)
        mail.To      = TO_EMAIL
        mail.Subject = subject
        mail.Body    = body
        mail.Send()
        print(f"  Email sent -> {TO_EMAIL}")
    except Exception as e:
        print(f"  Email failed: {e}")


ok  = status == "ok"
tag = "[OK]" if ok else "[ERROR]"
subject = f"{tag} LSEG-Rollex — {today}"

git_line = {
    "pushed":  "GitHub  : Pushed successfully",
    "skipped": "GitHub  : No changes — push skipped",
    "failed":  "GitHub  : PUSH FAILED",
}.get(git_status, f"GitHub  : {git_status}")

body = f"""LSEG Rollex — Daily Update
Run time : {run_dt}
Status   : {"OK" if ok else "ERROR — builder failed, check run_log.txt"}
{git_line}

{"=" * 60}
ROLLEX SUMMARY
{"=" * 60}
{parquet_summary()}
{"=" * 60}
Note: LSEG interim-migration pipeline. Roll-adjustment logic is ported
unchanged from the ICE source; c1/c2 come from LSEG continuation RICs
(KCc1/KCc2 etc). Occasional data gaps in the LSEG feed near a roll date can
cause small one-off return spikes on that day — this is a source data
completeness characteristic, not a builder bug.

Log: C:\\Users\\virat.arya\\ETG\\SoftsDatabase - Documents\\Database\\Hardmine\\LSEG\\Rollex\\Automator\\run_log.txt
"""

print(body)
send_outlook_email(subject, body)
