@echo off
setlocal EnableDelayedExpansion
set LOG="C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\LSEG\Rollex\Automator\run_log.txt"
set INGEST_STATUS=ok
set GIT_STATUS=skipped

:: Prevent Git Credential Manager from showing an interactive dialog in unattended runs.
:: If credentials are cached it pushes silently; if not, it fails immediately instead of hanging.
set GCM_INTERACTIVE=never
set GIT_TERMINAL_PROMPT=0
echo. >> %LOG%
echo ============================= >> %LOG%
echo Run started: %date% %time% >> %LOG%
echo ============================= >> %LOG%

:: Step 1 - Incremental Rollex build (upsert, LSEG)
echo [1] Running rollex_builder_lseg.py... >> %LOG%
python "C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\LSEG\Rollex\Code\rollex_builder_lseg.py" >> %LOG% 2>&1

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: rollex_builder_lseg.py failed >> %LOG%
    set INGEST_STATUS=error
    goto notify
)

:: Step 2 - Push updated parquets to GitHub
echo [2] Pushing to GitHub... >> %LOG%
cd /d "C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\LSEG\Rollex"
git add Database\rollex_KC.parquet Database\rollex_RC.parquet Database\rollex_CC.parquet Database\rollex_SB.parquet Database\rollex_CT.parquet Database\rollex_LCC.parquet Database\rollex_LSU.parquet >> %LOG% 2>&1
git diff --cached --quiet
if %ERRORLEVEL% NEQ 0 (
    git commit -m "Auto update: Rollex (LSEG) %date%" >> %LOG% 2>&1
    git push >> %LOG% 2>&1
    if !ERRORLEVEL! EQU 0 (
        set GIT_STATUS=pushed
        echo Git push done. >> %LOG%
    ) else (
        set GIT_STATUS=failed
        echo ERROR: git push failed >> %LOG%
    )
) else (
    echo No changes to commit. >> %LOG%
    set GIT_STATUS=skipped
)

:notify
echo [3] Sending email notification... >> %LOG%
python "C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\LSEG\Rollex\Automator\notify.py" %INGEST_STATUS% %GIT_STATUS% >> %LOG% 2>&1

echo Run finished: %date% %time% >> %LOG%
