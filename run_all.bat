@echo off
REM run_all.bat -- the HAN-side experiment on the REBUILT AMAZON data (HAN review part H6).
REM
REM Every run is one model trained on one meta-path set with one seed. Its result goes to
REM output\AMAZON\v3\<model>_<key>_s<seed>.pt and its log to output\AMAZON\v3\logs\<same>.txt.
REM A run whose result already exists is skipped, so the script can be stopped (or the PC
REM switched off) and started again: only unfinished runs are redone. Any single stage can
REM be run on its own, and "dry" prints every command for running some of them by hand.
REM
REM   run_all.bat <stage>
REM
REM Before the first run:
REM   1. copy the rebuilt data\AMAZON from the LMSPS repo (node.dat, link.dat, label.dat,
REM      label.dat.test, meta.json) and delete the old graph folders under data\AMAZON
REM   2. point SEARCH_RECORD below at any one Fleet v3 search record
REM   3. once the fleets are done, paste the Fleet v3 arch.py entries (logs\Fleet v3\
REM      final_ns4|6\reselect_d010.txt in the LMSPS repo: amazon_lp_v3_f4_a1, ...) into
REM      arch.py. Until then build, han and m2v run the manual and random sets only, and
REM      seeds waits
REM
REM Stages ("all" runs them in this order; about 20 GPU hours in total, see
REM review\h6_run_protocol.md):
REM   check       check_handover.py: data\AMAZON must be the data the search ran on
REM   build       the meta-path graphs of every set (graphs from other data are rebuilt)
REM   audit       check_graph_bias.py and check_edition_twins.py
REM   smoke       3 epochs of HAN on a 6-path set and 2 of GAT: GPU memory check, nothing
REM               kept
REM   popularity  the popularity floor (seconds)
REM   han         HAN, seed 1, on every set: LMSPS Fleet v3, manual M1-M8, random draws
REM   gat         GAT on the union graph, seed 1
REM   m2v         metapath2vec, seed 1, on every set
REM   seeds       seeds 2 and 3 for the main comparison: HAN and metapath2vec on the
REM               headline sets, and GAT
REM   summarize   the results table (output\AMAZON\v3\results.csv), with full ranking
REM   push        commit the logs and the table (the .pt results stay local) and push
REM   all         every stage above, in that order
REM   dry         print every command of every stage, run nothing
REM Not run any more: --paper-config (HAN's attention collapses onto one path without the
REM warm-up, review part H3) and --paper-walks (about 60 GPU hours for a fidelity row).

setlocal EnableDelayedExpansion
cd /d "%~dp0"
REM inside a called :label, %0 is the label, so keep the file name for messages
set SELF=%~nx0

set STAGE=%~1
if "%STAGE%"=="" goto usage
set DRY=0
if /i "%STAGE%"=="dry" set DRY=1

REM ---- settings you may edit ---------------------------------------------------------
if not defined CONDA_ROOT set CONDA_ROOT=E:\miniconda3
set SEARCH_RECORD=..\LMSPS\hgb\logs\Fleet v3\final_ns4\search_ns4_seed1.json
set SPLIT=1
set GDIR=data\AMAZON\metapath_graphs_v3
set OUT=output\AMAZON\v3
set LOGDIR=%OUT%\logs
REM the main comparison: the simplest LMSPS set of each size against the manual set of that
REM size (frozen before any result: review\manual_metapath_sets in the LMSPS repo)
set HEADLINE=amazon_lp_v3_f4_a1 amazon_lp_v3_f6_a1 amazon_lp_manual_5 amazon_lp_manual_6
set MORE_SEEDS=2 3
set SMOKE_KEY=amazon_lp_manual_6

REM ---- environment ---------------------------------------------------------------------
REM in a plain cmd `python` is the Windows-Store stub, so activate lmsps_env (another
REM machine: set CONDA_ROOT to its conda folder before calling this script)
if /i not "%CONDA_DEFAULT_ENV%"=="lmsps_env" (
    call "%CONDA_ROOT%\Scripts\activate.bat"
    call conda activate lmsps_env
)
if not defined PY set PY=python
%PY% -c "import sys; print('python:', sys.executable)"
if not exist "%LOGDIR%" mkdir "%LOGDIR%"

REM ---- the meta-path sets (arch.py keys) and the model settings --------------------------
set MANUAL=amazon_lp_manual_1 amazon_lp_manual_2 amazon_lp_manual_3 amazon_lp_manual_4 amazon_lp_manual_5 amazon_lp_manual_6 amazon_lp_manual_7 amazon_lp_manual_8
set RANDOM_SETS=amazon_lp_rand4_d1 amazon_lp_rand4_d2 amazon_lp_rand4_d3 amazon_lp_rand4_d4 amazon_lp_rand4_d5 amazon_lp_rand6_d1 amazon_lp_rand6_d2 amazon_lp_rand6_d3 amazon_lp_rand6_d4 amazon_lp_rand6_d5
REM the Fleet v3 sets: every amazon_lp_v3_f4_a<i> and amazon_lp_v3_f6_a<i> in arch.py
set LMSPS=
%PY% arch_keys.py amazon_lp_v3_f4_a amazon_lp_v3_f6_a > "%TEMP%\han_v3_keys.txt" 2> "%TEMP%\han_v3_keys.err"
if not errorlevel 1 set /p LMSPS=<"%TEMP%\han_v3_keys.txt"

set COMMON=--dataset AMAZON --split-seed %SPLIT%
set HAN_REC=%COMMON% --graph-dir %GDIR% --stage 150 --patience 10 --beta-warmup 15
set M2V_FAST=%COMMON% --epochs 10 --evals-per-epoch 4 --patience 6
set GAT_REC=%COMMON% --stage 150 --patience 10

REM ---- dispatch ------------------------------------------------------------------------
if /i "%STAGE%"=="all" goto all
if /i "%STAGE%"=="dry" goto all
for %%S in (check build audit smoke popularity han gat m2v seeds summarize push) do (
    if /i "%STAGE%"=="%%S" (
        call :%%S
        goto done
    )
)
:usage
echo usage: %~nx0 check ^| build ^| audit ^| smoke ^| popularity ^| han ^| gat ^| m2v ^| seeds ^| summarize ^| push ^| all ^| dry
exit /b 1

:all
call :check || goto stopped
call :build || goto stopped
call :audit
call :smoke || goto stopped
call :popularity
call :han
call :gat
call :m2v
call :seeds
call :summarize
call :push
goto done

:stopped
echo.
echo Stopped: fix the problem above, then run "%~nx0 %STAGE%" again (finished runs are skipped).
exit /b 1

:done
echo.
echo Stage "%STAGE%" finished.
if "%DRY%"=="0" pause
exit /b 0

REM =============================== stages ===============================================
:check
echo === check: is data\AMAZON the data the LMSPS search ran on?
if "%DRY%"=="1" (
    echo   [dry] %PY% check_handover.py "%SEARCH_RECORD%"
    exit /b 0
)
%PY% check_handover.py "%SEARCH_RECORD%"
if errorlevel 1 (
    echo   copy the rebuilt data and set SEARCH_RECORD at the top of %SELF%
    exit /b 1
)
exit /b 0

:build
echo === build: graphs of every set into %GDIR%
if not defined LMSPS call :no_lmsps_yet
for %%A in (%LMSPS% %MANUAL% %RANDOM_SETS%) do (
    call :exec "build_%%A" %PY% build_metapath_graphs.py --dataset AMAZON --arch %%A --topk 20 --split-seed %SPLIT% --graph-dir %GDIR% --skip-existing || exit /b 1
)
exit /b 0

:audit
echo === audit
call :exec "audit_graph_bias" %PY% check_graph_bias.py %GDIR%
call :exec "audit_edition_twins" %PY% check_edition_twins.py --split-seed %SPLIT%
exit /b 0

:smoke
echo === smoke: a few epochs on the biggest runs, to see they fit in GPU memory
call :exec "smoke_han" %PY% train_han.py %HAN_REC% --arch %SMOKE_KEY% --seed 1 --stage 3 --beta-warmup 1 --save-emb %OUT%/smoke_han.pt || exit /b 1
call :exec "smoke_gat" %PY% train_gat.py %GAT_REC% --seed 1 --stage 2 --save-emb %OUT%/smoke_gat.pt || exit /b 1
if exist "%OUT%\smoke_*.pt" del "%OUT%\smoke_*.pt"
exit /b 0

:popularity
echo === popularity
call :train popularity_s1 %PY% popularity_baseline.py %COMMON% --seed 1
exit /b 0

:han
echo === han: seed 1 on every set
if not defined LMSPS call :no_lmsps_yet
for %%A in (%LMSPS% %MANUAL% %RANDOM_SETS%) do (
    call :train han_%%A_s1 %PY% train_han.py %HAN_REC% --arch %%A --seed 1
)
exit /b 0

:gat
echo === gat: seed 1
call :train gat_union_s1 %PY% train_gat.py %GAT_REC% --seed 1
exit /b 0

:m2v
echo === m2v: seed 1 on every set
if not defined LMSPS call :no_lmsps_yet
for %%A in (%LMSPS% %MANUAL% %RANDOM_SETS%) do (
    call :train m2v_%%A_s1 %PY% train_metapath2vec.py %M2V_FAST% --arch %%A --seed 1
)
exit /b 0

:seeds
echo === seeds %MORE_SEEDS%: HAN and metapath2vec on %HEADLINE%, and GAT
call :need_lmsps || exit /b 1
for %%S in (%MORE_SEEDS%) do (
    for %%A in (%HEADLINE%) do (
        call :train han_%%A_s%%S %PY% train_han.py %HAN_REC% --arch %%A --seed %%S
        call :train m2v_%%A_s%%S %PY% train_metapath2vec.py %M2V_FAST% --arch %%A --seed %%S
    )
    call :train gat_union_s%%S %PY% train_gat.py %GAT_REC% --seed %%S
)
exit /b 0

:summarize
echo === summarize
call :exec "summarize" %PY% summarize_results.py %OUT%\*.pt --group --full-ranking --csv %OUT%\results.csv
if "%DRY%"=="0" type "%LOGDIR%\summarize.txt"
exit /b 0

:push
echo === push: logs and results table
if "%DRY%"=="1" (
    echo   [dry] git add "%OUT%" ^& git commit ^& git push
    exit /b 0
)
git add "%OUT%"
git commit -q -m "HAN v3 logs and results table (run_all.bat)" || echo   nothing new to commit
git push || echo   push failed: check that git is on PATH and the branch tracks a remote
exit /b 0

REM ============================== helpers ===============================================
REM :no_lmsps_yet -- the Fleet v3 keys are not in arch.py yet: say so, train the others
:no_lmsps_yet
type "%TEMP%\han_v3_keys.err"
echo   the Fleet v3 sets are not in arch.py yet: only the manual and random sets now; run this
echo   stage again once they are pasted ^(finished runs are skipped^)
exit /b 0

REM :need_lmsps -- the Fleet v3 keys must be in arch.py (dry mode: placeholders)
:need_lmsps
if defined LMSPS exit /b 0
type "%TEMP%\han_v3_keys.err"
if "%DRY%"=="1" (
    set LMSPS=amazon_lp_v3_f4_a1 amazon_lp_v3_f6_a1
    echo   [dry] placeholder keys until the Fleet v3 entries are in arch.py
    exit /b 0
)
echo   paste the Fleet v3 entries ^(amazon_lp_v3_f4_a1, ...^) into arch.py first
exit /b 1

REM :train <name> <command...> -- result %OUT%\<name>.pt, log %LOGDIR%\<name>.txt; skipped if
REM the result exists
:train
set NAME=%~1
shift
if exist "%OUT%\%NAME%.pt" (
    echo   skip %NAME% ^(result exists^)
    exit /b 0
)
set CMD=
:train_collect
if "%~1"=="" goto train_run
set CMD=%CMD% %~1
shift
goto train_collect
:train_run
set CMD=%CMD% --save-emb %OUT%/%NAME%.pt
call :exec "%NAME%" %CMD%
exit /b %ERRORLEVEL%

REM :exec <logname> <command...> -- runs (or prints, in dry mode) with stdout+stderr in
REM %LOGDIR%\<logname>.txt; returns the command's exit code
:exec
set LOGNAME=%~1
shift
set ECMD=
:exec_collect
if "%~1"=="" goto exec_run
set ECMD=%ECMD% %~1
shift
goto exec_collect
:exec_run
if "%DRY%"=="1" (
    echo   [dry] %ECMD%   ^> "%LOGDIR%\%LOGNAME%.txt"
    exit /b 0
)
for /f %%T in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd_HH:mm:ss"') do set TS=%%T
echo   [%TS%] %LOGNAME%
> "%LOGDIR%\%LOGNAME%.txt" echo Command:%ECMD%
>> "%LOGDIR%\%LOGNAME%.txt" echo Start Time: %TS%
>> "%LOGDIR%\%LOGNAME%.txt" echo ------------------------------------------------------------
%ECMD% >> "%LOGDIR%\%LOGNAME%.txt" 2>&1
set RC=%ERRORLEVEL%
for /f %%T in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd_HH:mm:ss"') do set TS=%%T
>> "%LOGDIR%\%LOGNAME%.txt" echo ------------------------------------------------------------
>> "%LOGDIR%\%LOGNAME%.txt" echo End Time: %TS%  exit code %RC%
if not "%RC%"=="0" echo   FAILED ^(exit code %RC%^): see %LOGDIR%\%LOGNAME%.txt
exit /b %RC%
