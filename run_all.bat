@echo off
REM run_all.bat -- the complete experiment flow for this repo (HAN, metapath2vec, GAT)
REM on AMAZON, split seed 1, graphs from data\AMAZON\metapath_graphs_all_unique.
REM
REM   run_all.bat <stage> [extra arch keys...]
REM
REM stages (in the order "all" runs them):
REM   build      materialize the manual + random ablation paths into the shared graph dir
REM   audit      check_graph_bias.py + check_edition_twins.py on the shared graph dir
REM   han        HAN config of record, seed 1, on the 13 shortlist keys + 3 ablation keys
REM   m2v        metapath2vec fast sweep, seed 1, same 16 keys
REM   gat        GAT (arch-independent), seeds 1 2 3
REM   seeds      HAN seeds 2 3 on the headline keys (+ any extra keys given as arguments,
REM              e.g. the best-by-val-BPR shortlist key per ns from summarize_results.py)
REM   m2v_paper  metapath2vec --paper-walks, seeds 1 2 3, headline keys (slowest stage)
REM   han_paper  HAN --paper-config, seed 1, headline keys
REM   summarize  summarize_results.py --group --csv "logs\Final 5\results.csv"
REM   all        every stage above, in that order
REM   dry        print every command of every stage without running anything
REM
REM Every run writes logs\Final 5\<stage>_<arch>_s<seed>.txt and a blob
REM output\AMAZON\<model>_<arch>_s<seed>[_paper].pt; a run whose blob already exists
REM is skipped, so the script can be re-launched after an interruption.

setlocal EnableDelayedExpansion
cd /d "%~dp0"

set STAGE=%~1
if "%STAGE%"=="" set STAGE=summarize
set DRY=0
if /i "%STAGE%"=="dry" set DRY=1

REM ---- environment (same as the old execute.bat) ------------------------------
if "%DRY%"=="0" if not defined CONDA_DEFAULT_ENV (
    call E:\miniconda3\Scripts\activate.bat
    call activate lmsps_env
)
if not defined PY set PY=python
if "%DRY%"=="0" (
    echo Using interpreter:
    %PY% -c "import sys; print(sys.executable)"
)

REM ---- protocol constants -------------------------------------------------------
set SPLIT=1
set GDIR=data\AMAZON\metapath_graphs_all_unique
set OUT=output\AMAZON
set LOGDIR=logs\Final 5
if not exist "%LOGDIR%" mkdir "%LOGDIR%"

set COMMON=--dataset AMAZON --split-seed %SPLIT%
set HAN_REC=%COMMON% --graph-dir %GDIR% --stage 150 --patience 10 --beta-warmup 15
set HAN_PAPER=%COMMON% --graph-dir %GDIR% --paper-config --stage 200
set M2V_FAST=%COMMON% --epochs 10 --evals-per-epoch 4 --patience 6
set M2V_PAPER=%COMMON% --paper-walks --epochs 3 --evals-per-epoch 4 --patience 6
set GAT_REC=%COMMON% --stage 150 --patience 10

set SHORTLIST=amazon_lp_f4_s1 amazon_lp_f4_s2 amazon_lp_f4_s3 amazon_lp_f4_s4 amazon_lp_f4_s5 amazon_lp_f4_s6 amazon_lp_f6_s1 amazon_lp_f6_s2 amazon_lp_f6_s3 amazon_lp_f6_s4 amazon_lp_f6_s5 amazon_lp_f6_s6 amazon_lp_f6_s7
set ABLATION=amazon_lp_manual amazon_lp_random4_s0 amazon_lp_random6_s0
set HEADLINE=amazon_lp_f4_c amazon_lp_f6_s1 amazon_lp_manual
REM amazon_lp_f4_c is an alias of amazon_lp_f4_s1 (same paths); its seed-1 run comes
REM from the sweep, so "seeds" only adds seeds 2 and 3 for it.

REM ---- dispatch ------------------------------------------------------------------
if /i "%STAGE%"=="dry"       ( call :build & call :audit & call :han & call :m2v & call :gat & call :seeds %2 %3 %4 %5 & call :m2v_paper & call :han_paper & call :summarize & goto :done )
if /i "%STAGE%"=="all"       ( call :build & call :audit & call :han & call :m2v & call :gat & call :seeds %2 %3 %4 %5 & call :m2v_paper & call :han_paper & call :summarize & goto :done )
if /i "%STAGE%"=="build"     ( call :build & goto :done )
if /i "%STAGE%"=="audit"     ( call :audit & goto :done )
if /i "%STAGE%"=="han"       ( call :han & goto :done )
if /i "%STAGE%"=="m2v"       ( call :m2v & goto :done )
if /i "%STAGE%"=="gat"       ( call :gat & goto :done )
if /i "%STAGE%"=="seeds"     ( call :seeds %2 %3 %4 %5 & goto :done )
if /i "%STAGE%"=="m2v_paper" ( call :m2v_paper & goto :done )
if /i "%STAGE%"=="han_paper" ( call :han_paper & goto :done )
if /i "%STAGE%"=="summarize" ( call :summarize & goto :done )
echo Unknown stage "%STAGE%". See the header of this file.
exit /b 1

:done
echo.
echo Stage "%STAGE%" finished.
if "%DRY%"=="0" pause
exit /b 0

REM =============================== stages =====================================
:build
echo === build: manual + random ablation paths into %GDIR%
for %%A in (%ABLATION%) do (
    call :exec "build_%%A" %PY% build_metapath_graphs.py --dataset AMAZON --arch %%A --topk 20 --split-seed %SPLIT% --graph-dir %GDIR% --skip-existing
)
goto :eof

:audit
echo === audit
call :exec "audit_graph_bias" %PY% check_graph_bias.py %GDIR%
call :exec "audit_edition_twins" %PY% check_edition_twins.py --split-seed %SPLIT% --products-csv output/AMAZON/Filtered_Output/filtered_products.csv
goto :eof

:han
echo === han: config of record, seed 1
for %%A in (%SHORTLIST% %ABLATION%) do (
    call :train han_%%A_s1 %PY% train_han.py %HAN_REC% --arch %%A --seed 1
)
goto :eof

:m2v
echo === m2v: fast sweep, seed 1
for %%A in (%SHORTLIST% %ABLATION%) do (
    call :train m2v_%%A_s1 %PY% train_metapath2vec.py %M2V_FAST% --arch %%A --seed 1
)
goto :eof

:gat
echo === gat: seeds 1 2 3
for %%S in (1 2 3) do (
    call :train gat_union_s%%S %PY% train_gat.py %GAT_REC% --seed %%S
)
goto :eof

:seeds
echo === seeds: HAN seeds 2 3 on headline keys %HEADLINE% %*
for %%A in (%HEADLINE% %*) do (
    for %%S in (2 3) do (
        call :train han_%%A_s%%S %PY% train_han.py %HAN_REC% --arch %%A --seed %%S
    )
)
goto :eof

:m2v_paper
echo === m2v_paper: --paper-walks, seeds 1 2 3, headline keys
for %%A in (%HEADLINE%) do (
    for %%S in (1 2 3) do (
        call :train m2v_%%A_s%%S_paper %PY% train_metapath2vec.py %M2V_PAPER% --arch %%A --seed %%S
    )
)
goto :eof

:han_paper
echo === han_paper: --paper-config, seed 1, headline keys
for %%A in (%HEADLINE%) do (
    call :train han_%%A_s1_paper %PY% train_han.py %HAN_PAPER% --arch %%A --seed 1
)
goto :eof

:summarize
echo === summarize
call :exec "summarize" %PY% summarize_results.py "%OUT%\*.pt" --group --csv "%LOGDIR%\results.csv"
if exist "%LOGDIR%\summarize.txt" type "%LOGDIR%\summarize.txt"
goto :eof

REM ============================== helpers =====================================
REM :train <name> <command...>  -> blob %OUT%\<name>.pt, log %LOGDIR%\<name>.txt, skipped if the blob exists
:train
set NAME=%~1
shift
if exist "%OUT%\%NAME%.pt" (
    echo   skip %NAME% ^(blob exists^)
    goto :eof
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
goto :eof

REM :exec <logname> <command...>  -> runs (or prints, in dry mode) with stdout+stderr in %LOGDIR%\<logname>.txt
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
    goto :eof
)
for /f %%T in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd_HH:mm:ss"') do set TS=%%T
echo   [%TS%] %LOGNAME%
> "%LOGDIR%\%LOGNAME%.txt" echo Command:%ECMD%
>> "%LOGDIR%\%LOGNAME%.txt" echo Start Time: %TS%
>> "%LOGDIR%\%LOGNAME%.txt" echo ------------------------------------------------------------
%ECMD% >> "%LOGDIR%\%LOGNAME%.txt" 2>&1
for /f %%T in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd_HH:mm:ss"') do set TS=%%T
>> "%LOGDIR%\%LOGNAME%.txt" echo ------------------------------------------------------------
>> "%LOGDIR%\%LOGNAME%.txt" echo End Time: %TS%  exit code %ERRORLEVEL%
goto :eof
