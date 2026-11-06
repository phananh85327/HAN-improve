@echo off
REM recommend_api.bat -- start the recommendation API (recommend_api.py) on this machine, with the
REM representative results of README "Example recommendations". Stop it with Ctrl+C.
REM
REM   recommend_api.bat [port]
REM
REM Then a front end on the same machine calls http://127.0.0.1:<port>/api/... (README
REM "Recommendation API" lists the endpoints); try http://127.0.0.1:8000/api/models in a browser.

setlocal
cd /d "%~dp0"

set PORT=%~1
if "%PORT%"=="" set PORT=8000

REM ---- settings you may edit -------------------------------------------------------------
if not defined CONDA_ROOT set CONDA_ROOT=E:\miniconda3
REM the folder holding filtered_users.csv and filtered_products.csv (the rebuilt data's input)
if not defined CSV set CSV=..\LMSPS\data\AMAZON_filtered

REM ---- environment (as in run_all.bat) ---------------------------------------------------
if /i not "%CONDA_DEFAULT_ENV%"=="lmsps_env" (
    call "%CONDA_ROOT%\Scripts\activate.bat"
    call conda activate lmsps_env
)
if not defined PY set PY=python
if not exist "%CSV%\filtered_users.csv" (
    echo filtered_users.csv is not in %CSV%: set CSV to the folder with the filtered CSVs
    exit /b 1
)

%PY% recommend_api.py --port %PORT% --users-csv "%CSV%\filtered_users.csv" --products-csv "%CSV%\filtered_products.csv"
