@echo off
REM recommend_6path.bat -- recommend.py on the main representative result of each kind of
REM meta-path set (6 paths), plus GAT and popularity, with the same user and products.
REM
REM   recommend_6path.bat <customer id> [ASIN] [ASIN]
REM
REM Why these results (README "Example recommendations"; seed 1 for all, since every set has a
REM seed-1 run; the 6-path sets did better than the 4-path ones in every family, review H8):
REM   han_amazon_lp_v3_f6_a1_s1   automatic: the pre-registered main LMSPS set of 6 paths
REM   han_amazon_lp_manual_6_s1   manual: its main-comparison partner, M6
REM   han_amazon_lp_rand6_d3_s1   random: the median of the five 6-path draws by validation BPR
REM                               (a typical draw, not the luckiest)
REM   gat_union_s1                GAT, the best model (no meta-paths)
REM   popularity_s1               the floor; user query only (its product embedding is one number)
REM
REM customer id: the ID column of filtered_users.csv, or a user index 0-53568. The header of each
REM list says how many products the user reviewed; users with about 6-20 make the clearest lists.
REM ASIN: products for the "similar products" query, e.g. one book and one DVD.
REM
REM Output: output\AMAZON\v3_check\recommend_6path_<customer id>.txt, one section per result,
REM also shown on screen. --exclude-seen hides every product the user reviewed, held-out test
REM purchases included: the lists show what each model would recommend next, not whether it
REM found what the user bought.

setlocal
cd /d "%~dp0"

set USER_ID=%~1
if "%USER_ID%"=="" goto usage
set ASIN1=%~2
set ASIN2=%~3

REM ---- settings you may edit -------------------------------------------------------------
if not defined CONDA_ROOT set CONDA_ROOT=E:\miniconda3
REM the folder holding filtered_users.csv and filtered_products.csv (the rebuilt data's input)
if not defined CSV set CSV=..\LMSPS\data\AMAZON_filtered
set TOPK=10
set RES=output\AMAZON\v3
set OUT=output\AMAZON\v3_check
set FILES=han_amazon_lp_v3_f6_a1_s1 han_amazon_lp_manual_6_s1 han_amazon_lp_rand6_d3_s1 gat_union_s1 popularity_s1
set REPORT=%OUT%\recommend_6path_%USER_ID%.txt

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
if not exist "%OUT%" mkdir "%OUT%"

> "%REPORT%" echo recommend.py: user %USER_ID%, products %ASIN1% %ASIN2%, top %TOPK%
for %%F in (%FILES%) do call :one %%F
type "%REPORT%"
echo.
echo Saved to %REPORT%
exit /b 0

REM :one <result name> -- the user query and the product queries for one result
:one
set NAME=%~1
echo   %NAME%
>> "%REPORT%" echo.
if not exist "%RES%\%NAME%.pt" (
    >> "%REPORT%" echo === %NAME%: not found in %RES%, skipped
    exit /b 0
)
>> "%REPORT%" echo === %NAME%
%PY% recommend.py --emb %RES%\%NAME%.pt --user %USER_ID% --users-csv "%CSV%\filtered_users.csv" --products-csv "%CSV%\filtered_products.csv" --exclude-seen --topk %TOPK% >> "%REPORT%" 2>&1
REM popularity's product embedding is a single number, so "similar products" means nothing for it
if not "%NAME:popularity=%"=="%NAME%" exit /b 0
for %%A in (%ASIN1% %ASIN2%) do %PY% recommend.py --emb %RES%\%NAME%.pt --asin %%A --products-csv "%CSV%\filtered_products.csv" --topk %TOPK% >> "%REPORT%" 2>&1
exit /b 0

:usage
echo usage: %~nx0 ^<customer id^> [ASIN] [ASIN]
echo   customer id: the ID column of filtered_users.csv, or a user index 0-53568
echo   ASIN: products for the "similar products" query, e.g. one book and one DVD
exit /b 1
