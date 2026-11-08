@echo off
REM recommend_4path.bat -- recommend.py on the 4-path representative result of each kind of
REM meta-path set, for HAN and for metapath2vec, plus GAT, with the same user and products. The
REM main pick is recommend_6path.bat.
REM
REM   recommend_4path.bat <customer id> [ASIN] [ASIN]
REM
REM Why these results (README "Example recommendations"; seed 1 for all, since every set has a
REM seed-1 run for both models):
REM   han_amazon_lp_v3_f4_a1_s1   automatic: the pre-registered main LMSPS set of 4 paths
REM   han_amazon_lp_manual_5_s1   manual: its main-comparison partner, M5
REM   han_amazon_lp_rand4_d4_s1   random: the median of the five 4-path draws by HAN's validation
REM                               BPR (a typical draw, not the luckiest)
REM   m2v_amazon_lp_v3_f4_a1_s1   metapath2vec on the same automatic set
REM   m2v_amazon_lp_manual_5_s1   metapath2vec on the same manual set
REM   m2v_amazon_lp_rand4_d2_s1   random: the median draw by metapath2vec's own validation BPR
REM                               (HAN's pick, d4, is its best)
REM   gat_union_s1                GAT, the best model (no meta-paths)
REM Popularity is left out: it gives every user the same bestsellers.
REM
REM customer id: the ID column of filtered_users.csv, or a user index 0-53568. The header of each
REM list says how many products the user reviewed; users with about 6-20 make the clearest lists.
REM ASIN: products for the "similar products" query, e.g. one book and one DVD.
REM
REM Output: output\AMAZON\v3_check\recommend_4path_<customer id>.txt, one section per result,
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
set FILES=han_amazon_lp_v3_f4_a1_s1 han_amazon_lp_manual_5_s1 han_amazon_lp_rand4_d4_s1 m2v_amazon_lp_v3_f4_a1_s1 m2v_amazon_lp_manual_5_s1 m2v_amazon_lp_rand4_d2_s1 gat_union_s1
set REPORT=%OUT%\recommend_4path_%USER_ID%.txt

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
for %%A in (%ASIN1% %ASIN2%) do %PY% recommend.py --emb %RES%\%NAME%.pt --asin %%A --products-csv "%CSV%\filtered_products.csv" --topk %TOPK% >> "%REPORT%" 2>&1
exit /b 0

:usage
echo usage: %~nx0 ^<customer id^> [ASIN] [ASIN]
echo   customer id: the ID column of filtered_users.csv, or a user index 0-53568
echo   ASIN: products for the "similar products" query, e.g. one book and one DVD
exit /b 1
