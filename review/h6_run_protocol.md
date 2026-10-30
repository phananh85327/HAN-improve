# HAN review part H6: the run protocol and `run_all.bat`

**Scope:** what the HAN side runs on the rebuilt data, which runs are repeated with other seeds,
and the batch file that runs it all.

**Bottom line:** the old `run_all.bat` could not run the rerun. It is rewritten around how you
already work, with a few additions the thesis needs.

**What the old file would have done:**
- stopped on `amazon_lp_random4_s0`, which is no longer in `arch.py`
- trained the Fleet v2 sets instead of the Fleet v3 ones
- written into `output\AMAZON` beside your old-data results, and skipped any run whose old result
  had the same name
- logged every run's exit code as 0 (it recorded the date command's code, not the run's)

**What you did before, and what is kept:**
- all three models
- seed 1
- the LMSPS sets and your manual sets
- your hyperparameters, chosen to fit your PC

**What is added, and why:**
- **Popularity:** the floor every model has to clear (H4).
- **Random sets:** 5 random sets of 4 paths and 5 of 6, drawn from the 36 candidates the search
  itself chooses from.
  - The question it answers: does the search pick better than chance?
  - They are frozen in `arch.py` now, before any Fleet v3 result exists.
- **Seeds 2 and 3, for the main comparison only.** That is HAN and metapath2vec on the first LMSPS
  set and the manual set of each size, plus GAT.
  - On the small sample, the same HAN run with seeds 1-3 moved by up to 0.05 Hit@10, as much as
    between path sets.
  - With one seed, a win of 0.01 can't be told from luck. On the full data the spread will be
    smaller, but it has to be measured before it can be reported.
- **Two checks before the long runs:**
  - `check`: the data must be the data the search ran on (H1).
  - `smoke`: a few epochs of the biggest runs, to see they fit in GPU memory.

## What runs

A run is one model trained on one meta-path set with one seed.

| Path sets | HAN | metapath2vec | Seeds |
|---|---|---|---|
| LMSPS Fleet v3: `amazon_lp_v3_f4_a1`, `_a2`, `_c` and `amazon_lp_v3_f6_a1`, `_a2`, `_c` (per size, the first two averaged one-SE sets and the vote's consensus; LMSPS review part 4) | yes | yes | 1 |
| Manual M1-M8 (`amazon_lp_manual_1..8`) | yes | yes | 1 |
| Random: `amazon_lp_rand4_d1..5`, `amazon_lp_rand6_d1..5` | yes | yes | 1 |
| **Main comparison:** `amazon_lp_v3_f4_a1` against `amazon_lp_manual_5` (4 paths), `amazon_lp_v3_f6_a1` against `amazon_lp_manual_6` (6 paths) | yes | yes | **1, 2, 3** |
| GAT (no meta-paths) | | | **1, 2, 3** |
| Popularity (no training) | | | 1 |

- **Why `a1` is the LMSPS set of the main comparison.** It is the simplest averaged one-SE set
  (LMSPS review part 5), fixed before any result. The other LMSPS sets are in the full table; if
  you pick a "best" one, pick it by validation BPR, never by a test number.
- **Why only six LMSPS sets (added 2026-10-29, after the fleets, before any HAN run).** The ns 6
  one-SE window holds 32 near-identical sets, so "every `_a<i>`" would have meant 32 HAN runs of
  nearly one model. `_a2` and the vote's consensus `_c` show whether the conclusion depends on
  which LMSPS set stands for the search (LMSPS review part 4).
- **Why M5 and M6.** They are the expert sets with the same number of paths as the LMSPS sets
  (`review/manual_metapath_sets.md` in the LMSPS repo), also fixed before any result.
- **Why 4 and 6 paths.** One short set and one long set, as you chose, is a sound design: it
  shows the conclusion does not hinge on one budget. Compare like with like (4 with 4, 6 with 6):
  on the old data the 6-path sets were ahead by about 0.012 Hit@10.
- **Hyperparameters.** The repo's settings (hidden 512, lr 0.001, dropout 0.2, warm-up 15,
  patience 10) are a fair choice for your hardware, and the thesis just needs to state them. The
  paper's own settings make HAN's attention collapse onto one path on this data (H3), so the
  `--paper-config` rows are dropped. `--paper-walks` (about 60 GPU hours) is dropped too.
  `run_all.bat` writes every model's settings out in full (`HAN_REC`, `GAT_REC`, `M2V_FAST`;
  added 2026-10-29). Parsed by the scripts themselves, they equal your earlier runs on your PC,
  except GAT's longer budget (150 epochs and patience 10, against 50 and 5), which it needs
  now that it cannot learn from its own target edges.

**Cost:** estimated at about 20-25 GPU hours from your old logs; **measured on the rebuilt data
(2026-10-30): about 9 GPU hours.** Each epoch does about a tenth of the old work (fewer training
edges, graphs half the size; H8):

| Runs | Count | Estimated per run | Measured per run (median, range) | Measured total |
|---|---|---|---|---|
| HAN, seeds 1-3 | 32 (24 sets at seed 1, 8 for seeds 2-3) | 20-45 min | 9.0 min (5.6-13.7) | 5.0 h |
| metapath2vec, seeds 1-3 | 32 | 5-10 min | 5.2 min (2.4-10.0) | 2.9 h |
| GAT, seeds 1-3 | 3 | about 1 h | 12.8 min (11.4-14.1) | 0.6 h |
| Graphs, audit, popularity, table | | | | about 15 min |

## `run_all.bat`

`run_all.bat <stage>`. `all` runs every stage in order:

1. `check`
2. `build`
3. `audit`
4. `smoke`
5. `popularity`
6. `han`
7. `gat`
8. `m2v`
9. `seeds`
10. `summarize`
11. `push`

`dry` prints every command and runs nothing.

- **Stop and restart freely:**
  - every run writes `output\AMAZON\v3\<model>_<key>_s<seed>.pt`, and a run whose file exists is
    skipped
  - a run cut off half-way has no file, so it starts again
  - any stage can be run alone
- **Manual runs:**
  - `run_all.bat dry` prints the exact command of every run, with its result and log names
  - a run you start by hand with the same `--save-emb` name counts as done
- **Logs:**
  - `output\AMAZON\v3\logs\<same name>.txt`, with the command, start and end time and the run's
    exit code
  - a failed run prints `FAILED` and the stage moves on
- **Results:** `summarize` writes `output\AMAZON\v3\results.csv` and prints the table, grouped over
  seeds, with the full-ranking columns (H5).
- **Pushing:** `push` commits the logs and the table, not the `.pt` files (git-ignored), and
  pushes, as in the LMSPS fleet script. `logs\` is git-ignored in this repo, so the logs now live
  under `output\AMAZON\v3`.
- **Stopping early:**
  - `check` stops `all` when the data is not the search's
  - `build` and `smoke` stop it on a failure
- **The sets:** `LMSPS`, `MANUAL` and `RANDOM_SETS` list the `arch.py` keys explicitly. (Before
  the fleets finished, `arch_keys.py` looked the Fleet v3 keys up; with the six sets fixed it was
  removed on 2026-10-29.)
- **Settings at the top of the file:**
  - `SEARCH_RECORD`: the Fleet v3 record the data is checked against (any one)
  - `HEADLINE`, `MORE_SEEDS`, `SMOKE_KEY`, and the output and graph folders
  - `CONDA_ROOT`: as in the LMSPS script

## Verdicts

| # | Issue | Evidence | Verdict |
|---|---|---|---|
| P1 | The old `run_all.bat` cannot run the rerun | Removed random keys (KeyError), Fleet v2 keys, the old output folder (old results skipped by name or mixed into the table), `--paper-config` stages | **Must fix: rewritten** |
| P2 | Interrupted runs | Runs are skipped by result file; stages run alone; `dry` | Done |
| P3 | Data hand-over before the runs | `check` stage (`check_handover.py`, H1) | Added |
| P4 | GPU memory | GAT now builds an extra edge list per step (H4); HAN's 6-path sets are the biggest runs | `smoke` stage (HAN on the largest set, GAT). With the full graphs built (2026-10-29) the largest is `amazon_lp_rand6_d2`, 9.3M edges, not M6 (7.7M) |
| P5 | Exit codes in the logs | The old log line took `%ERRORLEVEL%` after the date command | Fixed: the run's own code |
| P6 | One seed for everything | Seed spread on the small sample is as large as the differences between sets | Seeds 2-3 for the main comparison |
| P7 | Random sets | The old ones were schema walks and no longer in `arch.py` | 10 draws from the search's 36 candidates, frozen in `arch.py` |
| P8 | Logs could not be pushed | `logs/` is in `.gitignore` | Logs under `output\AMAZON\v3`; `push` stage. The rule `logs/` matched every folder named `logs`, so it ignored `output/AMAZON/v3/logs` too; anchored to `/logs/` on 2026-10-29 |
| P9 | Line endings | A batch file checked out with LF endings can lose its labels in cmd | `.gitattributes`: `*.bat` always CRLF |
| P10 | Paper-setting rows | HAN collapses without warm-up (H3); `--paper-walks` costs about 60 h | Dropped |
| P11 | A run that reaches its epoch cap | GAT seed 2 ran all 150 epochs (best 147) and was rerun with `--stage 300`; it stopped at 157. `summarize` counted `stage` as a setting: the rerun got a row of its own, and the capped run, still in the folder, was averaged as seed 2 | **Must fix (2026-10-30):** the cap is not a setting; `summarize` warns about a capped run and about a seed counted twice. Every other run stopped early, far below its cap |

## How it was checked

- **Static checks, since cmd can't run here:** `tests/test_run_all.py`, 3 tests:
  - every label used is defined
  - every stage has a label
  - every script exists and has every flag the file passes it
  - every set in the file exists in `arch.py`, and `LMSPS` lists exactly the Fleet v3 keys there

  Deliberate typos are all caught: a misspelt flag, a missing label, an unknown key, a missing
  script, an unknown stage, a Fleet v3 set left out.
- **The random draws** are reproduced by the commands written above them in `arch.py`. The 36
  candidates are the same with the Fleet v3 flags (checked against a search record with those flags).
- **On your PC, first:** `run_all.bat dry`, then `run_all.bat check`.

## What changed

Committed on `claude/han-review`:
- `run_all.bat`: rewritten for the rerun.
- `arch_keys.py`: new.
- `arch.py`:
  - the 10 random draws
  - where the Fleet v3 entries go
- `.gitattributes`: new.
- `tests/test_run_all.py`: new.

After the fleets (2026-10-29): the six Fleet v3 sets in `arch.py`; `run_all.bat` lists them
and `arch_keys.py` is gone (`review/wrap_up_before_runs.md`).
