# HAN review: wrap-up before the runs

**Scope:** parts H1-H7 re-checked now that the LMSPS output exists (the Fleet v3 records and the
six sets in `arch.py`), plus `recommend.py`, the one file no part had covered. What is left needs
the runs.

**Bottom line:** one must-fix, done. `run_all.bat` looked up only the `_a<i>` keys, so it would
have skipped the two `_c` sets. It now lists the six Fleet v3 sets explicitly, like the manual
and random sets. The GPU memory check now uses the largest set. Everything else holds with the
real LMSPS output.

## Checked with the LMSPS output

| # | Check | Result | Verdict |
|---|---|---|---|
| W1 | `check_handover.py` against a real Fleet v3 record (`final_ns4/search_ns4_seed1.json`), on the rebuilt data's structure | SAME, training-edge fingerprint `c5f7bccbf61c1b0f` | OK |
| W2 | The six sets in `arch.py` against the re-selection output | Identical, in both repos | OK |
| W3 | The run list | `run_all.bat` trained 4 LMSPS sets: its key lookup matched `_a<i>` only | **Must fix: fixed.** `LMSPS` lists the six keys; a test checks it lists exactly the Fleet v3 keys in `arch.py` |
| W4 | Graphs for all 32 distinct paths of the 24 sets, full rebuilt data, CPU | All built: 416 s in total, at most 1.0 GB of memory | OK |
| W5 | The LMSPS sets' category paths | Isolated products: `PCCP` 81%, `PCCPP` 79%, `PCCPCP` 79%, `PCCPPP` 78%, `PPCCPP` 77%, `PCCPUP` 76% (the LMSPS repo measured the same) | Expected. In HAN those products see only themselves in that channel; README says how to read it |
| W6 | HAN and metapath2vec on sets with those paths (small sample, `amazon_lp_v3_f6_a1` and `_f4_c`) | Both train; no NaN; HAN's path weights spread over all six | OK |
| W7 | metapath2vec walks into a dead end (a book's walk along `PCCP`) | The walk stops there (`-1` padding); nothing breaks | OK |
| W8 | The GPU memory check (`smoke`) used M6 | The largest set is `amazon_lp_rand6_d2`, 9.3M graph edges against M6's 7.7M | **Fixed:** `SMOKE_KEY` is that set. The old data's 6-path sets had 19.4M edges and fit on your card, so a failure is unlikely |
| W9 | The frozen random sets | The 36 candidates are the same in Fleet v3 | OK |
| W10 | `recommend.py` (not reviewed before) | Works on the rebuilt data: products by ASIN, users by `--users-csv` (the rebuilt `filtered_users.csv` has one row per user in node-id order). Its docstring still says node.dat holds user display names; the rebuilt node.dat holds the customer ids | Ignore: a demo tool, not part of the evaluation |
| W11 | `check_edition_twins.py`'s usage example points at the old C# output folder | `--products-csv` is optional and `run_all.bat` doesn't pass it | Ignore |
| W12 | Does `push` upload the run logs? | `.gitignore`'s `logs/` matched every folder named `logs`, including `output/AMAZON/v3/logs`: `push` would have uploaded `results.csv` but no log | **Must fix: fixed** (found after the wrap-up commit). The rule is `/logs/`, the top-level folder only |

## Review status

| Part | Status |
|---|---|
| H1 data, split, hand-over | Done |
| H2 graph construction | Done; the full graphs checked again here (W4, W5) |
| H3 HAN model and training | Done |
| H4 baselines | Done |
| H5 metrics and reporting | Done |
| H6 run protocol | Done; updated here (six LMSPS sets, smoke key) |
| H7 docs | Done; updated here |
| **H8 results** | Done after the runs: `h8_results.md` (2026-10-30) |
| **LMSPS part 4b** (does HAN rank the sets as the search does?) | Done: LMSPS `review/part4b_search_vs_han.md` (2026-10-30) |
| **LMSPS part 10** (the synthesis) | Done: LMSPS `review/part10_synthesis.md` (2026-10-30) |

## What changed

Not committed yet:
- `run_all.bat`: the six Fleet v3 keys listed in `LMSPS`; the "sets not in `arch.py` yet"
  helpers removed; `SMOKE_KEY=amazon_lp_rand6_d2`; header.
- `arch_keys.py`: removed (it only looked up keys that did not exist yet).
- `tests/test_run_all.py`: `LMSPS` must list exactly the Fleet v3 keys in `arch.py`; the
  `arch_keys.py` test removed. Three deliberate breaks are caught: a set left out, a misspelt
  set, a misspelt label. 27 tests pass.
- `README.md`: inputs (the six sets), run counts, how to read `_a2`/`_c` and the category paths,
  the smoke set, the files table.
- `CLAUDE.md`: the six sets.
- `review/h6_run_protocol.md`: the six sets and why, the cost row, the `run_all.bat` section,
  the checks, the smoke set.
- `review/wrap_up_before_runs.md`: this note.

## How it was checked

- `python check_handover.py --root <rebuilt data> "<LMSPS>/hgb/logs/Fleet v3/final_ns4/search_ns4_seed1.json"`
- `python build_metapath_graphs.py --root <rebuilt data> --paths <all 32> --graph-dir <scratch>`:
  time, memory and isolated products per graph.
- `train_han.py` and `train_metapath2vec.py` on the small sample with both LMSPS sets, CPU.
- `python -m pytest tests`.
