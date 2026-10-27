# HAN review part H1: data, split and hand-over from LMSPS

**Scope:** what this repo takes from the LMSPS side and how it reads it:
- the data (`data/AMAZON`)
- the train/val/test split and the 99 evaluation negatives
- the meta-path sets in `arch.py`

It also covers whether files left over from the old data can leak into the runs on the rebuilt
data: graphs, saved results and the results table.

**Bottom line:** the split is right. Leftovers from the old data are not.

**The split is right.**
- The split and the negatives are exactly the ones the search used, for the same split seed.
- Every script runs on the rebuilt data's format.

**Leftovers from the old data were not caught: must fix.**
- Nothing in this repo knew which data a graph or a result came from.
- This repo reads its own `data/AMAZON`, not the LMSPS repo's. If the rebuilt data was not copied
  over, every model would have trained on the old data without a word.
- A graph left over from the old data either crashes the run with an unclear `IndexError` or, if
  it is smaller, is used without notice.
- The results table averaged a run on the old data with a run on the new data as if they were two
  seeds of one row.

**Fixed:**
- Every run prints and saves the fingerprint of its training edges. The LMSPS search records the
  same fingerprint.
- Runs stop on data without `meta.json`. Only the rebuilt data has one.
- Graphs, `eval_emb.py` and the results table check the fingerprint.
- `check_handover.py` compares this repo's data with an LMSPS search record in one command.

## Verdicts

| # | Issue | Evidence | Verdict |
|---|---|---|---|
| D1 | Same split and negatives as the search | `link_data.build_link_split` diffed against LMSPS: identical (this repo only adds the twin groups, which draw no random numbers). Run check on the rebuilt-format sample: train/val/test edges and both negative sets identical for split seeds 0, 1 and 7; the training-edge fingerprint equals the one in the LMSPS search record | OK |
| D2 | The rebuilt data's format | Type ids P 0 / C 1 / U 2 and relation ids 0-7 match `dat_generator.py`; feature sizes are read from the data (products now 260-d). Every script ran end to end on the rebuilt-format sample: graph build, both audits, HAN, metapath2vec, GAT, `eval_emb.py`, `summarize_results.py` | OK. The docs still say 504-d: H7 |
| D3 | The data is not where the search's is | This repo reads `data/AMAZON` under its own folder; the rebuilt data is in the LMSPS repo. Without the copy, every script runs on the old data and nothing says so | **Must fix: fixed.** Runs stop on data without `meta.json`, which only the rebuilt data has (`--allow-legacy-data` to use old data on purpose). Every run prints its training-edge fingerprint. `check_handover.py` is the definitive check: it compares with an LMSPS search record |
| D4 | Graphs are found by file name only | `<path>_top20_split1.npz` does not change with the data. Demonstrated on small graphs: a graph from bigger data crashes with `IndexError: index 1730 is out of bounds`; a graph from smaller data trains and saves without a warning. `--skip-existing` keeps such graphs | **Fixed.** The `.meta.json` sidecar records the training-edge fingerprint; `train_han.py` stops on a graph from other data or with another node count; `--skip-existing` rebuilds a stale graph |
| D5 | Results are found by file name only | `run_all.bat` skips a run whose result file exists; `eval_emb.py` scored an old-data result against the new data's edges (flagged only as `[STORED ... mismatch?]`); `summarize_results.py` averaged an old-data and a new-data run of the same key and seed as "2 runs, seeds 1,1" | **Must fix: fixed.** Results store the data record; the table groups by data version and warns on a mix; `eval_emb.py` skips a result from other data. The rerun gets its own output folder in H6 |
| D6 | Split-seed default 0 | In every script; `run_all.bat` passes 1, and your logged runs pass 1, but the README's commands say 0 | **Fixed:** default 1 in every script, matching the search |
| D7 | Duplicate-edition breakdown on the rebuilt data | The edition merge leaves almost no twins (sample: 0 of 1,900 test edges; LMSPS audit on the full data: 0.010% of test edges). The code handles an empty "twin in train" group | OK. The README's 46% / 50% / "+7-8 points" describe the old data: H7 |
| D8 | Titles for the audit | `run_all.bat audit` points at the old C# output folder; the rebuilt pipeline writes `filtered_products.csv` to the LMSPS repo's `data/AMAZON_filtered` | Optional (titles only): H6 |
| D9 | `requirements.txt` pins scipy 1.11.4; the data was built with 1.12.0 | Data files are plain text; this repo runs in the same `lmsps_env` as LMSPS | Ignore |

## How it was checked

1. **Code:** `link_data.py` and `data_loader.py` diffed against the LMSPS repo; every reader of
   the data, the graph files and the result files read.
2. **Split identity:** the LMSPS and HAN `build_link_split` on the rebuilt-format sample (1,730
   products, built by the current `dat_generator.py`), split seeds 0, 1 and 7: every edge array and
   adjacency compared.
3. **Hand-over:** `check_handover.py` against a real LMSPS search record on the same sample prints
   `SAME`. Against another sample it prints `DIFFERENT`, listing node counts, edges and fingerprint.
   Against a Fleet v2 record it says "cannot check", since those records predate the data
   fingerprint.
4. **Leftovers, before the fix:** graphs built from a bigger and a smaller sample loaded into a run
   on the 1,730-product sample; an old result through `eval_emb.py` and through
   `summarize_results.py --group` next to a new one.
5. **Tests:** `tests/test_data_handover.py`, 8 tests, also under pytest (about 40 s). Nine
   deliberate breaks of the new checks are all caught:
   - legacy data accepted
   - an order-dependent fingerprint
   - the graph fingerprint not compared
   - the graph size not compared
   - `--skip-existing` ignoring the data
   - the summary grouping across data versions
   - no mixed-data warning
   - `eval_emb.py` ignoring the data
   - `check_handover.py` ignoring the fingerprint

## What changed

Committed on `claude/han-review` (the branch for every commit of this review):
- `link_data.py`: `edges_fingerprint`, `data_provenance` (both the same as LMSPS) and
  `check_data` (prints the data line; stops on data without `meta.json`).
- `build_metapath_graphs.py`, `train_han.py`, `train_metapath2vec.py`, `train_gat.py`:
  - the data check
  - `--allow-legacy-data`
  - `--split-seed` default 1
  - the data record in every saved result
- `build_metapath_graphs.py`: the fingerprint in the `.meta.json` sidecar; `--skip-existing`
  rebuilds a graph built from other data or with other settings.
- `train_han.py`: stops on a graph from other data or with another node count.
- `eval_emb.py`: skips a result trained on other data.
- `summarize_results.py`: a `data` column; groups by data version; warns on a mix.
- `check_edition_twins.py`: `--split-seed` default 1.
- `check_handover.py`: new.
- `tests/test_data_handover.py`: new.

## What you do differently

Before the HAN runs on the rebuilt data:
1. Copy the rebuilt `data/AMAZON` from the LMSPS repo into this repo's `data/AMAZON`:
   `node.dat`, `link.dat`, `label.dat`, `label.dat.test` and `meta.json`.
2. Delete the old graph folders under `data/AMAZON`. They would be refused anyway; deleting them
   saves the error.
3. Run `python check_handover.py "<LMSPS>/hgb/logs/Fleet v3/final_ns4/search_ns4_seed1.json"`.
   It must print `SAME`.

To re-evaluate old results on the old data, pass `--allow-legacy-data` and point `--root` at a copy
of the old data.

## How to verify

```
python tests/test_data_handover.py      # or: python -m pytest tests
```
