# Full re-check of the repo (2026-10-30, after the rerun)

**Scope:** every file in the HAN-side flow, read again line by line now that the rerun's 68
results and their logs exist to test it against:
- data: `data_loader.py`, `link_data.py`, `check_handover.py`, `check_edition_twins.py`
- graphs: `build_metapath_graphs.py`, `check_graph_bias.py`, `random_paths.py`, `arch.py`
- model and training: `model_han.py`, `train_han.py`, `han_utils.py`
- baselines: `train_gat.py`, `train_metapath2vec.py`, `popularity_baseline.py`
- evaluation and reporting: `eval_emb.py`, `summarize_results.py`, `recommend.py`
- scripts, tests and docs: `run_all.bat`, `tests/`, `README.md`, `METAPATH_CONTRACT.md`,
  `CLAUDE.md`, `requirements.txt`

**Bottom line:** no bug that changes a result and no must-fix in the code; the rerun's results
stand.
- One dependency was missing from `requirements.txt`: scikit-learn, which every script imports
  through `data_loader.py`. Added.
- Some docs were stale, the run cost above all (estimated 20-25 GPU hours, measured 9). Fixed;
  the cost now matters for the thesis.
- Two behaviours worth stating in the thesis are now in the README:
  - metapath2vec draws its negative samples uniformly within each node type
  - in the meta-path graphs a product's walk back to itself takes one of its 20 neighbour slots

## What was checked

| Area | Checked | Result |
|---|---|---|
| Split and negatives | Unique edges, a seeded permutation, 80/10/10. Evaluation negatives avoid every positive of the user and repeat no item within an edge's 99; training negatives avoid training positives only. Same code and fingerprint as the LMSPS repo | OK |
| Data guards | Every script stops on data without `meta.json`; every graph and result carries the training-edge fingerprint; a graph from other data stops the run | OK |
| Features | Products: LSA title features plus group bits. Categories: name features. Users: hashed customer ids, so no interaction data reaches GAT's user features | OK |
| Graphs | Plain relations; user-product steps from training edges only; top-k by walk count with a seeded tie-break and fixed chunking; binarise, strip self-loops, symmetrise. All 24 builds: tie-break random, fingerprint `c5f7bccbf61c1b0f`, exit code 0; the audit shows no node-id bias | OK (note C4) |
| Random draws | The 10 draws in `arch.py` reproduce exactly from a Fleet v3 record with the commands written above them | OK |
| HAN model | One GAT per path as in the paper's eq. 1-5 (the gradcheck and DGL-equality tests pass); the edge softmax subtracts a detached maximum, which is exact; semantic attention as eq. 7-9; every path's GAT starts from the same weights | OK |
| HAN training | Warm-up epochs can't become the best checkpoint; the best checkpoint is reloaded for the test; test numbers are computed once, after training; user vectors train on training edges only | OK |
| GAT | Union of all relations with training user-product edges only; each step removes its positives in both directions; validation and test run on the full training graph | OK (note C7) |
| metapath2vec | Walks follow the templates on training edges only; negatives within the context's type; window pairs both ways; the best snapshot by validation | OK (notes C3, C5, C6) |
| Popularity | Training degree plus a tie-break below 1 | OK |
| Metrics | Per-edge ranks count ties against the model; per-user, by-degree, twin and full-ranking metrics (full ranking hides the user's other known items and keeps the target) | OK |
| Reporting | `results.csv` matches every log; the epoch-cap and repeated-seed checks (seeds review) | OK |
| Tests | `python -m pytest tests` | 28 pass |

## Findings

| # | Item | Evidence | Verdict |
|---|---|---|---|
| C1 | `requirements.txt` lacks scikit-learn | `data_loader.py` (verbatim HGB loader) imports `sklearn.metrics` at the top, so without it no script starts. Your environment has it (1.3.2) | **Fixed:** added, and named in the README's dependency line |
| C2 | The run cost in the docs | README "20-25 GPU hours", `run_all.bat` "about 20", H6's table: estimates from the old data. Measured: HAN 5.0 h (32 runs, median 9.0 min), metapath2vec 2.9 h (5.2 min), GAT 0.6 h (12.8 min), graphs 11 min: about 9 GPU hours | **Fixed** with the measured numbers (`review/evidence/h8_results.py`, section 7) |
| C3 | metapath2vec's negatives are uniform within each node type | The original samples them in proportion to frequency^0.75, so popular nodes are rarer negatives here, which can favour the sampled metric | Documented (README, deviations). The same for every set, so comparisons between sets hold. No rerun |
| C4 | A product's walk back to itself takes one of its top-20 slots | It has the highest walk count, and self-loops are stripped only after pruning. Rebuilt data, split 1: `PUP` 99.9% of rows, `PUPUP` 97.9%, `PCP` 84.4%, `PPUP` 34.7% (`review/evidence/recheck_topk_self.py`). Those rows keep about 19 other neighbours before symmetrizing | Documented (README, deviations). The same rule for every set. No rerun |
| C5 | metapath2vec's saved `epochs_run` | For an early-stopped run it holds the best snapshot's epoch + 1, not the epochs trained. Nothing reads it except the epoch-cap warning, which could misfire only if a run's best and its stop both fell in the last epoch (here the best epochs are 0.5-2 of 10) | Ignore |
| C6 | metapath2vec evaluates the end of each epoch twice | H8 R12: no effect on any result | Ignore |
| C7 | GAT removes all of a batch's 100,000 positives at each step (about 15% of the training user-product edges), not only the scored edge | Like edge dropout during training; validation and test use the full training graph | Ignore: the standard remedy (H4) |
| C8 | Two stale README lines | "All trainers default to `--user-tower embedding`" (only `train_han.py` has the flag); "review/: H1-H7, H8 follows the rerun" | **Fixed** |
| C9 | `check_edition_twins.py --jaccard` counts the test user in both reviewer sets | Inflates near-duplicate shares a little; an audit of the old data, not used by any result | Ignore |
| C10 | `recommend.py`'s docstring describes the old node names (wrap-up W10) | A demo tool | Ignore |

## Changed

Not committed yet:
- `requirements.txt`: scikit-learn 1.3.2.
- `README.md`: dependency line; the measured run cost; the user-tower sentence; two deviation
  notes (the self slot in top-k, metapath2vec's negatives); the `review/` row.
- `run_all.bat`: header comment only (the cost).
- `review/h6_run_protocol.md`: the cost table with the measured numbers.
- `review/evidence/h8_results.py`: section 7, minutes per run from the logs.
- `review/evidence/recheck_topk_self.py`: the C4 measurement.
- `CLAUDE.md`: points to this note.
- `review/recheck_full_repo.md`: this note.

## How to verify

    python -m pytest tests
    python review/evidence/h8_results.py output\AMAZON\v3
    python review/evidence/recheck_topk_self.py data PUP PCP PPUP PUPUP
    python random_paths.py --from-record "..\LMSPS\hgb\logs\Fleet v3\final_ns4\search_ns4_seed1.json" --size 4 --draws 5 --seed 4 --prefix amazon_lp_rand4
