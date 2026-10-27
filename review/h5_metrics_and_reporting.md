# HAN review part H5: metrics and reporting

**Scope:** how a run's numbers are computed and collected:
- `han_utils.py`: per-edge, per-user, by degree and by duplicate-edition twin
- `eval_emb.py`: recomputing from saved results
- `summarize_results.py`: the results table

**Bottom line:** the metrics are computed correctly. Two problems were in how they reach the
results table.

**1. Different settings were averaged as seeds: must fix, fixed.**
- The table grouped runs only by model, path set, preset and split.
- Your warm-up 0 and warm-up 15 runs of one key, or a tau or hidden-size change, would have been
  averaged into one "mean ± std" row as if they were seeds.
- A `variant` column now lists the settings that differ between runs of a model, and runs are
  grouped by it.

**2. The sampled metric flatters popularity: added a full-ranking check.**
- Every model is scored by ranking a test product against 99 random products. Against random
  products, a popular one wins easily.
- On your rebuilt data, popularity alone scores Hit@10 **0.42** that way, but **0.009** when the
  same test products are ranked against all products.
- Sampled metrics can also rank two models differently from full ranking (Krichene & Rendle,
  KDD 2020).
- `eval_emb.py --full-ranking` and `summarize_results.py --full-ranking` now add full-ranking
  numbers, computed afterwards from saved results.

## Verdicts

| # | Issue | Evidence | Verdict |
|---|---|---|---|
| E1 | Per-edge Hit/NDCG/Precision/Recall@k | Brute force on scores with many ties: identical. Ties rank pessimistically, the positive below every equal score | OK |
| E2 | Per-user metrics | Brute force: duplicate negatives kept once, other positives of the user as candidates, IDCG over min(positives, k): identical | OK |
| E3 | Degree and twin breakdowns | Equal to the per-edge metrics on the masked edges; an empty group is left out | OK |
| E4 | Chunked scoring | Equal to the unchunked dot products | OK |
| E5 | Twin groups | Identical reviewer sets grouped; the flag fires only for a train edge to another edition | OK. On the rebuilt data 40 of 81,984 test edges have a twin, so that column is close to empty |
| E6 | Settings averaged as seeds | Your logs ran the same keys with warm-up 0, 5 and 15; nothing in the table told them apart | **Must fix: fixed.** `variant` column; also separates tau, hidden size, learning rate, and so on |
| E7 | Sampled metric only | Popularity: sampled Hit@10 0.42, full ranking 0.009 (rebuilt data, split 1) | **Added:** full ranking (Hit/NDCG@k and MRR per edge, Recall/NDCG@k per user). Recommended for the headline rows; the runs themselves are unchanged |
| E8 | Validation selection | The trainers keep the best-validation checkpoint; the README's selection rule (choose by validation BPR, never by test) is right | OK |
| E9 | Graph provenance | A result did not record which graphs it trained on, and the tie seed changes them (H2) | **Added:** HAN results store each graph's `.meta.json` |

## Full ranking

`han_utils.evaluate_full_ranking`:
- **Candidates:** a test edge (u, p) ranks p against every product u has no observed edge to.
  Observed means training, validation or test edges; p itself stays in.
- **Per edge:** Hit@k, NDCG@k and MRR, averaged over edges.
- **Per user:** all of a user's test products ranked together. Recall@k and NDCG@k,
  macro-averaged over users, the protocol of most graph recommender papers since NGCF and LightGCN.
- **Cost:** about 2 minutes on a CPU for the whole test set; seconds on a GPU.

## How it was checked

1. **Code:** every function in `han_utils.py`, both reporting scripts.
2. **Tests:** `tests/test_metrics.py`, 6 tests against brute-force implementations, on integer
   scores so that ties are frequent.
3. **Popularity, full data:** `popularity_baseline.py` on the structure-only copy of the rebuilt
   data (H2), then `eval_emb.py --full-ranking` (the stored metrics are reproduced exactly) and
   `summarize_results.py --full-ranking`.

## What changed

Committed on `claude/han-review`:
- `han_utils.py`: `evaluate_full_ranking`, `print_full_ranking`.
- `eval_emb.py`: `--full-ranking`.
- `summarize_results.py`:
  - the `variant` column and grouping by it
  - `--full-ranking` (with `--root`, `--cpu`, `--gpu`)
  - the CSV now keeps every column
- `train_han.py`: each result stores its graphs' `.meta.json` (`graphs`).
- `tests/test_metrics.py`: new; `tests/test_data_handover.py`: `test_summary_keeps_settings_apart`.
