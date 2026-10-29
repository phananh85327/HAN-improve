# HAN review part H3: the HAN model and its training

**Scope:** `model_han.py` and `train_han.py` against the HAN paper (Wang et al., WWW 2019):
- node-level attention (one GAT per meta-path) and semantic attention (the path weights β)
- the β warm-up
- the user side, the loss and the training loop
- early stopping and determinism

Also how a training pair relates to the graphs it is scored on.

**Bottom line:** the model is implemented correctly, but its result depended on the order of the
paths in the list. That is a must-fix for comparing path sets, and it is fixed.

**The code is right.**
- The hand-written sparse backward passes PyTorch's gradcheck.
- The GAT layer equals DGL's reference `GATConv`.
- The semantic attention is the paper's eq. 7-9 exactly.

**The order of the paths in the list decided which path HAN used: must fix, fixed.**
- **The cause:** every path's GAT started from its own random weights, so the GAT in each slot of
  the list started differently, and the attention learned which slot started best.
- **With warm-up 15 (the configuration of record):** on the small sample the same 4 paths were run
  in sorted and in reversed order, 3 seeds each. In every run one path's weight dropped to about
  0.001, and it was always the path in one particular slot, whichever path sat there. Swapping the
  order alone moved test Hit@10 by 0.022 on average (up to 0.042), about the size of the
  differences between path sets.
- **Without warm-up (the paper's setting):** your own old-data logs show one path at weight 1.000
  in 11 of 13 runs, the first-listed path in all 6 four-path runs.
- **The fix:** every path's GAT now starts from the same weights. Then only the graphs tell the GATs
  apart: both orders give the same result (Hit@10 differs by 0.003), the weights belong to the
  paths, and no path drops out. `--independent-init` gives the old behaviour.

**The paper's settings still collapse onto one path.**
- Even with the fix, without the warm-up one path takes all the weight within the first epochs.
  It is now the same path in both orders (`PCP` on the small sample), so the cause is the data,
  not the list.
- The warm-up stays necessary. A `--paper-config` row is a one-path GAT and should be described as
  one (H6).

**Self-inclusion is real, but the tested fix does not help HAN: documented, not changed.**
- On your full rebuilt data, in HAN's `PUP` graph 12.5% of a training product's neighbours are
  items the same user reviewed in training, against 3.0% for a test product; `PUPUP` 12.1% vs
  4.0%. Other paths differ by at most 1.5 points.
- The remedy for this kind of mismatch is to score each fold of training edges on graphs built
  without that fold. With it, test Hit@10 did not improve (0.403 vs 0.399, 3 seeds), and `PUPUP`
  dropped out in every run.
- So HAN keeps its graphs. The thesis should say that the LMSPS search corrects for self-inclusion
  (part 3, `--wstep-kfold`) and its evaluation model does not.

## Verdicts

| # | Issue | Evidence | Verdict |
|---|---|---|---|
| M1 | Sparse matmul with a hand-written backward | `torch.autograd.gradcheck` passes, duplicate edges and chunk boundaries included | OK |
| M2 | GAT layer | Same weights as DGL's `GATConv`: outputs equal to 2e-7, gradients to 4e-6 | OK |
| M3 | Semantic attention | Equal to the paper's eq. 7-9 (scores averaged over nodes, softmax over paths) | OK |
| M4 | Results depend on the order of the paths | Table 1 | **Must fix: fixed** (identical start for every path's GAT; `--independent-init` for the old behaviour) |
| M5 | No warm-up | Old-data logs: one path at 1.000 in 11 of 13 runs. Small sample, with the fix: `PCP` takes everything in all 6 runs | Keep `--beta-warmup 15`. A `--paper-config` row shows a one-path GAT (H6, H7) |
| M6 | Self-inclusion | Measured above; the leave-fold-out graphs tested (Table 2) | Documented; no change |
| M7 | Early stopping | Warm-up epochs never become the checkpoint; any validation improvement updates it, only one above `--min-delta` resets patience; the test uses the best checkpoint with the learned β | OK |
| M8 | Determinism | Two same-seed CPU runs are bit-identical. On a GPU, scatter and index-add use atomic adds, so same-seed runs differ slightly | OK; report the spread over seeds |
| M9 | User side | A free vector per user. 8 of 81,984 test edges belong to users with no training edge (random vectors). A GAT builds its users from their items instead (H4) | OK |
| M10 | Hyperparameters | Repo defaults and `--paper-config` as documented in the README | OK |

## Table 1: the same 4 paths in two orders

Small sample (1,730 products), paths `PCP`, `PCPUP`, `PUP`, `PUPUP`, split 1, hidden 128,
warm-up 15. Sorted = in that order; reversed = `PUPUP`, `PUP`, `PCPUP`, `PCP`. Test Hit@10, and
the path whose weight dropped below 0.01:

| Seed | Independent start: sorted / reversed | Path dropped (slot) | Same start: sorted / reversed | Path dropped |
|---|---|---|---|---|
| 1 | 0.428 / 0.407 | `PUP` (3rd) / `PCPUP` (3rd) | 0.410 / 0.413 | none (lowest weight 0.23) |
| 2 | 0.469 / 0.471 | `PUP` (3rd) / `PCPUP` (3rd) | 0.454 / 0.451 | none (0.21) |
| 3 | 0.449 / 0.407 | `PCPUP` (2nd) / `PUP` (2nd) | 0.408 / 0.405 | none (0.18) |
| Order effect | mean 0.022, max 0.042 | | mean 0.003 | |

Mean Hit@10 is 0.439 with the independent start and 0.424 with the same start. On this sample,
dropping one of the four paths happened to score a little better. Which path got dropped was set
by the list order, which a comparison of path sets must not depend on. On the old full data,
with warm-up 15, no path ever dropped out, so the accuracy cost there is expected to be smaller.

Starting the semantic attention at zero as well was tested and is not needed: same order
invariance, same accuracy (0.423).

## Table 2: leave-fold-out graphs (self-inclusion remedy)

Small sample, paths `PUP`, `PUPUP`, `PCPUP`, `PUPPUP`, same-start model, warm-up 15. k0 = graphs
from all training edges (as now); k4 = each quarter of the training edges scored on graphs built
without it:

| Seed | k0 Hit@10 | k4 Hit@10 | `PUPUP` weight k0 / k4 |
|---|---|---|---|
| 1 | 0.397 | 0.386 | 0.34 / 0.005 |
| 2 | 0.408 | 0.413 | 0.30 / 0.000 |
| 3 | 0.392 | 0.410 | 0.31 / 0.019 |

## How it was checked

1. **Code:** both files in full.
2. **Numerical checks:** gradcheck, the comparison with DGL, the semantic attention against the
   paper's equations (now `tests/test_han_model.py`).
3. **Your old-data logs:** β after the first epoch and at the end of every warm-up 0 and
   warm-up 15 run in `output/AMAZON/Han_f*_beta_*`.
4. **Experiments on the small sample:** 38 HAN runs:
   - the two orders with each initialisation, with and without warm-up
   - the leave-fold-out graphs
5. **Self-inclusion:** measured on the full rebuilt structure (H2's copy).
6. **Tests:** `tests/test_han_model.py`, 5 tests:
   - the sparse backward
   - DGL equality (skipped where dgl is missing)
   - the semantic attention
   - the same start makes the output independent of the path order, the independent start does not
   - GATs that start equal still separate once trained

## What changed

Committed on `claude/han-review`:
- `model_han.py`: `HAN_LP(symmetric_init=True)`, which gives every path's GAT the same starting
  weights.
- `train_han.py`: `--independent-init`.
- `tests/test_han_model.py`: new.

## Against the paper's text (added 2026-10-29)

H3 was first checked against the equations as the code and the README quote them. With the
paper itself (Wang et al., WWW 2019; your `2.pdf`), every item was checked again:

| Paper | This repo | Verdict |
|---|---|---|
| Def. 3.3: a node's meta-path neighbours include itself | Self-loops are added to every graph (`train_han.py`) | Same |
| Eq. 1: type-specific projection before attention | Each path's GAT projects the product features (`fc`, no bias); only products are encoded, so one type | Same |
| Eq. 2-3: masked attention over N_i (self included), LeakyReLU, softmax over neighbours | `GATLayer`: LeakyReLU(0.2), softmax over each node's incoming edges | Same (the halves of the attention vector are labelled the other way round; same function) |
| Eq. 4-5: ELU after aggregation, K heads concatenated | ELU, heads concatenated | Same |
| Eq. 7: W, b and q shared by all paths, scores averaged over all nodes, q 128-d | `SemanticAttention`: Linear(hidden, 128), tanh, Linear(128, 1, no bias), mean over products | Same |
| Eq. 8-9: softmax over paths, weighted sum | Same (plus the off-by-default `--semantic-tau`, and `--beta-warmup`) | Same; extras documented |
| Eq. 10: cross-entropy over labelled nodes | BPR over (user, product) pairs | Deviation, documented (the paper allows a task-specific loss) |
| Sec. 3: neighbours from the full adjacency product | Top 20 per hop, symmetrised | Deviation, documented (H2) |
| Sec. 5.3: Adam, lr 0.005, L2 0.001, q 128, K 8, dropout 0.6, patience 100 on validation loss, 64-d | `--paper-config`: exactly these (dropout on features and attention, as in the reference code) | Same |
| Sec. 5.2: GAT and metapath2vec run on each single meta-path, best one reported | GAT on the union of all relations; metapath2vec walks a set's paths together | Different baselines, both valid; now stated in the README. The "best single path" rows are not run |
| Sec. 5.6 / Fig. 5: a path's weight β goes with how well that path does alone | Not tested by the protocol | Optional: single-path HAN runs (below) |

**No bug or must-fix in the model.** Two things the paper's text adds:
- **Reading β with near-empty paths.** Eq. 7 averages each path's score over all products. For
  the LMSPS sets' category paths, about 80% of products have only themselves in that channel,
  so the weight mostly measures a self-only channel. The README sentence written on 2026-10-29
  said to read such a weight as "DVDs and videos"; that was wrong and is corrected.
- **Optional, not in the protocol:** HAN on one path at a time is the paper's GAT baseline
  (Sec. 5.2) and the test of its Fig. 5 claim. For the 14 distinct paths of `a1`, `_c`, M5
  and M6 it would take about 2-3 GPU hours, and it would also give LMSPS part 4b a path-level
  comparison with the search.
