# HAN review H8: the results on the rebuilt data

**Scope:** the 68 results in `output/AMAZON/v3` (32 HAN, 32 metapath2vec, 3 GAT, popularity)
after `seeds` and `summarize` (2026-10-30), their logs and `results.csv`. Whether the search
ranks sets as HAN does is LMSPS review part 4b; its findings are summarised here.

**Bottom line**
- **The runs are valid.** One data version, the protocol's settings, every run ended by early
  stopping, and `results.csv` matches the logs. One must-fix, in the table (`summarize`): done,
  not committed yet.
- **The main comparison goes against LMSPS.** On the protocol's metric (Hit@10 against 99
  sampled negatives) the manual sets beat the LMSPS sets at both sizes, for HAN and for
  metapath2vec, by 6 to 20 times the seed spread. On full ranking HAN ties.
- **The LMSPS sets come last for HAN among the 8 sets per size drawn from the search's own 36
  candidates** (part 4b). The search and HAN agree on the paths through users and disagree on
  the category paths.
- **GAT is the best model on every metric.** HAN beats metapath2vec on the sampled metric in all
  24 sets, but metapath2vec wins on full ranking in 14 of them.

## Verdicts

| # | Item | Evidence | Verdict |
|---|---|---|---|
| R1 | The runs are what the protocol says | 67 training logs: exit code 0, fingerprint `c5f7bccbf61c1b0f`, the settings of `run_all.bat` (warm-up 15 in all 32 HAN runs) | OK |
| R2 | Every run ended by early stopping | HAN stopped at epochs 63-112 of 150, metapath2vec at 1.5-3 of 10, GAT at 130, 141 and 157. GAT seed 2 reached the 150 cap and was rerun with 300 | OK (H6 P11) |
| R3 | The table | GAT split into a `stage=150` and a `stage=300` row; before the move, the capped run was averaged as seed 2 | **Must fix: fixed** in `summarize_results.py` (the cap is not a setting; capped runs and repeated seeds are flagged). Rerun `summarize` after pulling |
| R4 | `results.csv` against the logs | All 67 trained rows: validation BPR and Hit@10 equal; no row matches a warm-up-5 log | OK |
| R5 | How big is seed noise? | Hit@10 spread over 3 seeds: 0.0017-0.0045 (HAN, metapath2vec), 0.0042 (GAT) | The yardstick for sets run at seed 1 only: differences under about 0.01 are noise |
| R6 | Does validation pick well? | Over the 24 HAN sets, validation BPR against test Hit@10: Spearman -0.94 (full-ranking Hit@10: -0.71) | OK: the "choose by validation" rule works |
| R7 | Runs still improving at the stop | Validation BPR gained in the last 10 epochs: HAN median 0.0006, max 0.0037 (M4), GAT 0.0007-0.0022; LMSPS runs at most 0.0006 | Report as a limitation. More training could only widen the gaps LMSPS loses and GAT's lead. No reruns |
| R8 | The three 6-path LMSPS sets score the same (Hit@10 0.6658-0.6659) | They differ in paths HAN drops (beta < 0.01), and `_c` also in one user path | Not a bug |
| R9 | metapath2vec on M4: Hit@10 0.09 | No path of M4 goes through a user, so users keep their random start | Report as not applicable |
| R10 | Twin columns | 40 of 81,984 test edges have a twin; `hit10_twin` jumps between 0.58 and 1.00 | Leave out of the thesis tables |
| R11 | Sampled and full ranking disagree | HAN against metapath2vec flips in 14 of 24 sets; LMSPS against manual is a tie for HAN on full ranking | Report both. The main metric was fixed in advance; full ranking is the check (Krichene and Rendle, 2020). Optional check below |
| R12 | metapath2vec evaluates the end of each epoch twice (its 4th check falls one batch before the end-of-epoch check), and both count toward patience | Validation was already getting worse after every run's best, by 0.001-0.02 | Minor, no effect on any result. Not changed: it is the code that produced the results |

## The main comparison

Test numbers, mean +/- std over seeds 1-3; Welch t-test p-values. The split is fixed, so the
spread is training noise only.

| | Hit@10 | NDCG@10 | Full-ranking Hit@10 | Full-ranking Recall@20 |
|---|---|---|---|---|
| **HAN** `v3_f4_a1` | 0.6348 +/- 0.0028 | 0.4053 +/- 0.0025 | 0.0207 +/- 0.0005 | 0.0376 +/- 0.0007 |
| **HAN** M5 | 0.6635 +/- 0.0019 | 0.4258 +/- 0.0017 | 0.0211 +/- 0.0004 | 0.0379 +/- 0.0008 |
| difference, p | **-0.029, p 0.0003** | -0.021, p 0.0006 | -0.0004, p 0.32 | -0.0004, p 0.57 |
| **HAN** `v3_f6_a1` | 0.6663 +/- 0.0017 | 0.4353 +/- 0.0015 | 0.0245 +/- 0.0004 | 0.0444 +/- 0.0014 |
| **HAN** M6 | 0.6870 +/- 0.0043 | 0.4495 +/- 0.0041 | 0.0246 +/- 0.0011 | 0.0442 +/- 0.0018 |
| difference, p | **-0.021, p 0.008** | -0.014, p 0.017 | -0.0001, p 0.87 | +0.0002, p 0.88 |
| **metapath2vec** `v3_f4_a1` vs M5 | 0.5996 vs 0.6397: **-0.040, p 0.001** | -0.031, p 0.007 | +0.0007, p 0.38 | -0.0018, p 0.46 |
| **metapath2vec** `v3_f6_a1` vs M6 | 0.5889 vs 0.6375: **-0.049, p < 0.0001** | -0.047, p < 0.0001 | -0.0051, p 0.0004 | -0.0170, p 0.0008 |

- **The answer to the main question:** the searched sets do not beat the expert sets in either
  meta-path model. They lose clearly on the metric fixed in advance. On full ranking they tie in
  HAN, and in metapath2vec they tie at 4 paths and lose at 6.
- **It holds for the other LMSPS sets:** `_a2` and `_c` (seed 1) are below M5 too (0.650 and 0.659
  against 0.664), and all three 6-path sets are below M6 (0.666 against 0.687).
- **It holds for every kind of user:** HAN's LMSPS sets lose by 0.034 / 0.031 / 0.025 Hit@10
  (4 paths) and 0.030 / 0.023 / 0.015 (6 paths) for users with 1-5 / 6-20 / 21+ training purchases.

## All 24 sets (seed 1)

| HAN Hit@10 | 4 paths | 6 paths |
|---|---|---|
| LMSPS (`a1`, `a2`, `_c`) | 0.635, 0.650, 0.659 | 0.666, 0.666, 0.666 |
| Manual | M5 0.665, M7 0.670, M8 0.663 | M6 0.692 |
| Random draws (5 per size) | 0.660-0.677, mean 0.668 | 0.673-0.706, mean 0.691 |

- **Expert sets are about as good as a random draw** from the search's candidates (M5 0.665
  against a mean 0.668; M6 0.692 against 0.691). The LMSPS sets are below all five draws of their
  size on Hit@10 and on validation BPR (part 4b).
- **More paths help HAN:** on average the 6-path sets beat the 4-path ones in every family.
- **The validation pick** over all 24 sets is `rand6_d2` (validation BPR 0.2580, test 0.706); among
  the expert sets, M6.
- **metapath2vec ranks the sets differently** (Spearman +0.09 against HAN). Which paths are "best"
  depends on the model that uses them.

## What HAN does with the paths

HAN's path weights (beta) at the end of training, seed 1, as a multiple of an equal share:

| Kind of path | Search's mean rank (of 36) | Uses in the 24 sets | Dropped by HAN (beta < 0.01) | Median weight |
|---|---|---|---|---|
| Through users (`PUPUP`, `PUP`, `PPUP`, ...) | 12.7 | 46 | 0 | 1.34 |
| Category paths that reach nothing for 76-81% of products (`PCCP`, `PCCPP`, `PPCCPP`, ...) | 11.8 | 25 | 16 | 0.00 |
| Other category paths (`PCP`, `PCCCP`, `PCPCP`, ...) | 23.8 | 22 | 3 | 1.12 |
| Product-product only (`PP`, `PPP`, ...) | 28.8 | 16 | 2 | 0.85 |

- **In every LMSPS set HAN keeps one category path and drops the others.** The one it keeps
  (beta 0.26-0.31; `_c` at 4 paths: 0.06) holds only the product itself for most books and music,
  so it acts as the product's own-content channel. `PPCCPP` is in five of the six LMSPS sets, and
  HAN drops it in every set that has it. The 6-path LMSPS sets therefore work like 4-path sets.
- **Among the 4- and 6-path sets, only the LMSPS sets lack a category path that reaches (almost)
  every product.** At both sizes, every set with one beats every set without (4 paths:
  0.660-0.677 against 0.635-0.659; 6 paths: 0.673-0.706 against 0.666). At 4 paths the closest
  pair, 0.6596 against 0.6585, is within noise.
- **A likely reason, not yet tested:** books and music are 75% of the test edges, and the LMSPS
  sets give HAN no category signal for them. The search ranked the dense category paths near the
  bottom (`PCP` 27-29 of 36, `PCCCP` 22, `PCPCP` 25-28) and the near-empty ones around 10-14. The
  two optional checks below test this.

## The baselines

| Model | Hit@10 | Full-ranking Hit@10 | Hit@10 by training purchases: 1-5 / 6-20 / 21+ |
|---|---|---|---|
| GAT (3 seeds) | **0.787 +/- 0.004** | **0.048 +/- 0.005** | 0.756 / 0.798 / 0.793 |
| HAN, validation pick (`rand6_d2`) | 0.706 | 0.028 | 0.639 / 0.700 / 0.739 |
| HAN, M6 (3 seeds) | 0.687 | 0.025 | 0.622 / 0.686 / 0.715 |
| metapath2vec, M5 (3 seeds; validation pick) | 0.640 | 0.027 | 0.555 / 0.645 / 0.670 (seed 1) |
| Popularity | 0.423 | 0.009 | 0.431 / 0.435 / 0.412 |

- **GAT wins on every metric and for every kind of user,** most for users with few purchases. It
  builds each user from the user's own items; HAN and metapath2vec learn a free vector per user
  (27.4M of HAN's 28M parameters). The size of GAT's lead for light users fits that explanation.
- **HAN against metapath2vec depends on the metric.** Popularity alone reaches 0.42 sampled and
  0.009 full: the sampled metric gives a lot of credit for knowing which products are popular.
  If HAN leans more on popularity than metapath2vec does, that would explain the flip; the
  optional check measures it.
- Every model clears the popularity floor, except metapath2vec on M4 (R9).

## For the thesis

- Report the main comparison as in the table above: the LMSPS sets lose on the protocol's metric
  and tie (HAN) on full ranking. Say which metric was fixed in advance.
- Report LMSPS against the random draws on test numbers (part 4b), and the spread of the draws,
  never one draw.
- State the stopping rule and its limit (R7), and that the split is fixed (seeds vary training
  only).
- Leave out the twin columns (R10) and metapath2vec on M4 (R9).
- Keep the explanation of the LMSPS result as a hypothesis unless the optional checks confirm it.

## Optional checks (about 25 minutes of GPU time in all)

1. **Where LMSPS loses, on the saved runs (seconds each):**

        python review/evidence/h8_by_group.py output\AMAZON\v3\han_amazon_lp_v3_f4_a1_s1.pt output\AMAZON\v3\han_amazon_lp_manual_5_s1.pt output\AMAZON\v3\han_amazon_lp_v3_f6_a1_s1.pt output\AMAZON\v3\han_amazon_lp_manual_6_s1.pt output\AMAZON\v3\m2v_amazon_lp_manual_6_s1.pt output\AMAZON\v3\gat_union_s1.pt

   Hit@10 by product group and by whether `PCCP` reaches the product, plus how popular each run's
   full-ranking top 10 is. If the hypothesis holds, the LMSPS sets lose mostly on books and music,
   and HAN's top 10 is more popular than metapath2vec's.
2. **Does a dense category path close the gap? (two HAN runs, about 10 minutes each).** Put
   `PCP` in place of `PPCCPP`, the path HAN drops in every LMSPS set. These are diagnostics, not
   LMSPS sets, so they go to their own folder:

        mkdir output\AMAZON\v3_check
        python train_han.py --paths PCCPCP,PCP,PPUPP,PUPUP --dataset AMAZON --split-seed 1 --graph-dir data\AMAZON\metapath_graphs_v3 --topk 20 --hidden 512 --num-heads 8 --feat-drop 0.2 --attn-drop 0.2 --lr 0.001 --weight-decay 0 --user-tower embedding --batch-size 100000 --vram-fraction 0.9 --stage 150 --patience 10 --min-delta 0.0005 --beta-warmup 15 --seed 1 --save-emb output\AMAZON\v3_check\han_f4_a1_pcp_s1.pt > output\AMAZON\v3_check\han_f4_a1_pcp_s1.txt 2>&1
        python train_han.py --paths PCCP,PCCPP,PCP,PPUP,PUPPUP,PUPUP --dataset AMAZON --split-seed 1 --graph-dir data\AMAZON\metapath_graphs_v3 --topk 20 --hidden 512 --num-heads 8 --feat-drop 0.2 --attn-drop 0.2 --lr 0.001 --weight-decay 0 --user-tower embedding --batch-size 100000 --vram-fraction 0.9 --stage 150 --patience 10 --min-delta 0.0005 --beta-warmup 15 --seed 1 --save-emb output\AMAZON\v3_check\han_f6_a1_pcp_s1.pt > output\AMAZON\v3_check\han_f6_a1_pcp_s1.txt 2>&1

   All their graphs are already built. If Hit@10 rises to about M5's 0.665 and M6's 0.69, the
   missing category channel explains the gap. If it stays near 0.635 and 0.666, it does not.

## What changed

Not committed yet:
- `review/h8_results.md`: this note.
- `review/evidence/h8_results.py`: every number above from the logs and `results.csv`.
- `review/evidence/h8_by_group.py`: optional check 1. Tested on the full rebuilt data with a random
  and a popularity-only run: Hit@10 0.10 and 0.4235 (the popularity baseline's 0.4233), PCCP
  reaching 18.9% of products (part 4's number).
- From the seeds review (same day): `summarize_results.py`, its test, README, H6 P11.

## How to verify

    python review/evidence/h8_results.py output\AMAZON\v3
    python "..\LMSPS\review\evidence\part4b_search_vs_han.py" "..\LMSPS\hgb\logs\Fleet v3" .
    python -m pytest tests
