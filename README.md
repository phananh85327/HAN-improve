# han-recsys: HAN link-prediction recommender for auto-discovered meta-paths

Standalone PyTorch reimplementation of HAN (WWW-2019) for product recommendation
on the Amazon-Meta HGB dataset. Consumes meta-paths discovered by ANY automatic
discovery method (LMSPS or otherwise): a discovery method only has to output a
list of path strings that start and end at the target type, e.g. "PUP,PPPP".
The exact requirements a discovery repo must satisfy -- schema, target-type,
leakage and fairness rules -- are in [METAPATH_CONTRACT.md](METAPATH_CONTRACT.md).

Dependencies: torch (+CUDA), numpy, scipy, tqdm. No dgl / torch_sparse.

## Inputs

- `data/AMAZON/` -- HGB-format node.dat / link.dat / label.dat[.test]
  (built by dat_generator.py from the C# preprocessing pipeline)
- a comma-separated meta-path list (from your discovery method), or an
  `arch.py` key via `--arch` (e.g. `amazon_lp_han` = the discovered set of
  record, `amazon_lp_manual` = the hand-picked ablation set)

## Flow

    # 1. materialize each meta-path as a product-product graph (train-split-safe).
    #    Output files are named per path, so builds for different arches coexist.
    python build_metapath_graphs.py --dataset AMAZON --arch amazon_lp_han --topk 20 --split-seed 0

    # 2. HAN -- config of record. --beta-warmup 15 is REQUIRED: it freezes the
    #    semantic attention to uniform for the first 15 epochs so every per-path
    #    GAT gets gradient signal; without it, attention collapses onto 1-2 paths
    #    at initialization (rich-get-richer) and the other paths are never used.
    python train_han.py --dataset AMAZON --arch amazon_lp_han --split-seed 0 \
        --stage 150 --patience 10 --beta-warmup 15 --seed 1 \
        --save-emb output/AMAZON/han_emb_auto_s1.pt

    # 2b. baselines on the same split (thesis comparison rows)
    python train_metapath2vec.py --dataset AMAZON --arch amazon_lp_han --split-seed 0 \
        --epochs 50 --evals-per-epoch 4 --patience 6 --seed 1 \
        --save-emb output/AMAZON/m2v_emb_auto_s1.pt
    python train_gat.py --dataset AMAZON --split-seed 0 --stage 50 --patience 4 --seed 1 \
        --save-emb output/AMAZON/gat_emb_s1.pt

    # 2c. ablations -- does AUTOMATIC discovery beat manual / random paths?
    #     (same pipeline with --arch amazon_lp_manual; materialize in step 1 first)
    python random_paths.py --seed 0     # prints a seeded random valid path list

    # 3. recompute test metrics from ANY saved embeddings -- no retraining.
    #    Rebuilds the deterministic split from the blob's own split_seed, prints
    #    per-edge, per-user and degree-stratified metrics, and cross-checks each
    #    per-edge number against the metrics stored at train time.
    python eval_emb.py --emb output/AMAZON/*.pt "../LMSPS/hgb/output/AMAZON/lmsps_emb.pt"

    # 4. serve queries (works with any saved *_emb.pt, incl. lmsps_emb.pt)
    python recommend.py --emb output/AMAZON/han_emb_auto_s1.pt --interactive \
        --products-csv <...>/filtered_products.csv --users-csv <...>/filtered_users.csv --exclude-seen

## Protocol knobs (keep consistent across everything you compare)

- `--split-seed` pins WHICH edges are train/val/test and the fixed 99 eval
  negatives per test edge -- identical across all models and both arches, so
  every row takes the same exam.
- `--seed` is the RUN seed (init, dropout, shuffling, BPR negatives, walks).
  Final tables should be mean +/- std over 3 run seeds: repeat each trainer with
  `--seed 1`, `--seed 2`, `--seed 3`. WARNING: the default `--save-emb auto`
  writes the same file every run -- pass a per-seed filename (as above) or
  `--save-emb ""` for the extra seeds, or the last seed silently overwrites.
- Early stopping (all trainers): the best-val checkpoint is always kept; an
  epoch only resets patience if it improves val bpr by more than `--min-delta`
  (default 5e-4). `--min-delta 0` restores strict run-to-plateau behavior.
  train_metapath2vec additionally takes `--evals-per-epoch` (validation checks
  per corpus pass; its patience counts EVALUATIONS) because SGNS peaks against
  the ranking objective inside the first epochs.
- `--vram-fraction` (default 0.9) caps the CUDA allocator below total VRAM so
  overshoot raises a clean OOM instead of Windows silently paging GPU memory to
  system RAM (multi-x slowdowns) or bluescreening the driver.

## Metrics

Each trainer (and eval_emb.py) reports two tables:
- **per-edge**: each test edge's positive ranked against its own 99 sampled
  negatives. With one positive per ranking, Recall@k == Hit@k and
  Precision@k = Hit@k / k by construction.
- **per-user**: all of a user's test positives ranked together against the
  deduplicated union of that user's negatives, macro-averaged over users --
  the four metrics genuinely differ. Not comparable to per-edge numbers;
  report separately.
Plus a Hit/NDCG breakdown by user training degree (cold vs heavy users).

All trainers default to `--user-tower embedding`: the user side is a free
per-user trained vector, matching the per-user capacity metapath2vec/GAT get
implicitly, so the ranking numbers compare ITEM encoders (the meta-path
question) instead of user-representation capacity. Pass `--user-tower feats`
for the original inductive hash-feature tower.

## Files

| file | role |
|---|---|
| data_loader.py | HGB .dat parser (verbatim from the HGB benchmark) |
| link_data.py | edge split + vectorized negative sampling |
| han_utils.py | BPR loss, chunked ranking scorer, per-edge/per-user/stratified metrics, seeding |
| build_metapath_graphs.py | sparse chain-multiply + top-k pruning per meta-path |
| model_han.py | GAT-per-path + semantic attention (pure torch, custom SpMM backward, gradient checkpointing) |
| train_han.py | training / beta warmup / early stopping / evaluation |
| train_metapath2vec.py | baseline: meta-path-guided walks + skip-gram (frozen-embedding eval) |
| train_gat.py | baseline: homogeneous GAT on the raw union graph (NO meta-paths) |
| random_paths.py | seeded random valid meta-paths (ablation baseline) |
| eval_emb.py | recompute all test metrics from saved embeddings, no retraining |
| recommend.py | ASIN -> similar products; user -> top-k products |
| METAPATH_CONTRACT.md | what a meta-path discovery repo must output for this repo to accept it |

Saved embedding blobs (`*_emb.pt`) contain P_emb/U_emb plus provenance:
model, dataset, paths, split_seed, the test metrics, and the full run config.
