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
    #    Top-k pruning breaks walk-count ties with a seeded RANDOM key by default
    #    (--tie-break random, --tie-seed = split seed). Graphs built before this
    #    option existed used the id tie-break (--tie-break id), which made low
    #    product ids hubs -- rebuild them. Each .npz gets a .meta.json sidecar
    #    recording the mode; train_han.py warns when the sidecar is missing.
    python build_metapath_graphs.py --dataset AMAZON --arch amazon_lp_han --topk 20 --split-seed 0
    python check_graph_bias.py data/AMAZON/metapath_graphs      # per-id-decile degree audit

    # 2. HAN -- config of record. --beta-warmup 15 is REQUIRED: it freezes the
    #    semantic attention to uniform for the first 15 epochs so every per-path
    #    GAT gets gradient signal; without it, attention collapses onto 1-2 paths
    #    at initialization (rich-get-richer) and the other paths are never used.
    python train_han.py --dataset AMAZON --arch amazon_lp_han --split-seed 0 \
        --stage 150 --patience 10 --beta-warmup 15 --seed 1 \
        --save-emb output/AMAZON/han_emb_auto_s1.pt

    # 2a. paper-faithful hyperparameters (Wang et al. 2019, Sec. 5.3): lr 0.005,
    #     weight-decay 0.001, dropout 0.6, hidden 64 (8x8), patience 100, no
    #     warmup / temperature. Explicit flags still override the preset.
    python train_han.py --dataset AMAZON --arch amazon_lp_han --split-seed 0 \
        --paper-config --seed 1 --save-emb output/AMAZON/han_emb_auto_paper_s1.pt

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

## Deviations from the paper (Wang et al., WWW-2019)

The HAN model itself (`model_han.py`) follows the paper: per-path multi-head
node-level attention (eq. 1-5), semantic-level attention with node-averaged
scores and a 128-d attention vector (eq. 7-9). The task adaptation and the
pipeline around it differ; keep these in mind when reading results. A full
review is in `chat/2026-10-16_han-paper-fidelity-review.md`.

- **Loss / head.** Cross-entropy classification (eq. 10) is replaced by BPR over
  dot-product scores of (user, product) pairs; the paper explicitly allows a
  task-specific loss.
- **Meta-path neighbors are pruned.** The paper uses every meta-path based
  neighbor (adjacency product). `build_metapath_graphs.py` keeps the top-20 per
  row by walk count after every hop and symmetrizes. Ties are broken by a seeded
  random key (`--tie-break random`); the previous id tie-break biased the graphs
  toward low product ids (see `check_graph_bias.py`).
- **Users bypass HAN.** Only products are encoded by HAN; users are a free
  embedding table (or a linear tower over hash features). The paper would give
  users their own type-specific projection and meta-paths.
- **Semantic-attention extras.** `--semantic-tau` (temperature) and
  `--beta-warmup` (frozen uniform beta) are repo additions; both are off by
  default, but the config of record uses `--beta-warmup 15`.
- **Hyperparameters.** Repo defaults are lr 1e-3, no weight decay, dropout 0.2,
  hidden 512 (8 heads x 64), patience 20 with `--min-delta 5e-4`. The paper
  (Sec. 5.3) uses lr 5e-3, L2 1e-3, dropout 0.6, 64-d embeddings, patience 100:
  `--paper-config` applies those.
- **Product features.** The HGB `node.dat` product features are 504-d binary and
  very sparse (about a quarter of products have a single active bit, so many
  products share an identical feature vector); the paper used bag-of-words
  features. Node-level attention can only separate neighbors whose features
  differ.

## Files

| file | role |
|---|---|
| data_loader.py | HGB .dat parser (verbatim from the HGB benchmark) |
| link_data.py | edge split + vectorized negative sampling |
| han_utils.py | BPR loss, chunked ranking scorer, per-edge/per-user/stratified metrics, seeding |
| build_metapath_graphs.py | sparse chain-multiply + top-k pruning per meta-path (seeded random tie-break, .meta.json provenance) |
| check_graph_bias.py | per-id-decile degree audit of built graphs (detects tie-break / hub bias) |
| model_han.py | GAT-per-path + semantic attention (pure torch, custom SpMM backward, gradient checkpointing) |
| train_han.py | training / beta warmup / early stopping / evaluation; `--paper-config` for the paper's Sec. 5.3 hyperparameters |
| train_metapath2vec.py | baseline: meta-path-guided walks + skip-gram (frozen-embedding eval) |
| train_gat.py | baseline: homogeneous GAT on the raw union graph (NO meta-paths) |
| random_paths.py | seeded random valid meta-paths (ablation baseline) |
| eval_emb.py | recompute all test metrics from saved embeddings, no retraining |
| recommend.py | ASIN -> similar products; user -> top-k products |
| METAPATH_CONTRACT.md | what a meta-path discovery repo must output for this repo to accept it |

Saved embedding blobs (`*_emb.pt`) contain P_emb/U_emb plus provenance:
model, dataset, paths, split_seed, the test metrics, and the full run config.
