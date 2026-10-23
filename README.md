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
  (built by `preprocessing/dat_generator.py` from the CSVs the C# pipeline in
  `preprocessing/` writes; see `preprocessing/README.md` for the filter chain and
  the known data issues). `data_leafcat/AMAZON/` is the same dataset with
  products linked only to their leaf categories (`--category-link leaf`); run any
  script against it with `--root data_leafcat`.
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

## Run flow (locked protocol, 2026-10-23)

Everything below is driven by `run_all.bat <stage>` (`execute.bat` forwards to it; no
argument = `summarize`). Split seed 1 everywhere (the LMSPS fleets selected their paths on
split 1), graphs read from `data/AMAZON/metapath_graphs_all_unique` via `--graph-dir`, one
blob per run at `output/AMAZON/<model>_<archkey>_s<seed>[_paper].pt`, one log per run under
`logs/Final 5/`. A run whose blob exists is skipped, so the script can be relaunched.

| stage | what | runs | GPU time (12 GB card) |
|---|---|---|---|
| `build` | manual + random ablation paths into the shared graph dir (`--skip-existing`) | 3 | ~1 h |
| `audit` | `check_graph_bias.py`, `check_edition_twins.py` | 2 | 10 min |
| `han` | HAN config of record (`--stage 150 --patience 10 --beta-warmup 15`), seed 1, 13 shortlist keys + `amazon_lp_manual` + `amazon_lp_random4_s0` + `amazon_lp_random6_s0` | 16 | ~16 h |
| `m2v` | metapath2vec fast sweep (`--epochs 10 --evals-per-epoch 4 --patience 6`), same 16 keys | 16 | ~8 h |
| `gat` | GAT on the union graph (arch-independent), seeds 1 2 3 | 3 | ~3 h |
| `seeds` | HAN seeds 2 3 on the headline keys `amazon_lp_f4_c`, `amazon_lp_f6_s1`, `amazon_lp_manual` (+ extra keys passed as arguments) | 6+ | ~10 h |
| `m2v_paper` | metapath2vec `--paper-walks` (walk length 100, 10 walks/node, window 5), seeds 1 2 3, headline keys | 9 | ~60 h |
| `han_paper` | HAN `--paper-config` (Wang et al. Sec. 5.3 hyperparameters), seed 1, headline keys | 3 | ~8 h |
| `summarize` | `summarize_results.py --group --csv "logs/Final 5/results.csv"` | 1 | minutes |

`run_all.bat all` runs them in that order (the 60-hour `m2v_paper` stage is second to last, so
every other table can be drafted after ~45 h); `run_all.bat dry` prints every command.

How to read the result table (`summarize_results.py`):
- **Choose the best arch by validation BPR**, never by a test metric. Then pass that key to
  `run_all.bat seeds <key>` if it is not already a headline key.
- Rows: *discovered* (13 shortlist sets from the LMSPS fleet, 4 or 6 paths), *manual* (5 hand-picked
  paths), *random* (schema-valid random paths at the same budgets) -- the discovery question is
  whether the discovered rows beat manual and random at equal path budget; GAT is the
  heterogeneity-blind floor; metapath2vec the shallow meta-path baseline.
- Report `hit10_notwin` next to `hit10`: the duplicate-edition shortcut (section "Metrics") lifts
  every model by about 7-8 points and the non-twin column is the recommendation-quality number.
- `han_paper` / `m2v_paper` rows are fidelity rows (paper hyperparameters / paper walk corpus), not
  the config of record; keep them in a separate table.

Legacy: `logs/Final 1-4` and the `han_emb_*`, `m2v_emb_*`, `Old 2/` blobs predate the tie-break fix
and use retired arch keys; they are not comparable with this flow.



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
Plus a Hit/NDCG breakdown by user training degree (cold vs heavy users), and a
breakdown by **duplicate-edition twin**: Amazon shares one review pool across the
editions of a title and the SNAP dump copies it into every edition, so 46 % of
products have a reviewer set identical to another product's and 55 % of U-P edges
sit on such products. A test edge whose user already has a TRAIN edge to a twin of
the test item is predictable from one P-U-P hop; 50 % of test edges are in that
situation by exact identity (61 % at Jaccard >= 0.99, 73 % at >= 0.9; split 1).
Every trainer and `eval_emb.py` therefore report Hit@10 / NDCG@10 separately for
"twin in train" and "no twin in train" edges
(`link_data.product_twin_groups`, `han_utils.report_twin_breakdown`;
`check_edition_twins.py` prints the dataset-level numbers). Read the non-twin row
as the number closest to recommendation quality rather than edition matching
(saved HAN runs: ~0.95 Hit@10 on twin edges vs ~0.80 on the rest; metapath2vec
~0.87 vs ~0.70).

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
| check_edition_twins.py | duplicate-edition audit: identical reviewer sets, share of test edges with a train twin |
| preprocessing/ | raw SNAP -> CSV (C#) -> HGB .dat (`dat_generator.py`); README documents filters and known issues |
| model_han.py | GAT-per-path + semantic attention (pure torch, custom SpMM backward, gradient checkpointing) |
| train_han.py | training / beta warmup / early stopping / evaluation; `--graph-dir` shared graph folder; `--paper-config` for the paper's Sec. 5.3 hyperparameters |
| train_metapath2vec.py | baseline: meta-path-guided walks + skip-gram (frozen-embedding eval); `--paper-walks` for the HAN paper's walk settings |
| train_gat.py | baseline: homogeneous GAT on the raw union graph (NO meta-paths); no ELU on the output layer |
| random_paths.py | seeded random valid meta-paths (ablation baseline; the `amazon_lp_random*` keys in arch.py) |
| eval_emb.py | recompute all test metrics from saved embeddings, no retraining |
| summarize_results.py | one table over all saved blobs (val BPR, Hit/NDCG, twin / non-twin), mean +/- std over seeds |
| run_all.bat / execute.bat | the complete experiment flow by stage (see "Run flow") |
| recommend.py | ASIN -> similar products; user -> top-k products |
| METAPATH_CONTRACT.md | what a meta-path discovery repo must output for this repo to accept it |

Saved embedding blobs (`*_emb.pt`, `<model>_<arch>_s<seed>.pt`) contain P_emb/U_emb plus
provenance: model, dataset, paths, split_seed, the test metrics (`metrics`, `metrics_twin`),
`best_val_loss` / `best_epoch` / `epochs_run` / `train_time_s`, and the full run config.
