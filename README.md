# han-recsys: HAN link-prediction recommender for auto-discovered meta-paths

Standalone PyTorch reimplementation of HAN (WWW-2019) for product recommendation
on the Amazon-Meta HGB dataset. Consumes meta-paths discovered by ANY automatic
discovery method (LMSPS or otherwise): a discovery method only has to output a
list of path strings that start and end at the target type, e.g. "PUP,PPPP".
The exact requirements a discovery repo must satisfy -- schema, target-type,
leakage and fairness rules -- are in [METAPATH_CONTRACT.md](METAPATH_CONTRACT.md).

Dependencies: torch (+CUDA), numpy, scipy, tqdm. No dgl / torch_sparse (dgl only for
one optional test).

## Inputs

- `data/AMAZON/`: the HGB-format data (`node.dat`, `link.dat`, `label.dat`, `label.dat.test`,
  `meta.json`). Use the rebuilt data from the LMSPS repo (edition duplicates merged, placeholder
  customers dropped; LMSPS review part 1): copy those five files into `data/AMAZON/` and delete
  graph folders built from older data. Every script prints which data it uses and the
  fingerprint of its training edges, and stops on data without `meta.json` (the data before the
  rebuild; `--allow-legacy-data` to use it on purpose). `python check_handover.py <LMSPS search
  record>` confirms that the data is the data the search ran on.
- the meta-path sets, as `arch.py` keys (`--arch`) or a comma-separated list (`--paths`):
  - the LMSPS Fleet v3 sets `amazon_lp_v3_f4_a1`, `_a2`, `_c` and `amazon_lp_v3_f6_a1`, `_a2`,
    `_c`, from the LMSPS repo's `hgb/logs/Fleet v3/final_ns4|final_ns6/reselect_d010.txt` (per
    size, the first two averaged one-SE sets and the vote's consensus; LMSPS review part 4)
  - the manual sets `amazon_lp_manual_1..8` (M1-M8 of the LMSPS repo's
    `review/manual_metapath_sets.md`)
  - the random draws `amazon_lp_rand4_d1..5` / `amazon_lp_rand6_d1..5`
  - older keys (`amazon_lp_f4_*`, `amazon_lp_f6_*`, `amazon_lp_han*`, `amazon_lp_manual`) were
    searched on the data before the rebuild

## Run flow (the rerun on the rebuilt data)

`run_all.bat <stage>` runs the whole HAN-side experiment. Every run is one model trained on one
meta-path set with one seed. It writes `output/AMAZON/v3/<model>_<key>_s<seed>.pt` and the log
`output/AMAZON/v3/logs/<same name>.txt`.

- A run whose result exists is skipped, so the script can be stopped and started again.
- Any stage can be run alone.
- `run_all.bat dry` prints every command, so single runs can also be started by hand.

The design, the reasons and the cost are in `review/h6_run_protocol.md`.

First time:
1. Copy the rebuilt data from the LMSPS repo's `data/AMAZON` (`node.dat`, `link.dat`,
   `label.dat`, `label.dat.test`, `meta.json`) into this repo's `data/AMAZON`, and move the old
   `metapath_graphs*` folders out of it.
2. Check `SEARCH_RECORD` and `CONDA_ROOT` at the top of `run_all.bat` (the record path assumes the
   two repos sit side by side).
3. `run_all.bat dry`, then `run_all.bat check`: it must print SAME.

| stage | what |
|---|---|
| `check` | `check_handover.py` against a Fleet v3 search record (`SEARCH_RECORD` at the top of the file) |
| `build` | the graphs of every set into `data/AMAZON/metapath_graphs_v3` (graphs from other data are rebuilt) |
| `audit` | `check_graph_bias.py`, `check_edition_twins.py` |
| `smoke` | a few epochs of HAN on the largest set (`amazon_lp_rand6_d2`, 9.3M graph edges) and of GAT: GPU memory check, nothing kept |
| `popularity` | the popularity floor |
| `han` | HAN, seed 1, on the Fleet v3, manual and random sets |
| `gat` | GAT on the union graph, seed 1 |
| `m2v` | metapath2vec, seed 1, on the same sets |
| `seeds` | seeds 2 and 3 for the main comparison (HAN and metapath2vec on `amazon_lp_v3_f4_a1`, `amazon_lp_v3_f6_a1`, `amazon_lp_manual_5`, `amazon_lp_manual_6`) and for GAT |
| `summarize` | `summarize_results.py --group --full-ranking`, also saved as `output/AMAZON/v3/results.csv` |
| `push` | commit the logs and the table (results `.pt` files stay local) and push |

In all: 24 sets (6 LMSPS, 8 manual, 10 random) for HAN and metapath2vec at seed 1, the 4
headline sets at seeds 2 and 3, GAT at seeds 1-3 and popularity; about 20-25 GPU hours on a
12 GB card.

How to read the results table:
- **Main comparison:** LMSPS against manual at the same number of paths, `amazon_lp_v3_f4_a1`
  against M5 and `amazon_lp_v3_f6_a1` against M6, 3 seeds each.
  - `_a2` and `_c` show whether the conclusion depends on which LMSPS set stands for the search:
    the second simplest one-SE set, and the set most runs agreed on.
  - The LMSPS sets' category paths (`PCCP`, `PCCPP`, `PPCCPP`, ...) reach no product for about
    80% of products (books and music; LMSPS review part 4). In HAN those products see only
    themselves in that channel. A path's weight averages a score over all products (the
    paper's eq. 7), so a category path's weight mostly measures that self-only channel, not
    the category neighbours of DVDs and videos: don't read it as the value of the path.
  - The random draws ask whether the search picks better than chance from its own 36
    candidates. Report their spread, never a single draw.
  - Popularity is the floor every model must clear.
  - GAT is the type-blind GNN; metapath2vec the shallow meta-path baseline.
- **Choose by validation BPR,** never by a test number, if you pick a "best" set among several.
- **Compare LMSPS with manual and random sets on test numbers.** The search picked its
  sets on the same validation edges, so their validation numbers lean in their favour.
- **Read the `data` and `variant` columns:** the table keeps different data versions and different
  settings apart, and warns when it mixes data versions.
- **Every run should end by early stopping.** `summarize` warns about a run that used its whole
  epoch cap: rerun it with a larger cap (GAT seed 2 reached 150 and was rerun with `--stage 300`;
  it stopped at 157). The cap is not a setting, so such a rerun stays in its model's row; move
  the capped result out of `output/AMAZON/v3`, or the table warns that a seed is counted twice.
- **Full ranking:** the full-ranking columns rank against every product instead of 99 random ones.
  Sampled numbers flatter popular items (popularity: 0.42 sampled, 0.009 full).
- **The rerun's results** and how to report them: `review/h8_results.md`.

Earlier results (`logs/Final 1-5`, `output/AMAZON/Han_*`, `M2v_*`, `gat_s1.txt`) were trained on
the data before the rebuild and with the old GAT and HAN initialisation. They are not comparable
with the rerun.

## Running one model by hand

    python check_handover.py "../LMSPS/hgb/logs/Fleet v3/final_ns4/search_ns4_seed1.json"
    python build_metapath_graphs.py --arch amazon_lp_manual_5 --graph-dir data/AMAZON/metapath_graphs_v3 --skip-existing
    python train_han.py --arch amazon_lp_manual_5 --graph-dir data/AMAZON/metapath_graphs_v3 \
        --stage 150 --patience 10 --beta-warmup 15 --seed 1 --save-emb output/AMAZON/v3/han_amazon_lp_manual_5_s1.pt
    python train_metapath2vec.py --arch amazon_lp_manual_5 --epochs 10 --evals-per-epoch 4 --patience 6 \
        --seed 1 --save-emb output/AMAZON/v3/m2v_amazon_lp_manual_5_s1.pt
    python train_gat.py --stage 150 --patience 10 --seed 1 --save-emb output/AMAZON/v3/gat_union_s1.pt
    python popularity_baseline.py --save-emb output/AMAZON/v3/popularity_s1.pt
    python summarize_results.py "output/AMAZON/v3/*.pt" --group --full-ranking
    python eval_emb.py --emb output/AMAZON/v3/han_amazon_lp_manual_5_s1.pt --full-ranking
    python recommend.py --emb output/AMAZON/v3/han_amazon_lp_manual_5_s1.pt --interactive \
        --products-csv <...>/filtered_products.csv --users-csv <...>/filtered_users.csv --exclude-seen

A result saved under the name `run_all.bat` would give it counts as done for the batch file.

Settings worth knowing:
- `--split-seed` (default 1, the split the LMSPS fleets used) fixes WHICH edges are train/val/test
  and the 99 evaluation negatives per test edge -- identical for every model, so every row takes
  the same exam. `--seed` is the RUN seed (initialisation, dropout, batches, BPR negatives, walks).
- `--beta-warmup 15` is required for HAN: without it the semantic attention puts all its weight
  on one path within the first epochs (`review/h3_han_model_and_training.md`).
- Early stopping (all trainers): the best-validation checkpoint is always kept; an epoch only
  resets patience if it improves validation BPR by more than `--min-delta` (default 5e-4).
  train_metapath2vec also takes `--evals-per-epoch` (its patience counts evaluations), because
  skip-gram peaks against the ranking objective inside the first epochs.
- The default `--save-emb auto` writes the same file every run: always pass a per-run name.
- `--vram-fraction` (default 0.9) caps the CUDA allocator below total VRAM so overshoot raises a
  clean OOM instead of Windows paging GPU memory to system RAM or the driver crashing.
- `python -m pytest tests` runs the repo's tests (about a minute on a CPU).

## Metrics

Each trainer (and eval_emb.py) reports:
- **per-edge**: each test edge's positive ranked against its own 99 sampled negatives. With one
  positive per ranking, Recall@k == Hit@k and Precision@k = Hit@k / k by construction.
- **per-user**: all of a user's test positives ranked together against the deduplicated union of
  that user's negatives, macro-averaged over users; the four metrics genuinely differ. Not
  comparable to per-edge numbers; report separately.
- a Hit/NDCG breakdown by user training degree (cold vs heavy users)
- a breakdown by **duplicate-edition twin**: test edges whose user has a training edge to another
  edition of the test item. On the data before the rebuild that was half of all test edges; the
  rebuilt data merges editions, which leaves 40 of 81,984 test edges, so the column is nearly
  empty now (`link_data.product_twin_groups`, `check_edition_twins.py`)
- with `--full-ranking` (`eval_emb.py`, `summarize_results.py`): the same test edges ranked
  against every product the user has no observed edge to (`han_utils.evaluate_full_ranking`)

All trainers default to `--user-tower embedding`: the user side is a free per-user trained
vector. GAT builds its users from their items instead, which is part of any difference between
GAT and HAN. Pass `--user-tower feats` for the inductive hash-feature tower.

## Deviations from the paper (Wang et al., WWW-2019)

The HAN model itself (`model_han.py`) follows the paper: per-path multi-head
node-level attention (eq. 1-5), semantic-level attention with node-averaged
scores and a 128-d attention vector (eq. 7-9). The task adaptation and the
pipeline around it differ; keep these in mind when reading results. The check
against the paper's text is in `review/h3_han_model_and_training.md` (an earlier
review is in the local notes `chat/2026-10-16_han-paper-fidelity-review.md`).

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
- **Initialisation.** Every path's GAT starts from the same weights (the paper does
  not say; the reference code draws each one separately). With separate random starts
  the path weights beta followed each path's position in the list, not the path, and
  reversing the list changed the result (`review/h3_han_model_and_training.md`).
  `--independent-init` restores separate starts.
- **Hyperparameters.** Repo defaults are lr 1e-3, no weight decay, dropout 0.2,
  hidden 512 (8 heads x 64), patience 20 with `--min-delta 5e-4`. The paper
  (Sec. 5.3) uses lr 5e-3, L2 1e-3, dropout 0.6, 64-d embeddings, patience 100:
  `--paper-config` applies those.
- **Baselines.** The paper runs GAT and metapath2vec on each single meta-path and
  reports the best one (Sec. 5.2). Here GAT runs on the union of all relations, with
  no meta-paths (`train_gat.py`), and metapath2vec walks all the paths of a set
  together; the paper's "best single path" rows are not run.
- **Product features.** In the rebuilt data (LMSPS review part 1) a product's
  `node.dat` features are 260-d: a 256-d LSA projection of the title plus 4 product
  group bits. The data before the rebuild had 504-d binary, very sparse features.
  The paper used bag-of-words features.

## Files

| file | role |
|---|---|
| data_loader.py | HGB .dat parser (verbatim from the HGB benchmark) |
| link_data.py | edge split + vectorized negative sampling; the training-edge fingerprint and data check every run prints and saves |
| check_handover.py | is this repo's data the data an LMSPS search ran on? (compares with a search record) |
| han_utils.py | BPR loss, chunked ranking scorer, per-edge/per-user/stratified metrics, seeding |
| build_metapath_graphs.py | sparse chain-multiply + top-k pruning per meta-path (seeded random tie-break, .meta.json provenance) |
| check_graph_bias.py | per-id-decile degree audit of built graphs (detects tie-break / hub bias) |
| check_edition_twins.py | duplicate-edition audit: identical reviewer sets, share of test edges with a train twin |
| preprocessing/ | raw SNAP -> CSV (C#) -> HGB .dat (`dat_generator.py`); README documents filters and known issues |
| model_han.py | GAT-per-path + semantic attention (pure torch, custom SpMM backward, gradient checkpointing) |
| train_han.py | training / beta warmup / early stopping / evaluation; `--graph-dir` shared graph folder; `--paper-config` for the paper's Sec. 5.3 hyperparameters |
| train_metapath2vec.py | baseline: meta-path-guided walks + skip-gram (frozen-embedding eval); `--paper-walks` for the HAN paper's walk settings |
| train_gat.py | baseline: homogeneous GAT on the raw union graph (NO meta-paths); no ELU on the output layer; each training step removes its own positive edges from the graph (`--keep-target-edges` = old behaviour) |
| popularity_baseline.py | baseline: products scored by their number of training users; the floor for every model |
| random_paths.py | seeded random meta-paths (ablation baseline): schema walks, or `--from-record` draws from an LMSPS search's own candidates |
| eval_emb.py | recompute all test metrics from saved embeddings, no retraining; skips results from other data; `--full-ranking` |
| summarize_results.py | one table over all saved blobs (data version, settings variant, val BPR, Hit/NDCG, twin / non-twin), mean +/- std over seeds; `--full-ranking` |
| tests/ | `python -m pytest tests` (about a minute on a CPU) |
| review/ | the HAN repo review (parts H1-H7; H8, the results, follows the rerun) |
| run_all.bat | the complete experiment flow by stage (see "Run flow") |
| recommend.py | ASIN -> similar products; user -> top-k products |
| METAPATH_CONTRACT.md | what a meta-path discovery repo must output for this repo to accept it |

Saved embedding blobs (`*_emb.pt`, `<model>_<arch>_s<seed>.pt`) contain P_emb/U_emb plus
provenance: model, dataset, paths, split_seed, the test metrics (`metrics`, `metrics_twin`),
`best_val_loss` / `best_epoch` / `epochs_run` / `train_time_s`, the data record (`data`:
meta.json, node counts, training-edge fingerprint), HAN's graph sidecars (`graphs`), and the
full run config.
