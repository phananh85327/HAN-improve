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
  `arch.py` key via `--arch`

## Flow

    # 1. materialize each meta-path as a product-product graph (train-split-safe)
    python build_metapath_graphs.py --dataset AMAZON --paths "PPPP,PCPCCP,PPPPUP,PUPCPP,PUPPUP" --topk 20 --split-seed 0

    # 2. train + evaluate (saves output/AMAZON/han_emb.pt)
    python train_han.py --dataset AMAZON --paths "PPPP,PCPCCP,PPPPUP,PUPCPP,PUPPUP" --split-seed 0

    # 2b. baselines on the same split (thesis comparison rows)
    python train_metapath2vec.py --dataset AMAZON --arch amazon_lp_han --split-seed 0
    python train_gat.py --dataset AMAZON --split-seed 0

    # 2c. ablations -- does AUTOMATIC discovery beat manual / random paths?
    #     (same pipeline, different path list; materialize each list in step 1 first)
    python train_han.py --dataset AMAZON --arch amazon_lp_manual --split-seed 0
    python random_paths.py --seed 0     # prints a seeded random valid path list

    # 3. serve queries (works with han_emb.pt, m2v_emb.pt, gat_emb.pt, lmsps_emb.pt)
    python recommend.py --emb output/AMAZON/han_emb.pt --interactive \
        --products-csv <...>/filtered_products.csv --users-csv <...>/filtered_users.csv --exclude-seen

`--split-seed` pins the train/val/test edge split; keep it identical across
steps (and across discovery methods you compare) so results are comparable.

All trainers default to `--user-tower embedding`: the user side is a free
per-user trained vector, matching the per-user capacity metapath2vec/GAT get
implicitly, so the ranking numbers compare ITEM encoders (the meta-path
question) instead of user-representation capacity. Pass `--user-tower feats`
for the original inductive hash-feature tower. Test output also breaks Hit/NDCG
down by user training degree (cold vs heavy users).

## Files

| file | role |
|---|---|
| data_loader.py | HGB .dat parser (verbatim from the HGB benchmark) |
| link_data.py | edge split + vectorized negative sampling |
| han_utils.py | BPR loss, chunked ranking scorer, metrics, seeding |
| build_metapath_graphs.py | sparse chain-multiply + top-k pruning per meta-path |
| model_han.py | GAT-per-path + semantic attention (pure torch, custom SpMM backward) |
| train_han.py | training / early stopping / Hit@K-NDCG@K evaluation |
| train_metapath2vec.py | baseline: meta-path-guided walks + skip-gram (frozen-embedding eval) |
| train_gat.py | baseline: homogeneous GAT on the raw union graph (NO meta-paths) |
| random_paths.py | seeded random valid meta-paths (ablation baseline) |
| recommend.py | ASIN -> similar products; user -> top-k products |
| METAPATH_CONTRACT.md | what a meta-path discovery repo must output for this repo to accept it |
