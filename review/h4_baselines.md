# HAN review part H4: baselines

**Scope:** the comparison rows next to HAN:
- metapath2vec (`train_metapath2vec.py`): the shallow meta-path baseline
- GAT on the union graph (`train_gat.py`): the baseline that ignores node types
- a popularity row, which did not exist

**Bottom line:** metapath2vec is fine. GAT had a must-fix flaw. The popularity row was missing,
and every result needs it as the floor.

**GAT trained on the edge it was asked to predict: must fix, fixed.**
- The GAT's graph holds every training user-product edge, so a training pair (u, p) was scored
  by messages running along that very edge. A test pair never has its edge in the graph.
- On the small sample, its training loss fell to 0.15 while its validation loss was best at
  epoch 3 and then rose: it learned to use the edge.
- Removing each training batch's own edges from the graph it runs on is the standard remedy. It
  raises the GAT's test Hit@10 on the small sample from 0.464 to 0.598 (3 seeds; table below).
- Without the fix, the GAT row in the thesis understates what a type-blind GNN can do, which
  flatters HAN.

**Popularity row: must add, added.**
- `popularity_baseline.py` scores each product by its number of training users, the same for every
  user.
- On your rebuilt data it reaches test Hit@10 **0.42** under the 99-negative protocol. A model
  near that number has learned little more than popularity.
- The result is saved like a model's, so `eval_emb.py` and `summarize_results.py` include it.

## Verdicts

| # | Issue | Evidence | Verdict |
|---|---|---|---|
| B1 | metapath2vec walks | On the sample, every walk follows its template letter by letter, every step is an edge of the graph, and no U-P step is a validation or test edge (also a new test) | OK |
| B2 | metapath2vec training | Skip-gram with negatives drawn within the context node's type (metapath2vec++ style); both directions of each pair; chunked loss scaled to the exact batch mean; best validation snapshot kept | OK |
| B3 | metapath2vec scoring | Dot product of the input embeddings of user and product, as for every model | OK. A user who never appears in a walk keeps a random vector: the script warns when no template contains `U` |
| B4 | GAT target edges | Training positives scored through their own edge (above) | **Must fix: fixed.** Each step removes its positives (both directions) from the graph; validation and test use the full training graph. `--keep-target-edges` gives the old behaviour |
| B5 | Popularity row | Did not exist. Rebuilt data, split 1: Hit@10 0.4233, NDCG@10 0.2490 (99 negatives); full ranking Hit@10 0.0094 | **Must add: added** (`popularity_baseline.py`) |
| B6 | GAT architecture | 2 layers, 8 heads, per-type input projections, no activation on the output layer; the layer is the same `GATLayer` HAN uses, which matches DGL's `GATConv` (H3) | OK |
| B8 | Random path sets | `random_paths.py` drew random schema walks, not the candidates the search chose from; the manual-sets note asks for several draws from the 36 candidates | **Added:** `--from-record` draws N sets of k paths from a search record's candidates (arch.py entries `<prefix>_d1..`) |
| B7 | GAT memory | Each step now builds an edge list without the batch's edges: about 256 MB more on the GPU at full scale (about 8M edges) | Watch it on the first run; `--num-heads 4` is the documented first lever |

## GAT with and without the target edges

Small sample (1,730 products), split 1, hidden 128, 3 run seeds:

| | Test Hit@10 | Test NDCG@10 | Best validation BPR | Best epoch |
|---|---|---|---|---|
| Edges kept (old behaviour) | 0.464 ± 0.003 | 0.259 | 0.479 | 2-3 |
| Each batch's edges removed (new default) | **0.598 ± 0.009** | **0.368** | 0.343 | 67-99 |

For reference on the same sample: HAN with 4 paths 0.40-0.47, popularity 0.41.

With the edges kept, the training loss keeps falling (0.15 by epoch 13) while the validation loss
is best at epoch 2-3: the model fits the edges it can see. With them removed, training and
validation improve together for 70-100 epochs.

**What this means for the thesis.** On the old data the GAT row was 0.79 Hit@10 against HAN's
0.85. A GAT trained without the shortcut may match or beat HAN, which the thesis would then need
to discuss. Part of the difference is the user side, not meta-paths: GAT builds each user from
the user's items, while HAN gives each user a free vector.

## How it was checked

1. **Code:** both trainers in full.
2. **Walks:** `generate_walks` on the sample for `PUP`, `PCCP`, `PUPCP` and `PPUP`.
3. **GAT:** the training curves and test results above; the edge removal on a toy graph (exactly
   the batch's edges, both directions, nothing else).
4. **Tests:** `tests/test_baselines.py`, 4 tests:
   - popularity scores are the training counts, and `eval_emb.py` reproduces them
   - the GAT removes exactly the batch's edges
   - the GAT trains with and without the removal
   - metapath2vec walks follow their templates on training edges only

## What changed

Committed on `claude/han-review`:
- `train_gat.py`: target-edge removal (`make_target_dropper`), `--keep-target-edges`.
- `popularity_baseline.py`: new.
- `random_paths.py`: `--from-record`, `--size`, `--draws`, `--prefix`.
- `tests/test_baselines.py`: new.
