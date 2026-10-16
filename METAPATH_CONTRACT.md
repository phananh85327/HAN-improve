# Meta-path output contract

What an automatic meta-path discovery repo must produce for this repo (HAN and the
Metapath2vec baseline) to accept its output.

This exists so that a second discovery method can be swapped in beside LMSPS and
compared fairly. The discovery side and the model side communicate through
**one artifact only: a short ordered list of meta-path strings**. Nothing else
crosses the boundary -- no tensors, no checkpoints, no graph files.

---

## 1. The artifact

A list of strings, e.g.

```
['PPPP', 'PCPCCP', 'PPPPUP', 'PUPCPP', 'PUPPUP']
```

Each string is a sequence of **node-type letters** describing a walk through the
schema. `PUP` means product -> user -> product. That is the entire interface.

LMSPS already prints exactly this at the end of `train_search.py`:

```
paths [['PPPP', 'PCPCCP', 'PPPPUP', 'PUPCPP', 'PUPPUP'], []]
```

(the second, empty list is LMSPS's *label*-propagation paths; this repo ignores
them, so a new method does not need to produce anything analogous.)

---

## 2. Hard requirements

A path string is rejected -- loudly, before any training starts -- unless all of
these hold. The checks live in `build_metapath_graphs.py:145` and
`train_han.py:76`.

### R1. Letters come from the dataset's type alphabet

For AMAZON the alphabet is fixed in `han_utils.py::NODE_TYPE_IDS`:

| letter | node type | HGB type id | count | feature dim |
|---|---|---|---|---|
| `P` | Product | 0 | 90,659 | 504 |
| `C` | Category | 1 | 39,754 | 128 |
| `U` | User | 2 | 171,285 | 64 |

Any other character is an error. Lowercase is *not* normalized -- emit uppercase.

### R2. The path starts and ends at the target type

For the `U,P` link-prediction task the target type is `P`, so every path must
match `P...P`. This is not a style preference: HAN's node-level attention needs
a **product-product neighbor graph**, which only a P-to-P walk can define. A path
like `PUPU` or `PPC` cannot be materialized and is refused with

```
path "PPC" does not start and end at target type P -- not usable as a HAN
neighbor graph. Re-run the search with --tgt-ended-paths.
```

If your discovery method naturally emits paths ending anywhere (LMSPS does), give
it a filter equivalent to LMSPS's `--tgt-ended-paths` flag: keep only
`len(p) >= 2 and p[0] == tgt and p[-1] == tgt`. Filtering *during* the search is
much better than filtering afterwards -- otherwise the method spends its budget
scoring candidates that will be thrown away.

### R3. Minimum length 2

`'P'` alone is not a path. `'PP'` is the shortest legal one.

### R4. Every adjacent letter pair must be a real relation

The step matrices available are built in
`build_metapath_graphs.py::build_base_matrices` and are keyed by letter pair:

```
PP   CC   PC   CP   UP   PU
```

So the AMAZON schema graph is:

```
P -> P, C, U
C -> P, C
U -> P          (users connect to nothing but products)
```

`UU`, `UC` and `CU` do not exist. A path containing one dies with:

```
path PUUP: no base relation for step UU (available: ['CC', 'CP', 'PC', 'PP', 'PU', 'UP'])
```

The practical consequence: **`U` may never appear twice in a row, and `U` must
always be surrounded by `P`.** Every user hop is a `PUP` sandwich.

### R5. Paths must be unique *after* symmetrization

`materialize_path` finishes with `out = out.maximum(out.T)`. That means a path
and its reverse produce the same graph. `PUPCPP` and `PPCPUP` are **duplicates**
-- shipping both wastes an entire HAN attention channel on a redundant view.
Deduplicate with a canonical form, e.g. `min(p, p[::-1])`.

Palindromes (`PPPP`, `PUPUP`, `PCP`) are already self-symmetric and are fine.

---

## 3. Soft requirements -- legal, but they will hurt

### S1. Length budget

A path of `L` letters is `L-1` chain multiplications. Discovery methods usually
expose this as a hop budget; LMSPS's `--num-hops 5` yields strings up to 6 letters.
Beyond ~6 the materialized graph approaches all-pairs density, top-k pruning
starts discarding most of the signal, and the path stops being interpretable in
the thesis write-up. **Stay in the 2-6 letter range.**

### S2. Path count

3-6 paths is the sweet spot. HAN allocates one full GAT per path and holds all of
them on the GPU simultaneously, so **VRAM grows linearly with the number of
paths**. The 5-path run peaks comfortably inside 12 GB; 10 paths would not.
If your method wants to output more, rank them and truncate.

### S3. Do not emit dead paths

After materialization each path reports its isolated-node count:

```
PPPP:   90659 nodes, 1698430 edges, mean degree 18.7, 10720 isolated nodes
PCPCCP: 90659 nodes, 3383822 edges, mean degree 37.3, 0 isolated nodes
PUPPUP: 90659 nodes, 3426918 edges, mean degree 37.8, 49 isolated nodes
```

A path where most products are isolated contributes almost nothing but still
costs a full GAT. Roughly: under ~15% isolated is healthy, over ~50% means the
path is too rare to be worth a channel.

### S4. Order is part of the artifact

The list order fixes the order of the reported semantic-attention weights `beta`
and the column order inside the model. Emit it deterministically (sorted, or by
descending discovery score) so two runs of your method are comparable.

---

## 4. Fairness requirements (these decide whether the comparison is publishable)

The point of a second discovery method is a like-for-like comparison. Three
things must match, or the numbers mean nothing:

### F1. Same data

Read the same `data/AMAZON/` HGB `.dat` files. Do not regenerate or re-filter them.

### F2. Same split seed

Everything downstream is pinned by `--split-seed 0`. If your discovery method
looks at edges at all (LMSPS does -- it trains on them), it must build its split
with the same `build_link_split(dl, src_type, dst_type, seed=<same seed>)`.

### F3. Same leakage rule

**The discovery method must never see a validation or test U-P edge.** In this
repo the rule is implemented by replacing the U-P adjacency with
`link_split['train_adj_src2dst']` *before* any propagation:

```python
link_split = build_link_split(dl, type_ids['U'], type_ids['P'], seed=args.split_seed)
UP = link_split['train_adj_src2dst']     # train edges only
PU = UP.T.tocsr()
```

A method that scores candidate paths using the full U-P matrix will
systematically prefer user-heavy paths for the wrong reason and beat LMSPS on a
technicality. If your method only inspects the *schema* (type-level structure)
and never the edges, this requirement is vacuous -- state that in the thesis, it
is a genuine methodological difference worth noting.

---

## 5. How to hand the list over

Two supported channels; both `train_han.py` and `train_metapath2vec.py` accept
either.

**A. Command line** -- good for a one-off:

```
--paths "PPPP,PCPCCP,PPPPUP,PUPCPP,PUPPUP"
```

Whitespace around commas is stripped; empty entries are dropped.

**B. `arch.py` entry** -- good for anything you want to re-run. Add a key whose
value is a two-element list `[paths, label_paths]`, with the second element left
empty:

```python
archs = {
    ...
    "amazon_lp_newmethod": [
        ['PPPP', 'PCPCCP', 'PPPPUP', 'PUPCPP', 'PUPPUP'], []],
}
```

then pass `--arch amazon_lp_newmethod`. The nested-list shape is inherited from
LMSPS; keep it even though the second slot is unused.

---

## 6. Downstream, for reference

Once the list is accepted, this is what happens to it -- a new discovery repo does
**not** need to implement any of this:

1. `build_metapath_graphs.py` chain-multiplies the step matrices in row chunks,
   prunes to `--topk 20` neighbors per row after each step (walk-count ties are
   broken by a seeded random key, `--tie-break random`; the legacy `id` mode
   favored low product ids), binarizes, strips self-loops, symmetrizes, and writes
   `data/AMAZON/metapath_graphs/<path>_top20_split0.npz` plus a
   `<path>_top20_split0.meta.json` sidecar recording the tie-break mode.
2. `train_han.py` loads one graph per path, appends self-loops, runs one GAT per
   path and fuses them with semantic attention, trains with BPR, and evaluates
   with 99 sampled negatives -> Precision/Recall/NDCG/Hit@K.
3. `train_metapath2vec.py` reuses the **same strings** as random-walk templates
   (`PUP` repeated gives P-U-P-U-...), which is why R2 also applies to it -- a
   template must cycle cleanly back to the target type.
4. `train_gat.py` ignores the list entirely. That is the point of that baseline.

`--topk` and `--split-seed` must be identical between steps 1 and 2, or
`train_han.py` will not find the `.npz` files.

---

## 7. Validation checklist for a new discovery repo

Run these before trusting a comparison:

1. **Schema check** -- every path matches `^P[PCU]*P$`, has no `UU`/`UC`/`CU`
   substring, and no duplicate under `min(p, p[::-1])`.
2. **Materialize** -- `build_metapath_graphs.py --paths "<yours>" --topk 20
   --split-seed 0` completes and reports a plausible mean degree (roughly 10-40)
   and a low isolated-node count for most paths.
3. **Leakage** -- rebuild your discovery input with the *full* U-P adjacency in a
   throwaway run and confirm the edge count differs from the train-only one. If
   it does not, your method was reading the full matrix.
4. **Train** -- `train_han.py --paths "<yours>" --split-seed 0 --stage 2` runs end
   to end and prints metrics.
5. **Sanity** -- Hit@10 must land well above the 0.10 random-guess floor (1 positive
   among 100 candidates). A value near 0.10 is a bug, not a weak method -- almost
   always a global-vs-local node-id mismatch, since eval edges are local per type.

Reference numbers on AMAZON (per-edge Hit@10 / NDCG@10, 99 sampled negatives,
`--user-tower embedding`, split-seed 1, run-seed 1; logs in `logs/Final 1..4`).
These graphs were built with the legacy `--tie-break id`; numbers with the
random tie-break are not yet available.

| model | paths | Hit@10 | NDCG@10 |
|---|---|---|---|
| HAN | LMSPS-discovered sets `amazon_lp_auto_1..16` (4 paths each) | 0.848 - 0.867 | 0.689 - 0.719 |
| HAN | `amazon_lp_manual_1` = PP, PUP, PCP, PPP | 0.874 | 0.728 |
| HAN | `amazon_lp_manual_5` = PPP, PPPP, PPUP, PUPP | 0.875 | 0.733 |
| metapath2vec | `amazon_lp_auto_1..16` | 0.742 - 0.816 | 0.546 - 0.643 |
| metapath2vec | `amazon_lp_manual_1` | 0.785 | -- |
| random | any | ~0.10 | -- |

(An older protocol -- `--user-tower feats`, split-seed 0 -- gave HAN 0.5610 /
0.3472 and the LMSPS retrain 0.5756 / 0.3591; those are not comparable to the
rows above.)

A new discovery method is doing its job if HAN trained on *its* paths lands in
the same neighborhood; the interesting result is which paths it finds and how
the semantic-attention weights redistribute.

---

## 8. Adding a dataset other than AMAZON

Two edits, both small:

- `han_utils.py::NODE_TYPE_IDS` -- add the letter -> HGB type-id map.
- `build_metapath_graphs.py::build_base_matrices` -- add a branch mapping that
  dataset's relation ids to letter-pair keys. AMAZON's mapping is:

  | relation id | letter step |
  |---|---|
  | 0, 1 | `PP` (also-bought, both directions) |
  | 2, 3 | `CC` (category hierarchy, both directions) |
  | 4 | `PC` |
  | 5 | `CP` |
  | 6 | `UP` -- **overridden by the train-only split** |
  | 7 | `PU` -- **overridden by the train-only split** |

  If your discovery method emits relation-id sequences rather than letters, use
  this table to convert before handing the list over.

Everything else -- HAN, the baselines, `recommend.py` -- is dataset-agnostic.