# HAN review part H2: meta-path graph construction

**Scope:** `build_metapath_graphs.py` and `check_graph_bias.py`, which turn a meta-path string
into the product-product graph HAN's node-level attention runs on:
- the step relations
- the top-20 pruning and its tie-break
- binarising, self-loops and symmetrisation

What the pruning does to the paths the search favours is measured on the full rebuilt data:
the structure from your pushed `link.dat`, split 1. Product features are not needed for this.

**Bottom line:** no must-fix. The code builds what it says it builds. Two things are worth knowing
when you read HAN's results:

1. **For long paths, "top-20 neighbours" is a choice, not a fact.**
   - On the full data, a path with two user hops (`PUPUP`, `PUPPUP`, `PPUPUP`, `PCPUP`) reaches
     86-97% of all products.
   - Any 20 of them is a sample of an almost complete neighbourhood, and three sensible ways to
     pick the 20 share only 10-20% of their members.
   - The builder's way, the top 20 after every hop, is a reasonable one. The obvious alternative,
     the exact top 20 by walk count, is worse: it gives most products the same popular neighbours.
2. **Ties are common, but they barely move HAN's results.** Even for `PUP` (shared reviewers),
   two different tie-break seeds keep different neighbours. On the small sample, HAN's Hit@10
   varied about five times more across run seeds than across tie-break seeds, so building the graphs
   once per split is fine.

## Verdicts

| # | Issue | Evidence | Verdict |
|---|---|---|---|
| G1 | Step relations | `PP` and `CC` are the plain relations (both directions, no self-loops), `PC`/`CP` the leaf-category links, `UP`/`PU` the training edges only. Same semantics as the search since LMSPS part 9 | OK |
| G2 | Top-k with ties | Value first, then a seeded random key; rows visited in a fixed order, so the same inputs give the same graph | OK |
| G3 | Binarise, strip self-loops, symmetrise | As documented; a path and its reverse give the same graph (the search's `--dedupe-reversed` treats them as one, too) | OK |
| G4 | Pruning after every hop, instead of the paper's "all meta-path neighbours" | Table 1: for one intermediate node (`PUP`, `PCP`, `PPUP`) it is the exact top 20. For two user hops the paper's neighbour set is nearly every product, so some pruning is unavoidable | Keep. Document the definition in the thesis (H7) |
| G5 | Which 20, for long paths | Table 2: the exact top 20 by walk count collapses onto popular products (for `PUPPUP` the 100 most frequent products fill 77% of all neighbour slots). The builder's per-hop beam stays as diverse as the top 20 by random-walk probability, which is the weighting behind the search's mean-aggregated features | Keep the builder's way |
| G6 | Hub products | After symmetrisation the maximum degree is 3,559 (`PUP`) to 7,682 (`PUPPUP`); the top 1% of products hold 10-19% of edge ends | Not a bug: popular products are popular. Document (H7) |
| G7 | Dead paths | `PCCPUP`: 76% of products isolated (leaf categories: `PCCP` is nearly empty). `PP` 11%, `PPP` 16%, `PPPUPP` 11% | `PCCPUP` was in two Fleet v2 sets. If the rerun picks it again, HAN gets a mostly empty channel there; it costs a GAT but cannot mislead |
| G8 | Build time | Full data: 1-56 s per path, a few minutes for all 14 paths measured | OK |
| G9 | Tie-break seed | The graphs are built once per split, with the tie seed equal to the split seed, so run seeds never vary them. Small sample, manual set 5, 3 tie seeds x 3 run seeds: the means per tie seed differ by SD 0.005 Hit@10, the means per run seed by SD 0.024 | OK: one graph build per split |

## Table 1: graphs on the full rebuilt data (split 1, top-20)

| Path | Mean / median / max degree | Isolated | Top 1% of products hold | Exact reach (median products) | Builder's neighbours inside the exact top 20 |
|---|---|---|---|---|---|
| `PP` | 3.6 / 3 / 275 | 11.3% | 5.4% of edge ends | (1 hop) | |
| `PUP` | 34.6 / 25 / 3,559 | 0.0% | 10.6% | | exact (one pruning) |
| `PCP` | 32.9 / 27 / 430 | 0.0% | 4.2% | | exact |
| `PPP` | 10.3 / 8 / 244 | 16.2% | 5.2% | | exact |
| `PPUP` | 31.9 / 23 / 3,786 | 2.4% | 12.1% | | exact |
| `PCCCP` | 37.3 / 24 / 1,535 | 0.0% | 12.6% | 2,448 | 69% |
| `PCPCP` | 35.9 / 21 / 1,744 | 0.0% | 11.1% | 22,352 | 39% |
| `PPPPUP` | 31.8 / 22 / 4,142 | 4.3% | 13.1% | 5,860 | 72% |
| `PPPUPP` | 28.7 / 22 / 1,332 | 11.3% | 10.2% | 8,894 | 60% |
| `PUPUP` | 35.9 / 21 / 7,057 | 0.0% | 17.4% | 42,112 | 33% |
| `PCPUP` | 37.4 / 21 / 5,338 | 0.0% | 15.5% | 38,862 | 22% |
| `PUPPUP` | 37.0 / 21 / 7,682 | 0.0% | 18.9% | 46,499 | 19% |
| `PPUPUP` | 33.2 / 22 / 6,532 | 5.4% | 17.7% | 47,434 | 24% |
| `PCCPUP` | 7.5 / 0 / 2,708 | 76.2% | 38.8% | 26,690 | 26% |

"Inside the exact top 20" counts ties: a kept neighbour whose walk count is at least the row's
20th largest. Measured on 400 sampled products. The full data has 48,883 products.

## Table 2: how diverse the neighbour lists are

Over the 400 sampled products: how many distinct products appear in their 20-neighbour lists,
and the share of all list slots taken by the 100 most frequent ones (lower = more specific).

| Path | Builder (top 20 after every hop) | Exact top 20 by walk count | Exact top 20 by random-walk probability |
|---|---|---|---|
| `PUP` | 5,802 distinct, 8.7% | 6,020, 8.7% | 6,572, 5.7% |
| `PCPCP` | 5,438, 8.3% | 2,732, 22.9% | 5,794, 7.8% |
| `PUPUP` | 4,596, 15.1% | 1,500, **60.1%** | 5,934, 10.7% |
| `PCPUP` | 4,786, 12.6% | 1,301, **55.0%** | 4,830, 19.2% |
| `PUPPUP` | 4,521, 16.4% | 843, **77.1%** | 5,099, 20.4% |
| `PPUPUP` | 4,280, 15.3% | 813, **75.9%** | 5,241, 14.5% |

For the two-user-hop paths the three lists share 10-20% of their members (Jaccard). Ties alone
do a lot of this: for `PUP` the builder and the exact top 20 apply the same rule and still share
only about 40%, because two random tie-breaks keep different products. The builder's neighbours
carry 43-56% of the random-walk top 20's probability on the two-user-hop paths.

## Tie-break seed against run seed

HAN on the 1,730-product sample, manual set 5 (`PP`, `PUP`, `PCP`, `PPUP`), split 1, hidden 128,
warm-up 15; test Hit@10:

| | run seed 1 | run seed 2 | run seed 3 | mean |
|---|---|---|---|---|
| tie seed 1 | 0.412 | 0.422 | 0.460 | 0.431 |
| tie seed 2 | 0.425 | 0.450 | 0.450 | 0.441 |
| tie seed 3 | 0.394 | 0.456 | 0.460 | 0.437 |
| mean | 0.410 | 0.443 | 0.456 | |

## How it was checked

1. **Code:** every function of the builder; `topk_per_row` against its documented rule.
2. **Full scale:** a structure-only copy of the rebuilt data (your `link.dat` and label files,
   with `node.dat` regenerated without features). The split matches the LMSPS repo exactly: 655,879 / 81,984 /
   81,984 edges, and split 1 gives training-edge fingerprint `c5f7bccbf61c1b0f`. Then each path was built and
   compared with the exact chain product on 400 sampled products.
3. **Small samples:** the same comparison on the 1,730- and 3,635-product samples, where the
   exact products are small enough to compute fully. Same pattern: exact for one intermediate node,
   54-72% inside the exact top 20 for two user hops.

## What changed

Nothing in the code. This write-up.
