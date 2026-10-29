# Manual meta-path sets for the AMAZON user–product graph

Hand-picked meta-path sets to compare against the sets LMSPS finds automatically, when training
HAN (or any meta-path model) for **user → product link prediction**.

Prepared 2026-10-26. Freeze these sets before you see the automatic results on the rebuilt data,
so the comparison can't be tuned after the fact.

## 1. The graph and the notation

| Node type | What it is |
|---|---|
| **P** | Product (book, music, DVD, video): the node HAN encodes |
| **U** | User (customer) |
| **C** | Category |

| Relation | Meaning |
|---|---|
| **P–P** | "Customers also bought": Amazon's co-purchase list, built from *all* Amazon customers |
| **U–P** | A user gave the product a positive review |
| **P–C** | Product → its leaf category |
| **C–C** | Category hierarchy (parent ↔ child) |

A meta-path is written from the target product outwards, as in `arch.py`. For example, `PUP` means
product ← user ← product: two products reviewed by the same user.

A path and its reverse (`PPUP` / `PUPP`) give the same HAN graph once it is made symmetric. These
sets use the key the search keeps, the alphabetically smaller one.

## 2. How the paths were chosen

1. **One path per kind of evidence.** Products relate to each other in three independent ways:
   - **co-purchase:** Amazon's own behaviour data (`PP`)
   - **co-review:** your users' behaviour (`PUP`)
   - **content:** shared categories (`PCP`)

   A good set covers the evidence the model needs. The recommender literature uses exactly this
   pattern: one item–user–item path plus one item–attribute–item path per attribute.
2. **Short before long.** The HAN paper uses symmetric paths of 1–4 hops. Every extra user hop
   multiplies the neighbourhood, and HAN keeps only *which* products are reachable, not how
   strongly.
3. **No near-complete graphs.** Measured on the small graph built from your snippet (table below),
   every path with two user hops connects a product to **71–80% of all products**. Attention over
   "everyone" carries almost no signal, and at full scale such a graph has billions of edges.
4. **No nested pairs.** With self-loops on P–P and C–C (as LMSPS builds them), a longer path
   contains every edge of the shorter one: `PCCP` ⊇ `PCP`, `PPP` ⊇ `PP`, `PUPUP` ⊇ `PUP`.
   `PCCP` adds about **1** neighbour per product over `PCP`, so it is left out.
5. **Sizes match the automatic sets** (4 and 6 paths), so a win can't come from having more paths.

### Density of the candidate paths

On the k3 mini graph (1,730 products), this is what HAN would build. Treat it as illustrative and
measure on the real data before training (section 5).

| Path | Meaning | Neighbours per product (median) | Share of all product pairs |
|---|---|---|---|
| `PP` | also bought | 0* | 0.01%* |
| `PPP` | bought with what's bought with it | 0* | 0.02%* |
| `PUP` | reviewed by the same user | 52 | 4.4% |
| `PCP` | same leaf category | 52 | 3.4% |
| `PCCP` | same or parent category | 52 | 3.5% (= `PCP` + 1 neighbour) |
| `PPUP` | reviewed by people who reviewed its also-bought products | 62 | 5.3% |
| `PCCCP` | sibling categories (via the parent) | 92 | 6.2% |
| `PCPCP` | categories that co-occur with its categories | 329 | 19% |
| `PCPUP` | reviewed together with products of its category | 1,300 | **71%** |
| `PUPUP` | two co-review steps | 1,408 | **77%** |
| `PUPPUP`, `PPUPUP` | two co-review steps plus also-bought | 1,420–1,456 | **78–80%** |

\* The snippet keeps few also-bought partners. On the full data, 88% of products have at least one
(about 4 on average).

## 3. The sets

| ID | Set | Size | What it tests |
|---|---|---|---|
| **M1** | `PUP`, `PCP` | 2 | Minimal HAN-paper style: one behaviour path, one attribute path |
| **M2** | `PP`, `PUP`, `PCP` | 3 | The standard recommender trio |
| **M3** | `PP`, `PUP`, `PPUP` | 3 | Behaviour only: do categories add anything? |
| **M4** | `PCP`, `PCCCP`, `PCPCP` | 3 | Content only: how far do categories go alone? |
| **M5** | `PP`, `PUP`, `PCP`, `PPUP` | 4 | Balanced expert set, same size as the automatic ns 4 sets |
| **M6** | `PP`, `PPP`, `PUP`, `PCP`, `PPUP`, `PCCCP` | 6 | Balanced expert set, same size as the automatic ns 6 sets |
| **M7** | `PP`, `PPP`, `PUP`, `PCP` | 4 | No expertise: every path up to 2 hops |
| **M8** (optional) | `PUP`, `PCP`, `PUPUP`ᵏ, `PCPUP`ᵏ | 4 | Expert long-range, **only with neighbour pruning** (ᵏ) |

### M1: `PUP`, `PCP` (HAN-paper style)
- **`PUP`** is item-based collaborative filtering: products that share a reviewer.
- **`PCP`** is content similarity: products in the same leaf category.
- **Why:** HAN's own experiments use two short symmetric paths per dataset, one through the
  behaviour relation and one through an attribute. This is the smallest set an expert would
  defend.
- **Reading the result:** the floor. If an automatic set can't beat M1, the search has not earned
  its cost.

### M2: `PP`, `PUP`, `PCP` (the standard trio)
- **Adds `PP`**, Amazon's "customers also bought". It is built from all Amazon customers, so it is
  behaviour data that your reviewers alone can't provide.
- **Why:** item–user–item, item–attribute–item, and the item–item behaviour link are the paths
  heterogeneous-graph recommenders use for Amazon-style data.
- **Reading the result:** M2 vs M1 shows whether Amazon's co-purchase signal adds anything beyond
  your users' co-reviews.

### M3: `PP`, `PUP`, `PPUP` (behaviour only)
- **`PPUP`** links a product to what the reviewers of its also-bought products liked: it bridges
  Amazon's co-purchase list and your users' reviews.
- **Why:** in link prediction, behaviour usually dominates content. This tests that directly.
  `PUPUP` is left out because it is near-complete (77% of pairs).
- **Reading the result:** M3 ≈ M5 means categories contribute nothing; M3 < M5 means they do.

### M4: `PCP`, `PCCCP`, `PCPCP` (content only)
- **`PCCCP`** reaches sibling categories through the parent (e.g. other sub-genres of the same
  genre). **`PCPCP`** reaches categories that are often assigned together with this product's
  categories.
- **Why:** a control. Content paths are the only ones that work for a product nobody has bought
  or reviewed yet (cold start).
- **Reading the result:** expected to be the weakest set. The size of the gap to M2 shows how much
  of the task is behaviour. `PCPCP` is dense (19% of pairs); prune it if it is too slow.

### M5: `PP`, `PUP`, `PCP`, `PPUP` (balanced, 4 paths)
- **Why:** one path per evidence type (co-purchase, co-review, content) plus the bridge path, with
  no near-complete and no redundant graph. This is the set an expert would hand in.
- **Reading the result:** **the main comparison for the automatic ns 4 sets.** Same size, so a
  difference reflects path choice, not path count.

### M6: `PP`, `PPP`, `PUP`, `PCP`, `PPUP`, `PCCCP` (balanced, 6 paths)
- **Adds** `PPP` (bought with what's bought with it: a wider co-purchase neighbourhood) and
  `PCCCP` (sibling categories: coarser content).
- **Why:** the two extra slots go to the most meaningful *wider* versions of the strongest
  families, still avoiding near-complete graphs.
- **Reading the result:** **the main comparison for the automatic ns 6 sets.**

### M7: `PP`, `PPP`, `PUP`, `PCP` (no-expertise reference)
- **Every target-ended path up to 2 hops**, with no judgement involved. It mirrors the
  "maximum hop 2" setting the LMSPS paper compares against (its Table 2, SeHGNN).
- **Reading the result:** if the automatic search can't beat this set, the search isn't needed for
  this dataset. If M5/M6 beat M7, expertise adds something beyond "take all short paths".

### M8 (optional): `PUP`, `PCP`, `PUPUP`ᵏ, `PCPUP`ᵏ (expert long-range)
- **Use it only if your HAN graph builder can prune**: ᵏ means keeping each product's *k* strongest
  neighbours by walk count (e.g. k = 50, PathSim-style). Unpruned, these two paths connect 71–77%
  of all pairs.
- **Why:** the LMSPS paper argues that long paths help. This gives long-range paths an expert's
  fair chance: `PUPUP` (neighbours of co-review neighbours) and `PCPUP` (what the audience of this
  product's category reviews).
- **Reading the result:** compare with the automatic sets, which in Fleet v2 leaned on
  two-user-hop paths. If you prune M8, prune the automatic sets the same way.

## 4. Comparing fairly

- **Same graph construction for every set:** same self-loop convention, same symmetrisation, same
  pruning (or none). With binary reachability graphs, the automatic sets' long paths will be
  near-complete too. How the graphs are built can matter more than which paths are chosen.
- **User paths from training edges only.** Validation and test edges must not be in any HAN graph,
  as in LMSPS's own graph.
- **Same split and negatives** as the search (`--split-seed 1`). The search chose its paths with
  the validation edges, so report **test** results only.
- **At least 3 seeds per set**; report mean ± std. Differences smaller than the seed spread are
  ties.
- **Add a popularity baseline.** With 99 random negatives, ranking products by popularity alone
  already scores well. Every set should be reported against it.
- **Compare like with like:** M5 against the ns 4 automatic sets, M6 against ns 6, and M7 as the
  reference for both.

## 5. Measure the density on the real data first

Run this on the rebuilt `data/AMAZON` before training HAN. It reads `node.dat` / `link.dat` and
prints, per path, the neighbours per product and the share of pairs, using the same construction
as the table above:

```python
import numpy as np, scipy.sparse as sp

ROOT = 'data/AMAZON'          # rebuilt HGB-format folder
PATHS = ['PP', 'PPP', 'PUP', 'PCP', 'PPUP', 'PCCCP', 'PCPCP', 'PCPUP', 'PUPUP']

types = np.array([int(l.rstrip('\n').split('\t')[2]) for l in open(f'{ROOT}/node.dat', encoding='utf-8')])
shift = {t: int(np.flatnonzero(types == t)[0]) for t in (0, 1, 2)}   # 0=P 1=C 2=U
n = {t: int((types == t).sum()) for t in (0, 1, 2)}
rel = {}
for l in open(f'{ROOT}/link.dat'):
    h, t, r, _ = l.split('\t')
    rel.setdefault(int(r), ([], []))
    rel[int(r)][0].append(int(h)); rel[int(r)][1].append(int(t))

def adj(r, rt, ct):
    h, t = np.array(rel[r][0]) - shift[rt], np.array(rel[r][1]) - shift[ct]
    return sp.csr_matrix((np.ones(len(h)), (h, t)), shape=(n[rt], n[ct]))

pp = ((adj(0, 0, 0) + adj(1, 0, 0)) + (adj(0, 0, 0) + adj(1, 0, 0)).T + sp.eye(n[0])).sign().tocsr()
cc = ((adj(2, 1, 1) + adj(3, 1, 1)) + (adj(2, 1, 1) + adj(3, 1, 1)).T + sp.eye(n[1])).sign().tocsr()
up = adj(6, 2, 0).sign().tocsr()   # use TRAINING edges only when you build the real HAN graphs
A = {'PP': pp, 'CC': cc, 'PC': adj(4, 0, 1).sign().tocsr(), 'CP': adj(5, 1, 0).sign().tocsr(),
     'UP': up, 'PU': up.T.tocsr()}

for path in PATHS:
    m = A[path[:2]]
    for i in range(1, len(path) - 1):
        m = (m @ A[path[i:i + 2]]).sign()
    m = (m + m.T).sign().tolil(); m.setdiag(0); m = m.tocsr()
    d = np.diff(m.indptr)
    print(f'{path:<7} median {np.median(d):>7.0f}  mean {d.mean():>9.1f}  pairs {m.nnz / (n[0] * (n[0] - 1)):.2%}')
```

The relation IDs (0–7) follow `dat_generator.py`'s `link.dat`: 0/1 P–P, 2/3 C–C, 4 P→C, 5 C→P,
6 U→P. Long paths can take a lot of memory at full scale; run them one at a time.

**Rule of thumb:** a path whose graph covers more than about 20% of all product pairs needs
pruning (or should be dropped) before HAN can use it.
