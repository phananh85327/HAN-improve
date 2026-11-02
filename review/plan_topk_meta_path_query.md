# Plan: top-k meta-path query as a second discovery method for HAN

**Status:** an idea, nothing implemented (written 2026-10-31). The method is Zhu et al.,
"Effective and Efficient Discovery of Top-k Meta Paths in Heterogeneous Information Networks"
(IEEE TKDE 2020), code at `github.com/littlepig2013/topk-meta-path-query` (C++).

**Why:** the thesis now compares how path sets are chosen (expert, LMSPS search, random) by cost
and performance. This method chooses paths a completely different way: graph statistics on a
CPU, no model training. If it also fails to beat the expert and random sets, the conclusion no
longer rests on one search method; if it wins, that is a stronger result. The paper lists link
prediction and recommendation as its own future work.

**The approach in one paragraph:** query (user, product) pairs sampled from the training edges,
with those edges removed from the graph the search sees. For each pair the method returns its top
k meta-paths from the user to the product. Every such path starts with the user's own purchase
step, so dropping that step leaves a product-to-product path HAN can use (`U-P-U-P` becomes
`PUP`). Add up each path's scores over all pairs and keep the best 4 and the best 6. They become
two `arch.py` keys and go through the existing HAN and metapath2vec runs unchanged.

## 1. Decisions to freeze before any HAN result

Fixed in advance, like the LMSPS sets were, so the choice cannot lean on HAN's numbers.

| Decision | Value | Why |
|---|---|---|
| Data and split | The rebuilt `data/AMAZON`, split seed 1, training user-product edges only | Same data and split as HAN and LMSPS; checked by fingerprint (section 3) |
| Query pairs | M = 200 training edges, drawn uniformly with a fixed seed; the pilot may lower M | Enough pairs for a stable total; the pilot sets the budget |
| Who can be queried | Users and products with at least 2 training edges | After its edge is held out, the user must still connect to the graph |
| Held-out edges | All M query edges removed from the search graph at once | The search must not find the purchase it explains |
| Scoring | MNIS (the paper's method), length penalty beta^l with beta = 0.2, k = 5 paths per pair | The paper's defaults (Sec. 8.2: beta 0.2); k 5 as in its batch driver |
| Mapping | Drop the first user-to-product step; read both directions of `PP` and `CC` as one | HAN's graphs are symmetric and built from plain relations |
| Filter | Keep paths of 2-6 letters after the mapping | The metapath contract (S1) and the LMSPS candidate space (up to 5 hops) |
| Combining pairs | Sum of each path's MNIS score over all pairs; report the count of pairs listing it as a check | The paper's own aggregation (Sec. 8.4, step 3), done over all pairs instead of per node |
| Set sizes and keys | Top 4 and top 6 by the summed score: `amazon_lp_topk_f4`, `amazon_lp_topk_f6` | Same sizes as LMSPS and the manual sets |
| Evaluation | HAN and metapath2vec, seeds 1-3, the settings of record (`HAN_REC`, `M2V_FAST`) | Same protocol as the main comparison |
| Comparison | Test numbers against M5/M6, LMSPS `a1` and the random draws (their spread) | As for LMSPS; this method never sees validation edges, but keep one rule |

## 2. What to keep, change, add and salvage

### Their repository (`topk-meta-path-query`)

**Keep unchanged**
- `HIN_Graph.*`, `yagoReader.*`, `SimCalculator.*`, `CommonUtils.*`: graph storage and the readers.
- The search and the MNIS score in `TopKCalculator.cpp` (apart from the two fixes below).
- `main.cpp`: single queries, for the pilot (`--fast_advance AMAZON <user> <product> 5 3 MNIS 1 0.2`).
- `genTopKMPs.cpp` as the batch driver: it already skips a pair whose result file exists, so an
  interrupted run resumes.
- `--train` mode: it computes the per-relation statistics the score needs (`cache/AMAZON*.txt`).

**Change** (all small; the C++ is otherwise used as published)

| # | File | Change | Why | Size |
|---|---|---|---|---|
| C1 | `AppUtils.cpp` | An AMAZON branch in `loadHinGraph` reading `AMAZON/AMAZONAdj.txt`, `AMAZONEntityType.txt`, `AMAZONEdgeType.txt` (no name file, like DBLP) | The datasets are hard-coded (ACM, DBLP, YAGO) | about 10 lines |
| C2 | `TopKCalculator.cpp`, `getHit` | Use the graph's adjacency maps by reference instead of copying them, and delete the two `clear()` loops at its end (with references they would erase the graph) | It copies the whole adjacency at every call and at every path step, twice per candidate path: on 1.24M edges this would dominate the run time. Results do not change | about 10 lines |
| C3 | `TopKCalculator.cpp` | `srand(time(NULL))` becomes a fixed seed | Random tie-breaks in the priority queues make runs differ otherwise | 1 line |
| C4 | `genTopKMPs.cpp` | Accept AMAZON (it rejects all but DBLP and ACM); MNIS only (it runs four scores); k, beta, the pairs file and the output folder as arguments; print the time per query | Its settings are hard-coded (k 5, beta 0.1) | about 30 lines |
| C5 | Running it | One thread | `similarPairsSize` is a static variable every query overwrites, so parallel queries corrupt each other. Making it per-thread is possible but needs care; only worth it if the pilot is slow | none |

Leave alone, and say so in the thesis: `max_candidates_flag == true` (line 1122) compares instead
of assigning, so the "stop after two tied candidates" rule never fires. It only affects ties;
fixing it would change the published method.

**Not needed:** `test.cpp` (accuracy on labelled pairs, for node classification),
`genRandomWalk.cpp` and `mp2vec-Clf/` (their metapath2vec; the HAN repo has its own), the ACM,
DBLP and YAGO data, the existing `cache/` files.

**Building on Windows:** as you already do. Their Makefile targets plain `g++ -std=c++11 -pthread`.

### HAN repo

**Keep unchanged:** `data_loader.py`, `link_data.py` (the split and the fingerprint),
`build_metapath_graphs.py` (it builds any product-to-product path), `train_han.py`,
`train_metapath2vec.py`, `summarize_results.py`, `eval_emb.py`, `check_handover.py`.

**Add**
1. **`topk_export.py`, the converter.** From `data/AMAZON` and split seed 1 it writes:
   - `AMAZONAdj.txt`: one line `src dst type` per edge, global node ids as in `node.dat`, each
     relation once (their code walks both directions itself), types numbered from 1 because the
     sign of a type marks the direction:

     | Type | Edges | From `link.dat` |
     |---|---|---|
     | 1 | product → product, also-bought | relation 0 |
     | 2 | category → category, parent → child | relation 2 |
     | 3 | product → category | relation 4 |
     | 4 | user → product | the split's training edges, minus the held-out query edges |

   - `AMAZONEntityType.txt`: `id type` for every node (0 product, 1 category, 2 user).
   - `AMAZONEdgeType.txt`: `P_P 1`, `C_C 2`, `P_C 3`, `U_P 4`.
   - the query pairs, in `genTopKMPs`'s format (`user<TAB>product` per line, global ids).
   - a record (JSON): settings, the held-out pairs, and a `data` block with the same fields as
     an LMSPS search record (node counts, number of target edges, fingerprint of the full
     training edges, split seed, dataset, target relation), so `check_handover.py` can confirm it
     ran on HAN's data.

   A prototype of the edge file was written and run on the full rebuilt data (2026-10-30):
   1,241,899 edges.
2. **`topk_import.py`, the mapper.** Reads the per-pair result files (`score:type<TAB>type...`),
   turns each signed type sequence into letters, drops the leading `U`, filters, sums the scores
   per path and prints:
   - the ranking with scores and pair counts
   - the two `arch.py` entries with a provenance line
   - how many results were dropped by the filter, and the total query time

   It adds the per-pair results and the totals to the record.

   Type to letter (the walk starts at the user):

   | Signed type | Step | Letter added |
   |---|---|---|
   | +1 / -1 | product to product, either direction | `P` |
   | +2 / -2 | category to category, either direction | `C` |
   | +3 | product to category | `C` |
   | -3 | category to product | `P` |
   | +4 | user to product | `P` |
   | -4 | product to user | `U` |

   Example: `4 -4 4` → `UPUP` → `PUP`; `4 3 -3` → `UPCP` → `PCP`; `4 1` → `UPP` → `PP`.
3. **Tests:** the mapping table and the examples; the converter on the synthetic dataset of
   `tests/test_data_handover.py` (types start at 1, held-out edges absent, only training
   user-product edges, edge counts per type).

**Change**
- `arch.py`: the two keys with a provenance line (beta, k, M, pairs seed, date), committed before
  any HAN run.
- `run_all.bat`: a `TOPK` list in the `build`, `han` and `m2v` loops, and in `seeds` if the sets
  join the main comparison; `tests/test_run_all.py` then checks the new keys exist.
- `README.md`: inputs and run flow; `METAPATH_CONTRACT.md` already states the rules this method
  must meet.
- A review note on the results afterwards (H9).

### LMSPS repo

- **`dat_generator.py`: not needed, and better not used here.** It builds the HGB files from the
  extracted CSVs (ids, features, relations, labels, `meta.json`), and the split comes later, in
  `link_data.py`. The top-k input must hold exactly the nodes, edges and training split HAN and
  LMSPS used, so it is written from the existing `node.dat`/`link.dat` with the HAN repo's split
  code, and the fingerprint proves it. Worth taking from it:
  - its schema table (relation ids 0-7 and their directions), which the type table above follows
  - its name conventions (ASIN, customer id), if you want readable output. Their name reader
    splits on spaces, so category names ("Name [ID]") would need their id alone.
- **`link_data.py`:** the same file as the HAN repo's; use the HAN repo's.
- **The search-record format:** its `data` block is what the top-k record copies (above).
- **`reselect_fleet.py` and `review/evidence/part4_split_half.py`:** reuse the idea, not the
  code, for a stability check. Split the query pairs into two halves, combine each, and see how
  many of the top 4 / top 6 paths they share; or redraw the pairs a few hundred times.
- **`review/evidence/part4b_search_vs_han.py`:** its comparison of the search's path preferences
  with HAN's path weights works for this method too, with summed MNIS scores in place of alpha
  ranks.
- **`run_fleet_v3.bat`:** the pattern for a resumable Windows batch (pilot and full stages, logs).

## 3. Phases, gates and time

| Phase | Work | Time | Gate |
|---|---|---|---|
| 0. Pilot | C1-C3; the converter with M = 5; `--train AMAZON`; 5 queries with `main.cpp`, timed | 1-2 h | Median time per query: under 3 min, go with M = 200 (about 10 h); 3-10 min, M = 100, or C5 with threads; over 10 min, stop and report the cost as the finding |
| 1. Tools | `topk_export.py`, `topk_import.py`, tests | half a day | Tests pass; `check_handover.py <record>` prints SAME |
| 2. Queries | C4; M pairs, one thread, resumable | overnight | Every pair has a result file; per-query times recorded |
| 3. Freeze | Combine, stability check, two `arch.py` keys with provenance; commit | 1 h | At least 6 distinct paths after the filter |
| 4. Runs | `build`, HAN and metapath2vec at seeds 1-3 on both sets, `summarize` | about 1.5 GPU h | Every run early-stopped; one data version |
| 5. Write-up | Results against M5/M6, LMSPS and the random draws (test numbers); path weights and coverage; cost table (CPU hours here, 19 GPU hours for the LMSPS fleet, about 0 for the expert and random sets); review note; thesis section | half a day to a day | |

## 4. Risks

| Risk | What to do |
|---|---|
| Too slow on Amazon (the paper's ACM, a similar size: 50-300 s per query) | The pilot gate; fewer pairs; the C2 fix; threads only after C5 |
| The method was built to explain one pair, not to feed a model | State that combining pairs is this thesis's design; the stability check shows whether the result depends on which pairs were drawn |
| MNIS may favour narrow paths: its strength term penalises busy relations (user-product about 0.08, also-bought roughly 0.3-0.4) and rarity rewards paths few other pairs share | Run them anyway, as with LMSPS's near-empty category paths; report coverage per path |
| Reading both directions as one merges two of their paths into one HAN path | Their scores are summed; HAN cannot tell them apart anyway |
| Paths longer than 6 letters | Dropped by the filter; report how many |
| Randomness | C3 fixes the seed; one thread keeps runs repeatable |

## 5. Done when

- `amazon_lp_topk_f4` and `_f6` are in `arch.py` with provenance, committed before any HAN run
- the record (settings, pairs, per-pair results, totals, CPU time) is committed
- HAN and metapath2vec results at seeds 1-3 are in `results.csv`
- a review note compares them with the expert, LMSPS and random sets
- the cost table has the method's measured CPU time
