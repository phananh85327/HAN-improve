# Example recommendations: which results represent each kind of path set

**Scope:** the results that `recommend_6path.bat`, `recommend_4path.bat` and `recommend_api.py`
(its default list) run for the example recommendations (README "Example recommendations"), and
why these. The numbers come from `output/AMAZON/v3/results.csv` and `review/evidence/by_group.txt`;
`review/evidence/example_recommendations.py` prints them and checks that the `.bat` files name
each model's median random draw.

**Bottom line**
- **One HAN and one metapath2vec result of each kind of path set and size, plus GAT, all at
  seed 1.** Every set has a seed-1 run for both models; seeds 2-3 exist only for the main
  comparison.
- **Automatic and manual: the same sets for both models,** fixed by the protocol before any
  result: the pre-registered main LMSPS set of each size and its main-comparison partner.
- **Random: each model's own median draw by validation BPR.** HAN `rand6_d3` / `rand4_d4`,
  metapath2vec `rand6_d2` / `rand4_d2`. HAN's picks are metapath2vec's second-best and best
  draws, so they would not be typical for it.
- **metapath2vec is in because its lists differ most from HAN's:** its full-ranking top 10s are
  products with about 6 training buyers (HAN about 74-80, the average product 13.4), almost none
  of them from the 1% most-bought (HAN about a quarter).
- **Popularity is out:** it gives every user the same bestsellers. It stays in `results.csv` and
  the thesis tables as the floor.

## 1. The rules

| Kind | Rule | Depends on the results? | HAN | metapath2vec |
|---|---|---|---|---|
| Automatic (LMSPS) | The pre-registered main LMSPS set of that size | No: fixed by the protocol | `v3_f6_a1`, `v3_f4_a1` | the same |
| Manual | Its main-comparison partner | No | `manual_6` (M6), `manual_5` (M5) | the same |
| Random | The median of the five draws of that size by validation BPR: a typical draw, not the luckiest | Yes: each model ranks the draws itself | `rand6_d3`, `rand4_d4` | `rand6_d2`, `rand4_d2` |

GAT (`gat_union_s1`), the best model and one without meta-paths, is in both files. The random pick
goes by validation BPR, as every selection in the protocol does; test numbers only check that the
pick is typical.

## 2. The random draws

Validation BPR at seed 1, best first; the median (third) is the pick:

| Place | HAN, 6 paths | metapath2vec, 6 paths | HAN, 4 paths | metapath2vec, 4 paths |
|---|---|---|---|---|
| 1 | d2 0.2580 | d4 0.3190 | d3 0.2785 | d4 0.3151 |
| 2 | d5 0.2610 | d3 0.3237 | d1 0.2816 | d3 0.3378 |
| **3: the pick** | **d3 0.2708** | **d2 0.3374** | **d4 0.2827** | **d2 0.3733** |
| 4 | d4 0.2782 | d5 0.3509 | d2 0.2865 | d5 0.4075 |
| 5 | d1 0.2821 | d1 0.3762 | d5 0.2922 | d1 0.4152 |

- **The picks are typical on test too:** each pick's test Hit@10 is close to the mean of its five
  draws: HAN 0.694 against 0.691 (6 paths) and 0.671 against 0.668 (4 paths), metapath2vec 0.622
  against 0.613 and 0.583 against 0.585.
- **The two models rank the draws differently:** HAN's 6-path pick (d3) is metapath2vec's second
  best, HAN's 4-path pick (d4) its best, and HAN's best 6-path draw (d2) its median. So HAN's
  picks would show metapath2vec at or near its luckiest, which the rule exists to avoid.
- **The choice matters more for metapath2vec:** at 4 paths its five draws spread over validation
  BPR 0.315-0.415 and test Hit@10 0.548-0.631; HAN's over 0.279-0.292 and 0.660-0.677.
- **The cost:** the random rows of the two models are different path sets. To compare HAN with
  metapath2vec on the same set, use the automatic and manual rows.

## 3. Which size is the main pick

Mean test Hit@10 per family at seed 1:

| Family | HAN, 4 -> 6 paths | metapath2vec, 4 -> 6 paths |
|---|---|---|
| Searched (the three LMSPS sets of each size) | 0.648 -> 0.666 | 0.613 -> 0.593 |
| Manual (M5 -> M6) | 0.665 -> 0.692 | 0.639 -> 0.639 |
| Random (the five draws) | 0.668 -> 0.691 | 0.585 -> 0.613 |

6 paths is the main pick (`recommend_6path.bat`) because HAN's 6-path sets did better in every
family (H8: "More paths help HAN"). For metapath2vec neither size wins: with 6 paths the random
draws do better, the manual sets the same and the searched sets worse. This changes no pick, since
each result goes into the `.bat` file of its own size.

## 4. Why metapath2vec is in

The full-ranking top 10 of 5,000 test users (`by_group.txt`, written by `h8_by_group.py`): the
mean number of training buyers of the recommended products, and the share of them from the 1%
most-bought products. The average product has 13.4 training buyers.

| Result (seed 1) | Hit@10 | Full-ranking Hit@10 | Mean buyers | From the top 1% | Distinct products |
|---|---|---|---|---|---|
| HAN: `v3_f6_a1` / `manual_6` / `rand6_d3` | 0.666 / 0.692 / 0.694 | 0.025 / 0.026 / 0.027 | 73.9 / 78.4 / 74.1 | 24.3 / 26.6 / 23.8% | 10,752 / 10,081 / 11,463 |
| metapath2vec: `v3_f6_a1` / `manual_6` / `rand6_d2` | 0.588 / 0.639 / 0.622 | 0.026 / 0.031 / 0.023 | 5.9 / 6.3 / 5.6 | 0.0 / 0.1 / 0.0% | 9,952 / 12,689 / 10,051 |
| HAN: `v3_f4_a1` / `manual_5` / `rand4_d4` | 0.635 / 0.665 / 0.671 | 0.020 / 0.022 / 0.023 | 80.2 / 79.1 / 78.7 | 25.1 / 25.6 / 25.5% | 10,882 / 9,925 / 10,217 |
| metapath2vec: `v3_f4_a1` / `manual_5` / `rand4_d2` | 0.605 / 0.639 / 0.583 | 0.029 / 0.027 / 0.011 | 5.6 / 5.9 / 5.3 | 0.0 / 0.1 / 0.1% | 11,521 / 10,084 / 5,902 |
| GAT | 0.785 | 0.042 | 152.0 | 51.2% | 5,268 |
| Popularity | 0.423 | 0.009 | 550.1 | 100% | 19 |

- **HAN and GAT lean towards bestsellers; metapath2vec recommends products almost nobody
  bought,** fewer buyers than the average product. So the two models' lists differ even on the
  same path set: one drawn largely from bestsellers, the other almost entirely from products with
  a handful of buyers.
- **metapath2vec is not the weaker recommender on full ranking:** on the four sets both models
  share, its full-ranking Hit@10 is the higher one in all four (`v3_f6_a1` 0.026 against HAN's
  0.025, `manual_6` 0.031 against 0.026, `v3_f4_a1` 0.029 against 0.020, `manual_5` 0.027 against
  0.022), while HAN wins on the sampled Hit@10. Over all 24 sets metapath2vec wins on full ranking
  in 14 (H8, R11).

## 5. Why popularity is out

`popularity_s1` (`popularity_baseline.py`) scores each product by its number of training users,
the same score for every user; seed 1 only breaks ties between equal counts. So its list for any
user is the bestsellers they have not reviewed: 19 products in all across the 5,000 users' top 10s.
It is the floor of the results table (Hit@10 0.423, full ranking 0.009) and stays there, but it
adds nothing to one user's report. For its list anyway, pass
`--emb output\AMAZON\v3\popularity_s1.pt` to `recommend.py`, or serve it with
`recommend_api.py --emb`.

## 6. Reading "similar products"

Of the representative sets only the manual ones hold `PP`, the also-bought link itself; GAT has
those links as edges of its graph. The automatic and random sets reach also-bought neighbours only
through longer paths (`PPUP`, `PPUPP`, `PPP`, `PPCPP`, `PPPPPP`, ...). For a product with a clear
also-bought list, expect the manual sets and GAT to match it more closely.

## Evidence

- `python review/evidence/example_recommendations.py output/AMAZON/v3`: the numbers in sections
  2-4 and 6, from `results.csv`, `by_group.txt` and `arch.py`. It stops with an error if a `.bat`
  file's random pick is not that model's median draw.
- `tests/test_run_all.py::test_recommend_bats`: each `.bat` file names one HAN and one
  metapath2vec result of each kind of set, of the right size, the same automatic and manual sets
  for both, plus GAT; and `recommend_api.py` serves exactly these by default.
