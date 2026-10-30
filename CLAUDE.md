# HAN-improve: project notes for Claude Code sessions

## Session logs live in `chat/`

- Every session is saved automatically by `.claude/hooks/save_chat_log.ps1`
  (Stop and SessionEnd hooks in `.claude/settings.json`):
  `chat/<date>_<session-id>.md` is the readable transcript (user + assistant text,
  tool names only); the matching `.jsonl` is the raw Claude Code transcript
  (git-ignored).
- **Before continuing any prior work, read the newest relevant `chat/*.md`.**
- Written reports also live in `chat/`. Current ones:
  - `chat/2026-10-16_han-paper-fidelity-review.md` -- review of this repo's HAN
    link-prediction implementation against Wang et al. (WWW-2019, `2.pdf`), with
    measured evidence, what was fixed, and what is still deferred.
  - `chat/2026-10-16_18_lmsps-paper-fidelity-review.md` -- the LMSPS-side session
    (search fidelity, fleet v2 protocol, how the `amazon_lp_f4_*` / `f6_*` sets
    in `arch.py` were selected).
  - `chat/2026-10-23_graphs-preprocessing-and-edition-twins.md` -- audit of the
    rebuilt split-1 graphs (OK), the preprocessing chain, and the duplicate-edition
    finding: 46 % of products share an identical reviewer set, 50 % of test edges
    have an exact train twin (61-73 % counting near-duplicates); HAN scores ~0.95
    Hit@10 on twin edges vs ~0.80 on the rest. Every evaluation now prints a
    twin / non-twin breakdown.
- If the hook seems inactive in a session (no new `chat/` file after a turn), open
  `/hooks` once or restart Claude Code; the settings watcher only picks up
  `.claude/` if it existed when the session started.

## Environment

- Training runs in the logs used `C:\Users\PhanAnh\.conda\envs\lmsps_env`
  (torch + CUDA); that env is not on this machine.
- `C:\Users\PC\.conda\envs\han` (created 2026-10-16, conda-forge Python 3.11) has
  numpy 1.26.4 / scipy 1.11.4 / scikit-learn / tqdm and **CPU-only torch 2.14**:
  enough for graph building, the bias audit and smoke tests, far too slow for
  real training. Anaconda's default channels are ToS-blocked on this machine; use
  `-c conda-forge --override-channels`.
- Windows PowerShell 5.1 only: no bash, no jq, no pwsh. Hooks and scripts must be
  5.1-compatible.

## Protocol for the rerun on the rebuilt data (2026-10-27) and run flow

- **Do not change without the user:** the data in `data/AMAZON/` is the rebuilt data copied from
  the LMSPS repo (has `meta.json`; split 1 gives training-edge fingerprint `c5f7bccbf61c1b0f`,
  checked by `check_handover.py`); split seed 1; the manual sets `amazon_lp_manual_1..8` and
  the random draws `amazon_lp_rand4_d1..5` / `amazon_lp_rand6_d1..5` in `arch.py` (frozen
  before any Fleet v3 result); the main comparison `amazon_lp_v3_f4_a1` vs `amazon_lp_manual_5`
  and `amazon_lp_v3_f6_a1` vs `amazon_lp_manual_6`, 3 seeds.
- The Fleet v3 sets are in `arch.py` (2026-10-29), from the LMSPS repo's
  `hgb/logs/Fleet v3/final_ns4|final_ns6/reselect_d010.txt`: `amazon_lp_v3_f4_a1`, `_a2`, `_c`
  and `amazon_lp_v3_f6_a1`, `_a2`, `_c` (per size, the first two averaged one-SE sets and the
  vote's consensus; LMSPS review part 4). `run_all.bat` lists exactly these six. Their category
  paths are empty for ~80% of products (books, music): keep them as selected.
- The whole experiment is `run_all.bat <stage>`: check, build, audit, smoke, popularity, han,
  gat, m2v, seeds, summarize, push; `dry` prints every command. A run whose result exists is
  skipped; the user also starts single runs by hand. Details and the reading rules are in README
  "Run flow" and `review/h6_run_protocol.md`. Winner selection is by **validation BPR**.
- Naming: results `output/AMAZON/v3/<model>_<archkey>_s<seed>.pt`, logs
  `output/AMAZON/v3/logs/<same>.txt` (committed by `run_all.bat push`; only the top-level
  `logs/` is git-ignored). Never use `--save-emb auto` in
  batch runs. Everything trained before the rebuild (`logs/Final 1-5`, `output/AMAZON/Han_*`,
  `M2v_*`, old `.pt` files) is legacy.
- Baselines: GAT = 2-layer union-graph GAT without output ELU, hyperparameters matched to the
  HAN config of record, and each training step removes its own positive edges from the graph
  (review H4); metapath2vec = fast walk corpus; popularity (`popularity_baseline.py`) is the
  floor. The `--paper-config` and `--paper-walks` rows are no longer run (review H3, H6).

## Repo conventions worth knowing

- Meta-path graphs: `build_metapath_graphs.py` writes `<path>_top<k>_split<s>.npz`
  plus a `.meta.json` sidecar (tie-break mode, training-edge fingerprint). `train_han.py`
  refuses a graph built from other data; `--skip-existing` rebuilds such a graph.
  `check_graph_bias.py` audits a graph directory. The rerun's graphs live in
  `data/AMAZON/metapath_graphs_v3`; older `metapath_graphs_*` folders are from the data
  before the rebuild.
- Every run records its data (`data` in the result: meta.json, node counts, fingerprint);
  `summarize_results.py` keeps data versions and settings (`variant`) apart.
- HAN starts every path's GAT from the same weights (`--independent-init` = old behaviour), so
  results and path weights do not depend on the order of the paths (review H3).
- Review notes are in `review/h1..h8_*.md` (H8: the results of the rerun, with
  `review/evidence/h8_results.py`) and `review/recheck_full_repo.md` (the whole repo read
  again after the runs); `python -m pytest tests` runs the tests.
- Duplicate editions: `link_data.product_twin_groups` / `twin_in_train`,
  `han_utils.report_twin_breakdown`, `check_edition_twins.py`. On the data before the
  rebuild the "no twin in train" row was the recommendation-quality number; the rebuilt
  data merges editions (40 of 81,984 test edges left with a twin).
- `preprocessing/` holds the old C# parser/filter (reference copies) and
  `dat_generator.py`; its README lists the filter chain and known data issues. The rebuilt
  data comes from the LMSPS repo instead (`hgb/amazon_extract.py` + `hgb/dat_generator.py`).
  `--category-link leaf` writes a leaf-only P-C variant; keep it in
  `data_leafcat/AMAZON` and run scripts with `--root data_leafcat`.
- `arch.py`: the Fleet v2 keys (`amazon_lp_f4_*`, `_f6_*`, aliases `_f4_c` / `_f6_c`) were
  searched on the data before the rebuild and are kept for the record only.
- `train_han.py --paper-config` applies the paper's Sec. 5.3 hyperparameters;
  explicit flags override it.
- `README.md` has a "Deviations from the paper" section; keep it current when the
  model or pipeline changes.
