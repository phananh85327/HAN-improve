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

## Locked protocol (2026-10-23) and run flow

- **Locked, do not change without the user:** `preprocessing/` chain and `data/AMAZON/*.dat`;
  the 13 fleet-v2 shortlist keys in `arch.py` (+ `_c` aliases) and the ablation keys
  `amazon_lp_manual`, `amazon_lp_random4_s0`, `amazon_lp_random6_s0`; split seed 1; graphs in
  `data/AMAZON/metapath_graphs_all_unique` (random tie-break, sidecars). The duplicate-edition
  issue is reported (twin / non-twin breakdown), not fixed in the data.
- The whole experiment is `run_all.bat <stage>` (`execute.bat` forwards): build, audit, han,
  m2v, gat, seeds, m2v_paper, han_paper, summarize; `dry` prints every command. Details and
  the result-reading rules are in README "Run flow". Winner selection is by **validation BPR**.
- Naming: blobs `output/AMAZON/<model>_<archkey>_s<seed>[_paper].pt`, logs `logs/Final 5/<same>.txt`.
  Never use `--save-emb auto` in batch runs. `logs/Final 1-4` and `*_emb_*.pt` blobs are legacy.
- Baselines stay close to their original code on purpose: GAT = 2-layer union-graph GAT without
  output ELU, hyperparameters matched to the HAN config of record; metapath2vec default = fast
  walk corpus, `--paper-walks` = HAN-paper walk settings used only on the headline keys.

## Repo conventions worth knowing

- Meta-path graphs: `build_metapath_graphs.py` writes `<path>_top<k>_split<s>.npz`
  plus a `.meta.json` sidecar (tie-break mode). Graphs without the sidecar were
  built with the biased id tie-break and should be rebuilt. `check_graph_bias.py`
  audits a graph directory. As of 2026-10-23 `data/AMAZON/metapath_graphs_*`
  hold only sidecar-bearing split-1 graphs for the 13 fleet-v2 shortlist paths
  (`metapath_graphs_all_unique` = the union; the per-arch dirs are byte-identical
  copies of those files).
- Duplicate editions: `link_data.product_twin_groups` / `twin_in_train`,
  `han_utils.report_twin_breakdown`, `check_edition_twins.py`. Read the "no twin
  in train" row of any result as the recommendation-quality number.
- `preprocessing/` holds the C# parser/filter (reference copies) and
  `dat_generator.py`; its README lists the filter chain and known data issues.
  `--category-link leaf` writes a leaf-only P-C variant; keep it in
  `data_leafcat/AMAZON` and run scripts with `--root data_leafcat`.
- `arch.py`: `amazon_lp_f4_c` / `amazon_lp_f6_c` are aliases of `_f4_s1` / `_f6_s4`
  matching the LMSPS repo's consensus keys.
- `train_han.py --paper-config` applies the paper's Sec. 5.3 hyperparameters;
  explicit flags override it.
- `README.md` has a "Deviations from the paper" section; keep it current when the
  model or pipeline changes.
