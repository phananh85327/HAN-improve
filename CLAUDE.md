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

## Repo conventions worth knowing

- Meta-path graphs: `build_metapath_graphs.py` writes `<path>_top<k>_split<s>.npz`
  plus a `.meta.json` sidecar (tie-break mode). Graphs without the sidecar were
  built with the biased id tie-break and should be rebuilt. `check_graph_bias.py`
  audits a graph directory. `data/AMAZON/metapath_graphs_manual_tierandom/` holds
  the only rebuilt (random tie-break, split 0) set so far; all other
  `metapath_graphs_*` dirs are legacy.
- `train_han.py --paper-config` applies the paper's Sec. 5.3 hyperparameters;
  explicit flags override it.
- `README.md` has a "Deviations from the paper" section; keep it current when the
  model or pipeline changes.
