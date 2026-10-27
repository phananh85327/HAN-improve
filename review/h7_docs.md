# HAN review part H7: README, contract and project notes

**Scope:** the documents a reader of this repo (you, a thesis examiner, or a later session)
takes facts from:
- `README.md`
- `METAPATH_CONTRACT.md`
- `CLAUDE.md`
- the docstrings at the top of the scripts

**Bottom line:** they described the data before the rebuild and the Fleet v2 protocol. Updated
to the rebuilt data and the rerun. No must-fix of its own: the code changes were in H1-H6.

## What was out of date, and where it was fixed

| What | Where | Now |
|---|---|---|
| 504-d binary product features | README deviations, contract R1 | 260-d (256-d LSA of the title + 4 group bits) |
| Node counts 90,659 / 39,754 / 171,285 | Contract R1 | 48,883 / 37,411 / 53,569, with the old counts noted |
| `--split-seed 0` in commands | README flow, contract F2 and checklist, `train_han.py`, `train_metapath2vec.py` docstrings | Split 1, the fleets' split (and the scripts' default since H1) |
| 46% / 55% / 50% twin figures, "+7-8 points", "read the non-twin row" | README metrics, `CLAUDE.md`, `link_data.py`, `check_edition_twins.py` | Marked as the old data; the rebuilt data leaves 40 of 81,984 test edges with a twin |
| The locked protocol: 13 Fleet v2 keys, removed random keys, `logs/Final 5`, `metapath_graphs_all_unique`, paper rows | README run flow, `CLAUDE.md` | The rerun protocol (H6): Fleet v3 keys, M1-M8, the random draws, the main comparison, `output/AMAZON/v3` |
| Reference numbers (HAN 0.85-0.87 Hit@10) | Contract section 7 | Marked as the old data and kept for the record; the floor is popularity (0.42), not just 0.10 |
| Isolated-node examples on 90,659 products | Contract S3 | The rebuilt data's numbers (H2) |
| `train_han.py` prerequisites (`--num-hops 3 --ns 5`, split 0) | `train_han.py` docstring | The current flow |
| GAT and HAN behaviour changes | README deviations and files, `CLAUDE.md` | Same start for every GAT (H3), target-edge removal (H4), popularity row, full ranking, data checks |

Left as they were:
- The `chat/` references in `CLAUDE.md` and the README: your own session notes, on your PC.
- The environment notes.
- The history of the 30.6 GiB out-of-memory error in `model_han.py` (now marked as the old data).

## What changed

Committed on `claude/han-review`:
- `README.md`: inputs, run flow, running a model by hand, settings, metrics.
- `METAPATH_CONTRACT.md`: R1, S3, F1, F2, sections 6 and 7.
- `CLAUDE.md`: the rerun protocol and the repo conventions.
- Docstrings: `link_data.py`, `train_han.py`, `train_metapath2vec.py`, `build_metapath_graphs.py`,
  `check_edition_twins.py`, `model_han.py`.
