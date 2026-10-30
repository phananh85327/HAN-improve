"""Full re-check (2026-10-30): how often does a product's own walk back to itself take one of
its top-k slots in build_metapath_graphs.py? The builder strips self-loops only after pruning,
so such a row keeps k - 1 other neighbours (before symmetrizing).

    python review/evidence/recheck_topk_self.py data PUP PCP PPUP PUPUP

Uses the builder's own functions, split seed 1, top-20 and the random tie-break seeded as in
the builds (--tie-seed defaults to the split seed). About a minute per path on a CPU.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from build_metapath_graphs import build_base_matrices, topk_per_row  # noqa: E402
from data_loader import data_loader  # noqa: E402
from link_data import build_link_split  # noqa: E402

root, paths = sys.argv[1], sys.argv[2:]
dl = data_loader(os.path.join(root, 'AMAZON'))
base = build_base_matrices(dl, 'AMAZON', build_link_split(dl, 2, 0, seed=1))
for path in paths:
    steps = [path[i:i + 2] for i in range(len(path) - 1)]
    rng = np.random.default_rng(1)                # as materialize_path: one generator per path
    self_kept = rows = 0
    for start in range(0, base['PP'].shape[0], 4096):
        m = base[steps[0]][start:start + 4096]
        for s in steps[1:]:
            m = topk_per_row((m @ base[s]).tocsr(), 20, rng)
        m = topk_per_row(m, 20, rng).tocoo()
        self_kept += int((m.row + start == m.col).sum())
        rows += len(np.unique(m.row))
    print(f'{path}: the product itself is among its top 20 in {self_kept} of {rows} non-empty rows '
          f'({self_kept / max(rows, 1):.1%})')
