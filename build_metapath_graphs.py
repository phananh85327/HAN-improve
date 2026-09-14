#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Materialize LMSPS-discovered meta-paths as target-type adjacency graphs for HAN.

HAN (WWW-2019) consumes one precomputed target-type neighbor graph per meta-path
(the original TF repo reads data['PAP'], data['PLP'] from a .mat). This script
builds those graphs from the SAME data/<dataset> .dat files the LMSPS pipeline
uses -- no separate export from the C# preprocessing is needed.

Only paths that start AND end at the target type (e.g. PUP, PCP, PPUP) define such
a graph; run train_search.py with --tgt-ended-paths so every discovered path
qualifies.

Leakage rule (identical to utils.py::load_dataset): the U-P relation is replaced by
its TRAIN-ONLY split (same --split-seed), so a PUP edge can never encode a val/test
interaction the model is later evaluated on.

Scale rule: a path like PUP has sum-over-users(degree^2) potential edges -- heavy
users would blow this into hundreds of millions. The chain is therefore multiplied
in row chunks with per-row top-k pruning (by walk count) after every step, then
symmetrized (palindromic paths are symmetric pre-pruning; pruning breaks that).

Usage (from han-recsys/):
    python build_metapath_graphs.py --dataset AMAZON --arch amazon_lp_han \
        --topk 20 --split-seed 0 --target-relation "U,P"

Output: data/<dataset>/metapath_graphs/<path>_top<k>_split<seed>.npz (scipy CSR,
binary, no self-loops -- train_han.py adds self-loops when building DGL graphs).
"""

import argparse
import os

import numpy as np
import scipy.sparse as sp

from data_loader import data_loader
from link_data import build_link_split

# letter codes per dataset, matching utils.py::NODE_TYPE_IDS (kept local so this
# script does not import utils.py's full torch/dgl stack)
NODE_TYPE_IDS = {'ACM': {'P': 0, 'A': 1, 'C': 2, 'K': 3},
                 'AMAZON': {'P': 0, 'C': 1, 'U': 2}}


def local_csr(dl, rel_id):
    """One relation as a CSR over LOCAL (per-type) ids."""
    h_type, t_type = dl.links['meta'][rel_id]
    coo = dl.links['data'][rel_id].tocoo()
    row = coo.row - dl.nodes['shift'][h_type]
    col = coo.col - dl.nodes['shift'][t_type]
    shape = (dl.nodes['count'][h_type], dl.nodes['count'][t_type])
    return sp.csr_matrix((np.ones(len(row), dtype=np.float32), (row, col)), shape=shape)


def build_base_matrices(dl, dataset, link_split):
    """Pairwise step matrices keyed 'XY' (row type X, col type Y). Same-type
    relation pairs are merged symmetric, exactly like the utils.py branches.
    The target relation uses the TRAIN-ONLY adjacency from link_split."""
    if dataset != 'AMAZON':
        raise SystemExit('build_metapath_graphs currently supports AMAZON only '
                         '(relation ids are dataset-specific)')

    PP = local_csr(dl, 0) + local_csr(dl, 1)   # also-bought, both directions
    PP.data[:] = 1.0
    CC = local_csr(dl, 2) + local_csr(dl, 3)   # category hierarchy, both directions
    CC.data[:] = 1.0
    PC = local_csr(dl, 4)
    CP = local_csr(dl, 5)
    UP = link_split['train_adj_src2dst'].astype(np.float32)   # [num_U, num_P], train edges only
    PU = UP.T.tocsr()

    return {'PP': PP, 'CC': CC, 'PC': PC, 'CP': CP, 'UP': UP, 'PU': PU}


def topk_per_row(mat, k):
    """Keep the k largest-valued entries per row of a CSR matrix.

    Deterministic: ties are broken by ascending column index. Walk-count
    matrices are full of ties (a binary step like PP ties every entry at 1.0),
    and np.argpartition breaks them by memory layout -- the kept neighbor set
    would then silently change across scipy versions or re-runs."""
    mat = mat.tocsr()
    indptr, indices, data = mat.indptr, mat.indices, mat.data
    keep_idx = []
    for i in range(mat.shape[0]):
        s, e = indptr[i], indptr[i + 1]
        if e - s > k:
            # primary key: value descending; tie-break: column ascending
            sel = np.lexsort((indices[s:e], -data[s:e]))[:k] + s
            keep_idx.append(sel)
        elif e > s:
            keep_idx.append(np.arange(s, e))
    if not keep_idx:
        return sp.csr_matrix(mat.shape, dtype=np.float32)
    keep = np.concatenate(keep_idx)
    # rebuild row indices from indptr membership
    row_ids = np.searchsorted(indptr, keep, side='right') - 1
    return sp.csr_matrix((data[keep], (row_ids, indices[keep])), shape=mat.shape)


def materialize_path(path, base, topk, chunk_rows=4096):
    """Walk-count matrix for `path` (e.g. 'PUP'): chain-multiply the step matrices
    in row chunks, pruning to top-k per row after every step to bound memory."""
    steps = [path[i] + path[i + 1] for i in range(len(path) - 1)]
    for s in steps:
        if s not in base:
            raise SystemExit(f'path {path}: no base relation for step {s} '
                             f'(available: {sorted(base)})')

    first = base[steps[0]]
    n_rows = first.shape[0]
    chunks = []
    for start in range(0, n_rows, chunk_rows):
        m = first[start:start + chunk_rows]
        for s in steps[1:]:
            m = (m @ base[s]).tocsr()
            m = topk_per_row(m, topk)
        chunks.append(topk_per_row(m, topk))
    out = sp.vstack(chunks).tocsr()

    # binarize, strip self-loops, symmetrize (train_han adds self-loops itself)
    out.data[:] = 1.0
    out.setdiag(0)
    out.eliminate_zeros()
    out = out.maximum(out.T).tocsr()
    return out


def main():
    ap = argparse.ArgumentParser(
        description='Materialize meta-path adjacency graphs for HAN.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--root', type=str, default='data')
    ap.add_argument('--dataset', type=str, default='AMAZON')
    ap.add_argument('--paths', type=str, default='',
                    help='comma-separated meta-paths, each starting and ending at the target type, e.g. "PUP,PP,PCP" (overrides --arch)')
    ap.add_argument('--arch', type=str, default='',
                    help='arch.py key holding the path list, e.g. amazon_lp_han. Prefer this over '
                         '--paths: train_han.py reads the same key, so the built graphs and the '
                         'trained model cannot drift apart from a mistyped path string')
    ap.add_argument('--topk', type=int, default=20,
                    help='max neighbors kept per row (by walk count) at every chain step')
    ap.add_argument('--target-relation', type=str, default='U,P')
    ap.add_argument('--split-seed', type=int, default=0,
                    help='MUST match the --split-seed used for search/retrain')
    ap.add_argument('--num-neg-eval', type=int, default=99)
    args = ap.parse_args()

    if args.paths:
        paths = [p.strip() for p in args.paths.split(',') if p.strip()]
    elif args.arch:
        from arch import archs
        paths = archs[args.arch][0]
    else:
        raise SystemExit('pass --arch <arch.py key> (preferred) or --paths "PUP,..."')
    type_ids = NODE_TYPE_IDS[args.dataset]
    src_name, dst_name = args.target_relation.split(',')
    tgt = dst_name
    for p in paths:
        if not (len(p) >= 2 and p[0] == tgt and p[-1] == tgt):
            raise SystemExit(f'path "{p}" does not start and end at target type {tgt} -- '
                             f'not usable as a HAN neighbor graph. Re-run the search with '
                             f'--tgt-ended-paths.')

    print(f'Loading {args.root}/{args.dataset} ...')
    dl = data_loader(f'{args.root}/{args.dataset}')
    print(f'Building train-only split (seed {args.split_seed}) ...')
    link_split = build_link_split(dl, type_ids[src_name], type_ids[dst_name],
                                  seed=args.split_seed, num_neg_eval=args.num_neg_eval)
    base = build_base_matrices(dl, args.dataset, link_split)

    out_dir = os.path.join(args.root, args.dataset, 'metapath_graphs')
    os.makedirs(out_dir, exist_ok=True)

    for p in paths:
        print(f'Materializing {p} (topk {args.topk}) ...')
        adj = materialize_path(p, base, args.topk)
        isolated = int((np.diff(adj.indptr) == 0).sum())
        fname = os.path.join(out_dir, f'{p}_top{args.topk}_split{args.split_seed}.npz')
        sp.save_npz(fname, adj)
        print(f'  {p}: {adj.shape[0]} nodes, {adj.nnz} edges, '
              f'mean degree {adj.nnz / max(adj.shape[0], 1):.1f}, '
              f'{isolated} isolated nodes -> {fname}')

    print('Done.')


if __name__ == '__main__':
    main()
