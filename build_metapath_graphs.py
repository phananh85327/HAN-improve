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
NOTE: this is a deviation from the HAN paper, which uses ALL meta-path based
neighbors (the full adjacency product, Sec. 3 Def. 3.3); top-k is a scaling choice.

Tie-break rule: walk-count matrices are full of ties (every PP entry is 1.0, most
PUP/PCP counts are 1-2), so WHICH tied neighbors survive top-k matters. The original
implementation broke ties by ascending column index, which made the lowest product
ids win in every row; after symmetrization those ids became hubs (measured on the
split-0 manual graphs: the lowest-id tenth of products had mean degree 87 in PUP and
69 in PCP vs 24-42 for every other tenth). Default is now --tie-break random with a
fixed --tie-seed (reproducible, id-neutral); --tie-break id keeps the old behavior
for comparison. Each .npz gets a .meta.json sidecar recording the mode so
train_han.py can warn when it loads a legacy graph.

Usage (from han-recsys/):
    python build_metapath_graphs.py --dataset AMAZON --arch amazon_lp_manual_5 \
        --topk 20 --split-seed 1 --target-relation "U,P"

Output: data/<dataset>/metapath_graphs/<path>_top<k>_split<seed>.npz (scipy CSR,
binary, no self-loops -- train_han.py adds self-loops when building DGL graphs)
plus <same stem>.meta.json (tie_break, tie_seed, topk, split_seed, nnz, isolated,
num_train_edges, train_edges_hash). The file name does not change with the data, so
train_han.py refuses a graph whose training-edge fingerprint is not the current one,
and --skip-existing rebuilds such a graph instead of skipping it.
"""

import argparse
import json
import os

import numpy as np
import scipy.sparse as sp

from data_loader import data_loader
from link_data import build_link_split, data_provenance, check_data

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


def topk_per_row(mat, k, rng=None):
    """Keep the k largest-valued entries per row of a CSR matrix.

    Primary key is always the value (walk count), descending. Ties:
      rng is None -> ascending column index (legacy --tie-break id). Deterministic,
                     but systematically favors low node ids: with mostly-tied walk
                     counts the same low-id columns survive in every row, and
                     symmetrization then turns them into hubs.
      rng given   -> a uniform random key drawn from rng (--tie-break random).
                     Reproducible for a fixed seed because rows are visited in a
                     fixed order, and id-neutral.
    np.argpartition is avoided in both modes: it breaks ties by memory layout, so
    the kept set would silently change across scipy versions or re-runs."""
    mat = mat.tocsr()
    indptr, indices, data = mat.indptr, mat.indices, mat.data
    tie_key = indices if rng is None else rng.random(len(data))
    keep_idx = []
    for i in range(mat.shape[0]):
        s, e = indptr[i], indptr[i + 1]
        if e - s > k:
            # primary key: value descending; tie-break: tie_key ascending
            sel = np.lexsort((tie_key[s:e], -data[s:e]))[:k] + s
            keep_idx.append(sel)
        elif e > s:
            keep_idx.append(np.arange(s, e))
    if not keep_idx:
        return sp.csr_matrix(mat.shape, dtype=np.float32)
    keep = np.concatenate(keep_idx)
    # rebuild row indices from indptr membership
    row_ids = np.searchsorted(indptr, keep, side='right') - 1
    return sp.csr_matrix((data[keep], (row_ids, indices[keep])), shape=mat.shape)


def materialize_path(path, base, topk, chunk_rows=4096, rng=None):
    """Walk-count matrix for `path` (e.g. 'PUP'): chain-multiply the step matrices
    in row chunks, pruning to top-k per row after every step to bound memory.
    rng: numpy Generator for random tie-breaking in topk_per_row (None = by id)."""
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
            m = topk_per_row(m, topk, rng)
        chunks.append(topk_per_row(m, topk, rng))
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
    ap.add_argument('--split-seed', type=int, default=1,
                    help='MUST match the --split-seed of the search (the LMSPS fleets use 1)')
    ap.add_argument('--allow-legacy-data', action='store_true',
                    help='run on data without meta.json (from before the AMAZON rebuild) instead of stopping')
    ap.add_argument('--num-neg-eval', type=int, default=99)
    ap.add_argument('--tie-break', type=str, default='random', choices=['random', 'id'],
                    help='how top-k pruning resolves equal walk counts. "random": seeded '
                         'uniform key (id-neutral, reproducible). "id": legacy ascending '
                         'column index -- biased, low product ids become hubs after '
                         'symmetrization; kept only for comparison with old graphs')
    ap.add_argument('--tie-seed', type=int, default=None,
                    help='seed for --tie-break random (default: same as --split-seed)')
    ap.add_argument('--graph-dir', type=str, default='',
                    help='output directory (default <root>/<dataset>/metapath_graphs). Files are '
                         'named per path, so several arches can share one directory; pass the same '
                         'value to train_han.py --graph-dir')
    ap.add_argument('--skip-existing', action='store_true',
                    help='skip a path whose .npz and .meta.json already exist in the output '
                         'directory and were built with the same settings from the same training '
                         'edges, so arches sharing paths can be built into one directory without '
                         'redoing work. A graph built from other data is rebuilt')
    args = ap.parse_args()
    if args.tie_seed is None:
        args.tie_seed = args.split_seed

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
    prov = data_provenance(os.path.join(args.root, args.dataset), dl, link_split)
    check_data(prov, args.split_seed, args.allow_legacy_data)
    base = build_base_matrices(dl, args.dataset, link_split)

    out_dir = args.graph_dir or os.path.join(args.root, args.dataset, 'metapath_graphs')
    os.makedirs(out_dir, exist_ok=True)
    # what a graph depends on besides its path; --skip-existing keeps a graph only if all match
    settings = {'topk': args.topk, 'split_seed': args.split_seed, 'tie_break': args.tie_break,
                'tie_seed': args.tie_seed if args.tie_break == 'random' else None,
                'num_train_edges': len(link_split['train_edges']),
                'train_edges_hash': prov['train_edges_hash']}

    for p in paths:
        stem = os.path.join(out_dir, f'{p}_top{args.topk}_split{args.split_seed}')
        fname = stem + '.npz'
        if args.skip_existing and os.path.isfile(fname) and os.path.isfile(stem + '.meta.json'):
            with open(stem + '.meta.json', encoding='utf-8') as f:
                old = json.load(f)
            if all(old.get(k) == v for k, v in settings.items()):
                print(f'Skipping {p}: {fname} exists')
                continue
            print(f'Rebuilding {p}: {fname} was built from other data or with other settings')
        print(f'Materializing {p} (topk {args.topk}, tie-break {args.tie_break}) ...')
        # one generator per path, re-seeded from the same seed: a path's graph does
        # not depend on which other paths were built in the same invocation
        rng = (np.random.default_rng(args.tie_seed) if args.tie_break == 'random'
               else None)
        adj = materialize_path(p, base, args.topk, rng=rng)
        isolated = int((np.diff(adj.indptr) == 0).sum())
        sp.save_npz(fname, adj)
        meta = {'path': p, **settings,
                'num_nodes': int(adj.shape[0]), 'nnz': int(adj.nnz), 'isolated': isolated}
        with open(stem + '.meta.json', 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)
        print(f'  {p}: {adj.shape[0]} nodes, {adj.nnz} edges, '
              f'mean degree {adj.nnz / max(adj.shape[0], 1):.1f}, '
              f'{isolated} isolated nodes -> {fname}')

    print('Done.')


if __name__ == '__main__':
    main()
