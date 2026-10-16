#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Audit meta-path graphs for node-id degree bias.

Background: build_metapath_graphs.py prunes each row to --topk neighbors by walk
count. Walk counts are mostly tied, so the tie-break decides which neighbors
survive. The original ascending-column-index tie-break made the lowest product
ids win in every row; symmetrization then turned them into hubs. This script
makes that visible: per id-decile mean degree should be roughly flat for an
unbiased graph (modulo genuine popularity), and the biggest hubs should not
cluster at low ids.

Usage:
    python check_graph_bias.py data/AMAZON/metapath_graphs_manual
    python check_graph_bias.py data/AMAZON/metapath_graphs --paths PUP,PCP
    python check_graph_bias.py old_dir new_dir      # compare several dirs
"""

import argparse
import glob
import json
import os

import numpy as np
import scipy.sparse as sp


def stats(adj, n_hubs=100):
    adj = adj.tocsr()
    deg = np.diff(adj.indptr)
    n = len(deg)
    dec = max(n // 10, 1)
    decile_means = [float(deg[j * dec:(j + 1) * dec].mean()) for j in range(10)]
    hubs = np.argsort(-deg)[:n_hubs]
    return {
        'nodes': n, 'nnz': int(adj.nnz), 'mean_degree': float(deg.mean()),
        'isolated': int((deg == 0).sum()), 'max_degree': int(deg.max()),
        'decile_means': decile_means,
        'lowest_decile_over_rest': float(decile_means[0] / max(np.mean(decile_means[1:]), 1e-9)),
        'hub_median_id': int(np.median(hubs)), 'hub_mean_id': float(hubs.mean()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dirs', nargs='+', help='directories containing <path>_top<k>_split<s>.npz')
    ap.add_argument('--paths', type=str, default='',
                    help='comma-separated path names to restrict to (default: all .npz in dir)')
    ap.add_argument('--json', action='store_true', help='print one JSON object per graph')
    args = ap.parse_args()
    want = {p.strip() for p in args.paths.split(',') if p.strip()}

    for d in args.dirs:
        files = sorted(glob.glob(os.path.join(d, '*.npz')))
        if want:
            files = [f for f in files if os.path.basename(f).split('_')[0] in want]
        if not files:
            print(f'{d}: no matching .npz files')
            continue
        print(f'== {d}')
        print(f"{'path':8s} {'mean':>6s} {'max':>6s} {'isol':>6s} {'low/rest':>8s} "
              f"{'hubMedId':>9s}  tie-break  decile means")
        for f in files:
            name = os.path.basename(f).split('_')[0]
            s = stats(sp.load_npz(f))
            meta_file = f[:-4] + '.meta.json'
            tie = '(none)'
            if os.path.isfile(meta_file):
                with open(meta_file, encoding='utf-8') as fh:
                    m = json.load(fh)
                tie = str(m.get('tie_break'))
                if m.get('tie_seed') is not None:
                    tie += f"@{m['tie_seed']}"
            if args.json:
                print(json.dumps({'dir': d, 'path': name, 'tie_break': tie, **s}))
            else:
                print(f"{name:8s} {s['mean_degree']:6.1f} {s['max_degree']:6d} {s['isolated']:6d} "
                      f"{s['lowest_decile_over_rest']:8.2f} {s['hub_median_id']:9d}  {tie:9s}  "
                      + ' '.join(f'{x:.0f}' for x in s['decile_means']))
        print()


if __name__ == '__main__':
    main()
