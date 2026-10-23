#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Audit duplicate-edition products in the AMAZON U-P data and quantify the
evaluation shortcut they create.

Amazon shows one review pool across the editions/formats of a title (hardcover,
paperback, audio; full-screen / widescreen / platinum DVD ...). The SNAP
amazon-meta crawl copies that pool into every edition's record, and the
preprocessing (preprocessing/AmazonGraphFilter.cs) never merges editions. The
result: thousands of product pairs with byte-identical reviewer sets. For a
held-out edge (u, p) whose user already has a TRAIN edge to a twin of p, the
positive is predictable from a single P-U-P hop, so any co-reviewed meta-path
scores it trivially.

This script prints, for one dataset root and split seed:
  - how many products share an identical reviewer set with another product,
    group-size histogram, share of U-P edges in such groups;
  - the largest groups with ASIN and title (if --products-csv is given);
  - the share of val and test edges that have a twin edge in TRAIN
    (exactly the flag reported by the trainers / eval_emb.py);
  - optionally (--jaccard) the same share for near-duplicates at several
    Jaccard thresholds, computed per test edge against the user's train items
    (slower: a few minutes).

Usage:
    python check_edition_twins.py --split-seed 1
    python check_edition_twins.py --split-seed 1 --products-csv output/AMAZON/Filtered_Output/filtered_products.csv --jaccard
"""

import argparse
import collections
import csv
import os
import time

import numpy as np

from data_loader import data_loader
from link_data import build_link_split, twin_in_train
from han_utils import NODE_TYPE_IDS


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', type=str, default='data')
    ap.add_argument('--dataset', type=str, default='AMAZON')
    ap.add_argument('--target-relation', type=str, default='U,P')
    ap.add_argument('--split-seed', type=int, default=0)
    ap.add_argument('--num-neg-eval', type=int, default=99)
    ap.add_argument('--products-csv', type=str, default='',
                    help='filtered_products.csv to print titles of the largest groups')
    ap.add_argument('--top', type=int, default=8, help='largest groups to list')
    ap.add_argument('--jaccard', action='store_true',
                    help='also report near-duplicate twins at Jaccard 0.9/0.5/0.3 (slow)')
    args = ap.parse_args()

    src_name, dst_name = args.target_relation.split(',')
    tids = NODE_TYPE_IDS[args.dataset]
    t0 = time.time()
    dl = data_loader(f'{args.root}/{args.dataset}')
    ls = build_link_split(dl, tids[src_name], tids[dst_name],
                          seed=args.split_seed, num_neg_eval=args.num_neg_eval)
    print(f'loaded {args.root}/{args.dataset}, split seed {args.split_seed} '
          f'({time.time() - t0:.0f}s): train {len(ls["train_edges"])} '
          f'val {len(ls["val_edges"])} test {len(ls["test_edges"])}')

    groups = ls['dst_twin_group']
    full = ls['full_adj_src2dst'].tocsc()
    deg = np.diff(full.indptr)  # reviews per product
    sizes = np.bincount(groups)
    in_group = sizes[groups] > 1
    n_groups = int((sizes > 1).sum())
    print(f'\nproducts: {len(groups)}; with an identical reviewer set elsewhere: '
          f'{int(in_group.sum())} ({100 * in_group.mean():.1f}%) in {n_groups} groups')
    hist = collections.Counter(sizes[sizes > 1].tolist())
    print('group size histogram:', ' '.join(f'{k}:{v}' for k, v in sorted(hist.items())))
    e_in = int(deg[in_group].sum())
    print(f'U-P edges on such products: {e_in} of {int(deg.sum())} ({100 * e_in / deg.sum():.1f}%)')

    # largest groups by review count
    names = {}
    if args.products_csv and os.path.isfile(args.products_csv):
        asin_of = {}
        shift = dl.nodes['shift'][tids[dst_name]]
        with open(os.path.join(args.root, args.dataset, 'node.dat'), encoding='utf-8',
                  errors='replace') as fh:
            for line in fh:
                parts = line.split('\t')
                if len(parts) >= 3 and int(parts[2]) == tids[dst_name]:
                    asin_of[int(parts[0]) - shift] = parts[1].strip()
        titles = {}
        with open(args.products_csv, encoding='utf-8', errors='replace', newline='') as fh:
            for row in csv.DictReader(fh):
                titles[row.get('ASIN', '')] = (row.get('Title', '')[:60], row.get('Group', ''))
        names = {pid: (asin, *titles.get(asin, ('?', '?'))) for pid, asin in asin_of.items()}
    members = collections.defaultdict(list)
    for pid, g in enumerate(groups.tolist()):
        if sizes[g] > 1:
            members[g].append(pid)
    biggest = sorted(members.values(), key=lambda m: -deg[m[0]])[:args.top]
    print(f'\nlargest {len(biggest)} groups (reviews per member, members):')
    for m in biggest:
        print(f'  {int(deg[m[0]])} reviews x {len(m)} products: ' +
              '; '.join(f'{p} {names[p][0]} "{names[p][1]}" [{names[p][2]}]' if p in names else str(p)
                        for p in m[:6]) + (' ...' if len(m) > 6 else ''))

    for split in ('val', 'test'):
        flag = twin_in_train(ls, ls[f'{split}_edges'])
        print(f'\n{split} edges whose user has a TRAIN edge to an identical-reviewer-set twin: '
              f'{int(flag.sum())} of {len(flag)} ({100 * flag.mean():.1f}%)')

    if args.jaccard:
        t1 = time.time()
        A = ls['train_adj_src2dst'].tocsr()
        indptr, indices = full.indptr, full.indices
        sets = [set(indices[indptr[j]:indptr[j + 1]].tolist()) for j in range(full.shape[1])]
        test = ls['test_edges']
        maxj = np.zeros(len(test))
        for k, (u, p) in enumerate(test.tolist()):
            sp_ = sets[p]
            best = 0.0
            for q in A.indices[A.indptr[u]:A.indptr[u + 1]].tolist():
                sq = sets[q]
                inter = len(sp_ & sq)
                j = inter / (len(sp_) + len(sq) - inter)
                if j > best:
                    best = j
                    if best >= 0.99:
                        break
            maxj[k] = best
        print(f'\nnear-duplicate twins (max Jaccard of the test item\'s reviewer set vs the '
              f'user\'s train items; {time.time() - t1:.0f}s):')
        for thr in (0.99, 0.9, 0.5, 0.3):
            print(f'  test edges with a train twin at Jaccard >= {thr}: {100 * (maxj >= thr).mean():.1f}%')


if __name__ == '__main__':
    main()
