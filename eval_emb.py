#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Recompute test ranking metrics from SAVED embeddings -- no retraining. Every
model in the stack (HAN, GAT, metapath2vec, and LMSPS via hgb's lmsps_emb.pt)
scores user-product pairs as a dot product of its saved P_emb/U_emb, so the
saved .pt blob plus the deterministic split (--split-seed is stored in the
blob) reproduces the exact test-time scores.

Prints, per file:
  - the legacy per-edge metrics (must match the numbers printed at train time
    and the 'metrics' dict stored in the blob -- a built-in sanity check),
  - the per-user metrics (positives pooled per user, so Precision/Recall/
    NDCG/Hit genuinely differ),
  - the per-user-training-degree breakdown,
  - the duplicate-edition breakdown: test edges whose user already has a TRAIN
    edge to a product with an identical reviewer set (another edition of the
    same title) vs the rest. See link_data.product_twin_groups.
  - with --full-ranking, the same test edges ranked against every product the user
    has no observed edge to, instead of 99 sampled negatives (han_utils.evaluate_full_ranking).

    python eval_emb.py --emb output/AMAZON/han_emb_auto.pt
    python eval_emb.py --emb output/AMAZON/*.pt ../LMSPS/hgb/output/AMAZON/lmsps_emb.pt
"""

import argparse
import glob

import numpy as np
import torch

from data_loader import data_loader
from link_data import build_link_split, edges_fingerprint
from han_utils import (NODE_TYPE_IDS, score_edges, evaluate_ranking,
                       evaluate_ranking_by_user, print_by_user,
                       evaluate_ranking_stratified, print_stratified,
                       report_twin_breakdown, evaluate_full_ranking, print_full_ranking)


def main(args):
    device = 'cpu' if args.cpu or not torch.cuda.is_available() else f'cuda:{args.gpu}'
    paths = []
    for pat in args.emb:
        matches = sorted(glob.glob(pat))
        if not matches:
            print(f'WARNING: nothing matches {pat!r} -- skipped')
        paths.extend(matches)
    if not paths:
        raise SystemExit(f'no files match {args.emb}')

    splits = {}  # (dataset, relation, seed) -> link_split; rebuilt once per config
    for path in paths:
        blob = torch.load(path, map_location='cpu')
        dataset = blob['dataset']
        relation = blob.get('target_relation', 'U,P')
        seed = blob['split_seed']
        print(f'\n=== {path} ===')
        print(f"model {blob.get('model')} | dataset {dataset} | split-seed {seed} "
              f"| paths {blob.get('paths')}")
        if blob.get('config'):
            print(f"config: {blob['config']}")

        key = (dataset, relation, seed)
        if key not in splits:
            src_name, dst_name = relation.split(',')
            tids = NODE_TYPE_IDS[dataset]
            dl = data_loader(f'{args.root}/{dataset}')
            splits[key] = build_link_split(dl, tids[src_name], tids[dst_name],
                                           seed=seed, num_neg_eval=args.num_neg_eval)
            splits[key]['train_edges_hash'] = edges_fingerprint(splits[key]['train_edges'])
        link_split = splits[key]

        # the split is rebuilt from the data under --root: a blob trained on other data
        # would be scored against the wrong edges, so it is skipped instead
        trained_on = (blob.get('data') or {}).get('train_edges_hash')
        if trained_on is not None and trained_on != link_split['train_edges_hash']:
            print(f"SKIPPED: trained on other data or another split (training-edge fingerprint "
                  f"{trained_on}; {args.root}/{dataset} split {seed} gives "
                  f"{link_split['train_edges_hash']})")
            continue
        sizes = (blob['P_emb'].shape[0], blob['U_emb'].shape[0])
        if sizes != (link_split['num_dst'], link_split['num_src']):
            print(f"SKIPPED: the blob has {sizes[0]} products / {sizes[1]} users, the data "
                  f"{link_split['num_dst']} / {link_split['num_src']}: trained on other data")
            continue
        if trained_on is None:
            print('note: saved before runs recorded their data; the metrics below are valid only '
                  'if the data has not changed since (the [matches stored] check below)')

        # .detach() for old blobs where U_emb was saved as a live Parameter
        P_emb = blob['P_emb'].detach().float().to(device)
        U_emb = blob['U_emb'].detach().float().to(device)
        test_edges = torch.LongTensor(link_split['test_edges']).to(device)
        test_negs = torch.LongTensor(link_split['test_negs']).to(device)
        with torch.no_grad():
            pos_scores = score_edges(U_emb, P_emb, test_edges[:, 0], test_edges[:, 1])
            neg_scores = score_edges(U_emb, P_emb, test_edges[:, 0], test_negs)

        results = evaluate_ranking(pos_scores, neg_scores)
        stored = blob.get('metrics') or {}
        print('Test ranking metrics (per-edge, legacy):')
        for k, v in results.items():
            note = ''
            if k in stored:
                note = '  [matches stored]' if abs(stored[k] - v) < 5e-5 else \
                       f'  [STORED {stored[k]:.4f} -- split/emb mismatch?]'
            print(f'  {k}: {v:.4f}{note}')

        print_by_user(evaluate_ranking_by_user(
            pos_scores, neg_scores, link_split['test_edges'], link_split['test_negs']))

        deg = np.asarray(link_split['train_adj_src2dst'].sum(axis=1)).ravel()
        strat = evaluate_ranking_stratified(pos_scores, neg_scores,
                                            deg[link_split['test_edges'][:, 0]])
        print('By user training degree:')
        print_stratified(strat)
        report_twin_breakdown(pos_scores, neg_scores, link_split, link_split['test_edges'])
        if args.full_ranking:
            with torch.no_grad():
                print_full_ranking(evaluate_full_ranking(U_emb, P_emb, link_split['test_edges'],
                                                         link_split['full_adj_src2dst']))


def parse_args():
    ap = argparse.ArgumentParser(
        description='Recompute test metrics from saved *_emb.pt files (no retraining).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--emb', type=str, nargs='+', required=True,
                    help='one or more saved embedding files (globs ok)')
    ap.add_argument('--root', type=str, default='data')
    ap.add_argument('--num-neg-eval', type=int, default=99,
                    help='must match the value used at train time')
    ap.add_argument('--full-ranking', action='store_true',
                    help='also rank each test edge against every product the user has no observed '
                         'edge to (a few seconds per model on a GPU, a minute or two on a CPU)')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--cpu', action='store_true')
    return ap.parse_args()


if __name__ == '__main__':
    main(parse_args())
