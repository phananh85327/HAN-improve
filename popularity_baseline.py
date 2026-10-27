#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Popularity baseline: score(u, p) = number of TRAINING users of product p, the same for
every user. No learning, no meta-paths. With 99 uniformly sampled negatives per test edge,
a popular product beats most random negatives, so this row is the floor every model in the
table has to clear (on the rebuilt AMAZON data, split 1: test Hit@10 about 0.42).

Same split, negatives and metrics as the trainers. The result is saved in the shared blob
format, so eval_emb.py and summarize_results.py treat it like any other model: P_emb holds
the popularity (plus a seeded tie-break below 1, since most products share a small count),
U_emb is all ones, and their dot product is the score.

    python popularity_baseline.py --save-emb output/AMAZON/popularity_s1.pt
"""

import argparse
import os
import time

import numpy as np
import torch

from data_loader import data_loader
from link_data import build_link_split, data_provenance, check_data
from han_utils import (NODE_TYPE_IDS, bpr_loss, score_edges, evaluate_ranking,
                       evaluate_ranking_by_user, print_by_user,
                       evaluate_ranking_stratified, print_stratified,
                       report_twin_breakdown)


def main(args):
    start = time.time()
    src_name, dst_name = args.target_relation.split(',')
    type_ids = NODE_TYPE_IDS[args.dataset]
    dl = data_loader(f'{args.root}/{args.dataset}')
    link_split = build_link_split(dl, type_ids[src_name], type_ids[dst_name],
                                  seed=args.split_seed, num_neg_eval=args.num_neg_eval)
    data_prov = data_provenance(os.path.join(args.root, args.dataset), dl, link_split)
    check_data(data_prov, args.split_seed, args.allow_legacy_data)

    train_users = np.asarray(link_split['train_adj_src2dst'].sum(axis=0)).ravel()
    tie = np.random.default_rng(args.seed).random(train_users.shape[0]) * 0.5
    P_emb = torch.tensor(train_users + tie, dtype=torch.float32).unsqueeze(1)
    U_emb = torch.ones(link_split['num_src'], 1)

    val_edges = torch.LongTensor(link_split['val_edges'])
    val_negs = torch.LongTensor(link_split['val_negs'])
    val_loss = bpr_loss(score_edges(U_emb, P_emb, val_edges[:, 0], val_edges[:, 1]),
                        score_edges(U_emb, P_emb, val_edges[:, 0], val_negs)).item()
    test_edges = torch.LongTensor(link_split['test_edges'])
    test_negs = torch.LongTensor(link_split['test_negs'])
    pos_scores = score_edges(U_emb, P_emb, test_edges[:, 0], test_edges[:, 1])
    neg_scores = score_edges(U_emb, P_emb, test_edges[:, 0], test_negs)
    results = evaluate_ranking(pos_scores, neg_scores)

    print(f'val bpr {val_loss:.4f} (scores are raw counts, so this is not comparable with '
          f'the trained models\' val bpr)')
    print('Test ranking metrics:')
    for k, v in results.items():
        print(f'  {k}: {v:.4f}')
    print_by_user(evaluate_ranking_by_user(
        pos_scores, neg_scores, link_split['test_edges'], link_split['test_negs']))
    deg = np.asarray(link_split['train_adj_src2dst'].sum(axis=1)).ravel()
    strat = evaluate_ranking_stratified(pos_scores, neg_scores, deg[link_split['test_edges'][:, 0]])
    print('By user training degree:')
    print_stratified(strat)
    twin = report_twin_breakdown(pos_scores, neg_scores, link_split, link_split['test_edges'])

    if args.save_emb:
        os.makedirs(os.path.dirname(args.save_emb) or '.', exist_ok=True)
        torch.save({'model': 'popularity', 'dataset': args.dataset,
                    'target_relation': args.target_relation, 'paths': [],
                    'split_seed': args.split_seed, 'P_emb': P_emb, 'U_emb': U_emb,
                    'metrics': results, 'metrics_twin': twin,
                    'best_val_loss': None, 'best_epoch': None, 'epochs_run': 0,
                    'train_time_s': float(time.time() - start),
                    'data': data_prov, 'config': vars(args)}, args.save_emb)
        print(f'Saved to {args.save_emb}')
    return results


def parse_args():
    ap = argparse.ArgumentParser(
        description='Popularity baseline on the shared link-prediction protocol.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--root', type=str, default='data')
    ap.add_argument('--dataset', type=str, default='AMAZON')
    ap.add_argument('--target-relation', type=str, default='U,P')
    ap.add_argument('--split-seed', type=int, default=1,
                    help='must match the search --split-seed (the LMSPS fleets use 1)')
    ap.add_argument('--allow-legacy-data', action='store_true',
                    help='run on data without meta.json (from before the AMAZON rebuild) instead of stopping')
    ap.add_argument('--num-neg-eval', type=int, default=99)
    ap.add_argument('--seed', type=int, default=1, help='seed of the tie-break between equally popular products')
    ap.add_argument('--save-emb', type=str, default='')
    return ap.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)
    main(args)
