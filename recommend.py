#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Query interface over embeddings saved by train_han.py / main_path.py --save-emb:
input a product ASIN (or a user id), get a ranked top-k list of products.

    # products similar to an ASIN (item-to-item, cosine over product embeddings)
    python recommend.py --emb output/AMAZON/han_emb.pt --asin 0827229534 \
        --products-csv <dir>/filtered_products.csv --topk 10

    # a user's personal top-k (dot product -- the scorer BPR training optimized)
    python recommend.py --emb output/AMAZON/han_emb.pt --user A2EK9A6BFHW2AU \
        --users-csv <dir>/filtered_users.csv --exclude-seen

    # live demo loop
    python recommend.py --emb output/AMAZON/han_emb.pt --interactive \
        --products-csv <dir>/filtered_products.csv

Notes
-----
* node.dat's name column holds the ASIN for products, so ASIN lookup needs no
  extra file. For users it holds the DISPLAY NAME, not the customer id, so
  customer-id lookup requires --users-csv (filtered_users.csv row order equals
  the node-id order dat_generator assigned). --user also accepts a raw numeric
  node index without the csv.
* Similarity search runs on CPU numpy; one query is a [num_products, 512]
  matrix-vector product -- instantaneous.
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch

PRODUCT_TYPE = 0
CATEGORY_TYPE = 1
USER_TYPE = 2
REL_USER_PRODUCT = 6


def load_node_maps(root, dataset):
    """ASIN<->product-id maps and per-type node counts from node.dat."""
    asin_to_pid = {}
    pid_to_asin = []
    counts = {PRODUCT_TYPE: 0, CATEGORY_TYPE: 0, USER_TYPE: 0}
    with open(os.path.join(root, dataset, 'node.dat'), 'r', encoding='utf-8') as fh:
        for line in fh:
            th = line.rstrip('\n').split('\t')
            ntype = int(th[2])
            counts[ntype] = counts.get(ntype, 0) + 1
            if ntype == PRODUCT_TYPE:
                asin_to_pid[th[1]] = int(th[0])
                pid_to_asin.append(th[1])
    return asin_to_pid, pid_to_asin, counts


def load_titles(products_csv):
    titles = {}
    with open(products_csv, 'r', encoding='utf-8-sig', newline='') as fh:
        for row in csv.DictReader(fh):
            asin = (row.get('ASIN') or '').strip()
            if asin:
                titles[asin] = (row.get('Title') or '').strip()
    return titles


def load_user_ids(users_csv):
    """customer-id -> local user index; filtered_users.csv row order IS the node-id
    order dat_generator assigned (first appearance in the surviving review set)."""
    cid_to_uid = {}
    with open(users_csv, 'r', encoding='utf-8-sig', newline='') as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            cid_to_uid[(row.get('ID') or '').strip()] = i
    return cid_to_uid


def load_seen(root, dataset, num_products, num_categories):
    """user local id -> set of reviewed product ids, from link.dat relation 6."""
    seen = {}
    user_shift = num_products + num_categories
    with open(os.path.join(root, dataset, 'link.dat'), 'r', encoding='utf-8') as fh:
        for line in fh:
            th = line.split('\t')
            if int(th[2]) != REL_USER_PRODUCT:
                continue
            uid = int(th[0]) - user_shift
            seen.setdefault(uid, set()).add(int(th[1]))
    return seen


def topk_products(scores, k, exclude):
    scores = scores.copy()
    for pid in exclude:
        scores[pid] = -np.inf
    k = min(k, scores.shape[0])
    idx = np.argpartition(scores, -k)[-k:]
    return idx[np.argsort(scores[idx])[::-1]]


def print_recs(pids, scores, pid_to_asin, titles):
    for rank, pid in enumerate(pids, 1):
        asin = pid_to_asin[pid]
        title = titles.get(asin, '') if titles else ''
        if len(title) > 70:
            title = title[:67] + '...'
        print(f'  {rank:>2}. {scores[pid]:>8.4f}  {asin:<12} {title}')


def recommend_for_product(asin, k, P_norm, asin_to_pid, pid_to_asin, titles):
    pid = asin_to_pid.get(asin)
    if pid is None:
        print(f'  ASIN "{asin}" not found in node.dat (was it filtered out?)')
        return
    scores = P_norm @ P_norm[pid]
    print(f'Products similar to {asin}'
          + (f' ({titles.get(asin, "")[:70]})' if titles else '') + ':')
    print_recs(topk_products(scores, k, {pid}), scores, pid_to_asin, titles)


def recommend_for_user(key, k, P_emb, U_emb, cid_to_uid, pid_to_asin, titles, seen):
    if key.isdigit() and int(key) < U_emb.shape[0] and key not in cid_to_uid:
        uid = int(key)
    else:
        uid = cid_to_uid.get(key)
        if uid is None:
            print(f'  user "{key}" not found (pass --users-csv, or use a numeric node index)')
            return
    scores = P_emb @ U_emb[uid]
    exclude = seen.get(uid, set()) if seen is not None else set()
    print(f'Top products for user {key} (excluding {len(exclude)} seen):' if seen is not None
          else f'Top products for user {key}:')
    print_recs(topk_products(scores, k, exclude), scores, pid_to_asin, titles)


def main():
    ap = argparse.ArgumentParser(
        description='Top-k product recommendations from saved embeddings.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--emb', required=True, help='embedding file saved via --save-emb')
    ap.add_argument('--root', type=str, default='data')
    ap.add_argument('--dataset', type=str, default='AMAZON')
    ap.add_argument('--asin', type=str, default='', help='query product ASIN')
    ap.add_argument('--user', type=str, default='',
                    help='query customer id (needs --users-csv) or numeric node index')
    ap.add_argument('--interactive', action='store_true',
                    help='loop reading queries from stdin')
    ap.add_argument('--topk', type=int, default=10)
    ap.add_argument('--products-csv', type=str, default='',
                    help='filtered_products.csv for human-readable titles')
    ap.add_argument('--users-csv', type=str, default='',
                    help='filtered_users.csv for customer-id lookup')
    ap.add_argument('--exclude-seen', action='store_true',
                    help='user mode: hide products the user already reviewed')
    args = ap.parse_args()
    if not (args.asin or args.user or args.interactive):
        ap.error('give --asin, --user, or --interactive')

    blob = torch.load(args.emb, map_location='cpu')
    # .detach(): embedding-tower U_emb is the raw nn.Embedding.weight Parameter,
    # which keeps requires_grad=True through torch.save -- .numpy() refuses that
    P_emb = blob['P_emb'].detach().float().numpy()
    U_emb = blob['U_emb'].detach().float().numpy()
    print(f"Loaded {blob['model']} embeddings: {P_emb.shape[0]} products, "
          f"{U_emb.shape[0]} users, paths {blob.get('paths')}")

    asin_to_pid, pid_to_asin, counts = load_node_maps(args.root, args.dataset)
    assert len(pid_to_asin) == P_emb.shape[0], \
        'node.dat product count does not match the embedding file -- wrong dataset?'

    # cosine for item-to-item: normalize once (guard all-zero rows)
    norms = np.linalg.norm(P_emb, axis=1, keepdims=True)
    P_norm = P_emb / np.clip(norms, 1e-12, None)

    titles = load_titles(args.products_csv) if args.products_csv else {}
    cid_to_uid = load_user_ids(args.users_csv) if args.users_csv else {}
    seen = (load_seen(args.root, args.dataset, counts[PRODUCT_TYPE], counts[CATEGORY_TYPE])
            if args.exclude_seen else None)

    if args.asin:
        recommend_for_product(args.asin, args.topk, P_norm, asin_to_pid, pid_to_asin, titles)
    if args.user:
        recommend_for_user(args.user, args.topk, P_emb, U_emb, cid_to_uid,
                           pid_to_asin, titles, seen)
    if args.interactive:
        print('\nInteractive mode: enter an ASIN, or "u <customer-id-or-index>" '
              'for a user query. Blank line quits.')
        while True:
            try:
                line = input('> ').strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                break
            if line.lower().startswith('u '):
                recommend_for_user(line[2:].strip(), args.topk, P_emb, U_emb,
                                   cid_to_uid, pid_to_asin, titles, seen)
            else:
                recommend_for_product(line, args.topk, P_norm, asin_to_pid,
                                      pid_to_asin, titles)


if __name__ == '__main__':
    main()