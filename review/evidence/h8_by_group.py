"""HAN review H8, optional check on your PC: where do the runs differ?

For each saved run (.pt), with the split the run was trained on:
1. Hit@10 under the training-time protocol (99 sampled negatives) by the product group of the
   test edge's product (label.dat / label.dat.test), and by whether P-C-C-P reaches that product
   (the LMSPS sets' category paths reach nothing for the other ~81%: LMSPS review part 4)
2. the full-ranking top 10 of a fixed sample of test users, excluding their training items: how
   popular the recommended products are (mean training degree, share from the 1% most-bought
   products) and how many distinct products they cover

    python review/evidence/h8_by_group.py output/AMAZON/v3/han_amazon_lp_v3_f4_a1_s1.pt output/AMAZON/v3/han_amazon_lp_manual_5_s1.pt
    python review/evidence/h8_by_group.py output/AMAZON/v3/*_s1.pt

Seconds per run on a GPU, a minute or two on a CPU. The overall Hit@10 it recomputes must equal
the run's own; a mismatch is printed.
"""
import argparse
import glob
import os
import sys

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from data_loader import data_loader  # noqa: E402
from han_utils import NODE_TYPE_IDS, score_edges  # noqa: E402
from link_data import build_link_split, edges_fingerprint  # noqa: E402

GROUPS = {0: 'Book', 1: 'DVD', 2: 'Music', 3: 'Video'}


def product_groups(d, n_p):
    g = np.full(n_p, -1)
    for name in ('label.dat', 'label.dat.test'):
        for line in open(os.path.join(d, name), encoding='utf-8'):
            f = line.rstrip('\n').split('\t')
            if int(f[2]) == 0:
                g[int(f[0])] = int(f[3])
    return g


def pccp_reached(dl, tids):
    """Products with >= 1 P-C-C-P endpoint, as part4_category_paths.py counts them."""
    s, n = dl.nodes['shift'], dl.nodes['count']
    p, c = tids['P'], tids['C']
    rels = {}
    for r, (h, t) in dl.links['meta'].items():
        m = dl.links['data'][r].tocsr()[s[h]:s[h] + n[h], s[t]:s[t] + n[t]]
        rels.setdefault((h, t), []).append(m)
    pc = sum(rels[(p, c)]).sign().tocsr()
    cc = sum(rels[(c, c)])
    cc = (cc + cc.T).sign().tolil()
    cc.setdiag(0)
    cc = cc.tocsr()
    cc.eliminate_zeros()
    v = np.ones(n[p])
    for m in (pc.T.tocsr(), cc, pc):
        v = (m @ v > 0).astype(float)
    return v > 0


def main(args):
    paths = [p for g in args.emb for p in sorted(glob.glob(g))]
    dev = 'cpu' if args.cpu or not torch.cuda.is_available() else f'cuda:{args.gpu}'
    cache = {}
    for path in paths:
        try:
            blob = torch.load(path, map_location='cpu', weights_only=False)
        except TypeError:  # torch < 1.13: no weights_only argument
            blob = torch.load(path, map_location='cpu')
        key = (blob['dataset'], blob['split_seed'])
        if key not in cache:
            d = os.path.join(args.root, key[0])
            dl = data_loader(d)
            tids = NODE_TYPE_IDS[key[0]]
            ls = build_link_split(dl, tids['U'], tids['P'], seed=key[1])
            deg = np.asarray(ls['train_adj_src2dst'].sum(axis=0)).ravel()
            cache[key] = (ls, edges_fingerprint(ls['train_edges']), product_groups(d, ls['num_dst']),
                          pccp_reached(dl, tids), deg)
            g, reach = cache[key][2], cache[key][3]
            e = ls['test_edges'][:, 1]
            print(f'data {d}, split {key[1]}: test edges by product group '
                  + ', '.join(f'{GROUPS[k]} {np.mean(g[e] == k):.1%}' for k in GROUPS)
                  + f'; P-C-C-P reaches {reach.mean():.1%} of products, {reach[e].mean():.1%} of test edges\n')
        ls, fp, group, reach, deg = cache[key]
        if (blob.get('data') or {}).get('train_edges_hash') not in (None, fp):
            print(f'{os.path.basename(path)}: trained on other data, skipped\n')
            continue
        U = blob['U_emb'].float().to(dev)
        P = blob['P_emb'].float().to(dev)
        te = torch.as_tensor(ls['test_edges'], device=dev)
        with torch.no_grad():
            pos = score_edges(U, P, te[:, 0], te[:, 1])
            neg = score_edges(U, P, te[:, 0], torch.as_tensor(ls['test_negs'], device=dev))
        rank = (neg >= pos[:, None]).sum(1).cpu().numpy() + 1     # ties pessimistic, as in training
        hit = rank <= 10
        own = (blob.get('metrics') or {}).get('Hit@10')
        name = os.path.basename(path)
        print(f'{name}: Hit@10 {hit.mean():.4f}' + ('' if own is None or abs(own - hit.mean()) < 1e-4
                                                     else f'  MISMATCH: the run reported {own:.4f}'))
        e = ls['test_edges'][:, 1]
        print('  by product group: ' + '  '.join(f'{GROUPS[k]} {hit[group[e] == k].mean():.4f}' for k in GROUPS))
        print(f'  P-C-C-P reaches the product: {hit[reach[e]].mean():.4f}   does not: {hit[~reach[e]].mean():.4f}')

        # full-ranking top 10 of a fixed user sample, training items excluded
        users = np.unique(ls['test_edges'][:, 0])
        users = np.random.default_rng(0).choice(users, min(args.users, len(users)), replace=False)
        train = ls['train_adj_src2dst'].tocsr()
        top = []
        with torch.no_grad():
            for i in range(0, len(users), 1024):
                u = users[i:i + 1024]
                s = U[torch.as_tensor(u, device=dev)] @ P.T
                seen = train[u].tocoo()
                s[torch.as_tensor(seen.row, device=dev), torch.as_tensor(seen.col, device=dev)] = float('-inf')
                top.append(s.topk(10, dim=1).indices.cpu().numpy())
        top = np.concatenate(top).ravel()
        head = deg >= np.quantile(deg, 0.99)
        print(f'  full-ranking top 10 of {len(users)} test users: mean training degree {deg[top].mean():.1f} '
              f'(all products {deg.mean():.1f}), from the top 1% most-bought {head[top].mean():.1%}, '
              f'distinct products {len(np.unique(top))}\n')


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('emb', nargs='+', help='saved runs (.pt, globs ok)')
    ap.add_argument('--root', default='data')
    ap.add_argument('--users', type=int, default=5000, help='test users sampled for the full-ranking top 10')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--cpu', action='store_true')
    return ap.parse_args()


if __name__ == '__main__':
    main(parse_args())
