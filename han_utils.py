#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Self-contained helpers for the standalone HAN recommender -- extracted from the
LMSPS repo's hgb/utils.py so this repo needs neither dgl, torch_sparse, nor
sparse_tools. Pure torch/numpy.
"""

import random

import numpy as np
import torch
import torch.nn.functional as F

# node-type letter -> HGB type id, per dataset (matches the .dat layout)
NODE_TYPE_IDS = {'ACM': {'P': 0, 'A': 1, 'C': 2, 'K': 3},
                 'AMAZON': {'P': 0, 'C': 1, 'U': 2}}


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_n_params(model):
    return sum(p.numel() for p in model.parameters())


def bpr_loss(pos_scores, neg_scores):
    """BPR pairwise ranking loss. pos_scores: [N]; neg_scores: [N] or [N, K]
    (broadcast against pos_scores when K sampled negatives per positive)."""
    if neg_scores.dim() > pos_scores.dim():
        pos_scores = pos_scores.unsqueeze(-1)
    return -F.logsigmoid(pos_scores - neg_scores).mean()


def score_edges(src_emb, dst_emb, src_idx, dst_idx, chunk_size=1024):
    """Dot-product score for (src_idx[i], dst_idx[i]) pairs. dst_idx may be
    [N] (one dst per src) or [N, K] (K candidates per src, e.g. sampled negatives).
    The [N, K] case is scored in row chunks: gathering dst_emb[dst_idx] in one
    shot materializes an [N, K, hidden] tensor -- tens of GB at Amazon scale."""
    if dst_idx.dim() == 1:
        return (src_emb[src_idx] * dst_emb[dst_idx]).sum(-1)
    out = torch.empty(dst_idx.shape, dtype=src_emb.dtype, device=src_emb.device)
    for start in range(0, dst_idx.size(0), chunk_size):
        end = min(start + chunk_size, dst_idx.size(0))
        s = src_emb[src_idx[start:end]].unsqueeze(1)
        out[start:end] = (s * dst_emb[dst_idx[start:end]]).sum(-1)
    return out


def evaluate_ranking(pos_scores, neg_scores, ks=(5, 10, 20)):
    """pos_scores: [N], neg_scores: [N, K] sampled-negative scores per positive
    edge. Ranks the single positive among its K negatives and computes
    Precision@k/Recall@k/NDCG@k/Hit@k, averaged over all N edges. With exactly
    one relevant item per edge, Hit@k is the binary top-k indicator and
    Recall@k coincides with it; Precision@k = Hit@k / k."""
    num_candidates = neg_scores.size(1) + 1
    all_scores = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
    ranks = (all_scores >= all_scores[:, :1]).sum(dim=1).float()

    results = {}
    for k in ks:
        k = min(k, num_candidates)
        hit = (ranks <= k).float()
        ndcg = hit / torch.log2(ranks + 1)
        results[f'Precision@{k}'] = (hit / k).mean().item()
        results[f'Recall@{k}'] = hit.mean().item()
        results[f'NDCG@{k}'] = ndcg.mean().item()
        results[f'Hit@{k}'] = hit.mean().item()
    return results

def evaluate_ranking_by_user(pos_scores, neg_scores, test_edges, test_negs,
                             ks=(5, 10, 20)):
    """Per-USER ranking metrics: all test edges of a user are ranked together
    against the union of that user's sampled negatives, so the four metrics
    genuinely differ (unlike the per-edge protocol, where a single positive
    forces Recall@k == Hit@k and Precision@k == Hit@k / k).

    For user u with P_u test positives and candidate set C_u = positives +
    deduplicated negatives (negatives are sampled per edge, so the same item can
    appear for several of u's edges -- scored identically, kept once):
      Precision@k = |top-k ∩ P_u| / k
      Recall@k    = |top-k ∩ P_u| / |P_u|
      NDCG@k      = binary-gain DCG over top-k / IDCG(min(|P_u|, k))
      Hit@k       = 1 if any positive in top-k
    averaged over users (macro), so heavy users get ONE vote here versus one
    vote per edge in evaluate_ranking. Ties rank pessimistically (same rule as
    evaluate_ranking). test_edges: [N, 2] (src, dst); test_negs: [N, K] dst ids.
    """
    pos = pos_scores.detach().cpu().numpy().astype(np.float64)
    neg = neg_scores.detach().cpu().numpy().astype(np.float64)
    users = np.asarray(test_edges)[:, 0]
    pos_items = np.asarray(test_edges)[:, 1]
    neg_items = np.asarray(test_negs)

    order = np.argsort(users, kind='stable')
    users_s, pos_s, pitems_s = users[order], pos[order], pos_items[order]
    neg_s, nitems_s = neg[order], neg_items[order]
    starts = np.flatnonzero(np.r_[True, users_s[1:] != users_s[:-1]])
    bounds = np.r_[starts, users_s.size]

    idcg_table = np.cumsum(1.0 / np.log2(np.arange(2, max(ks) + 2)))
    sums = {k: np.zeros(4) for k in ks}  # precision, recall, ndcg, hit
    n_users = starts.size
    for i in range(n_users):
        lo, hi = bounds[i], bounds[i + 1]
        p_sc, p_it = pos_s[lo:hi], pitems_s[lo:hi]
        n_it = nitems_s[lo:hi].ravel()
        n_sc = neg_s[lo:hi].ravel()
        uniq, first = np.unique(n_it, return_index=True)
        keep = ~np.isin(uniq, p_it)
        cand = np.concatenate([p_sc, n_sc[first[keep]]])
        ranks = (cand[None, :] >= p_sc[:, None]).sum(1)  # >= counts self
        npos = hi - lo
        for k in ks:
            in_top = ranks <= k
            h = int(in_top.sum())
            dcg = (1.0 / np.log2(ranks[in_top] + 1)).sum()
            idcg = idcg_table[min(npos, k) - 1]
            sums[k] += (h / k, h / npos, dcg / idcg, 1.0 if h else 0.0)

    results = {'n_users': n_users}
    for k in ks:
        prec, rec, ndcg, hit = sums[k] / n_users
        results[f'Precision@{k}'] = prec
        results[f'Recall@{k}'] = rec
        results[f'NDCG@{k}'] = ndcg
        results[f'Hit@{k}'] = hit
    return results


def print_by_user(res):
    res = dict(res)
    n = res.pop('n_users')
    print(f'Per-user ranking metrics ({n} users, positives pooled per user):')
    for k, v in res.items():
        print(f'  {k}: {v:.4f}')


def evaluate_ranking_stratified(pos_scores, neg_scores, train_degrees,
                                buckets=((0, 0), (1, 5), (6, 20), (21, None)),
                                ks=(10,)):
    """Ranking metrics per USER-training-degree bucket (how many train
    interactions the user behind each test edge has). Shows WHERE a model wins:
    free-embedding / graph-aggregation user sides excel on well-observed users
    but have no signal for near-cold ones, while feature towers degrade more
    gracefully. train_degrees: [N] array aligned with pos_scores rows."""
    out = {}
    deg = np.asarray(train_degrees)
    for lo, hi in buckets:
        mask = (deg >= lo) if hi is None else ((deg >= lo) & (deg <= hi))
        if hi is None:
            name = f'deg>={lo}'
        elif lo == hi:
            name = f'deg={lo}'
        else:
            name = f'deg {lo}-{hi}'
        n = int(mask.sum())
        if n == 0:
            continue
        m = torch.from_numpy(mask).to(pos_scores.device)
        out[name] = {'n_edges': n,
                     **evaluate_ranking(pos_scores[m], neg_scores[m], ks=ks)}
    return out


def evaluate_ranking_by_flag(pos_scores, neg_scores, flag,
                             names=('no twin in train', 'twin in train'), ks=(10,)):
    """Ranking metrics split by a boolean per-edge flag (e.g. link_data.twin_in_train:
    does the user already have a TRAIN edge to a duplicate edition of the test item?).
    Same output shape as evaluate_ranking_stratified, so print_stratified applies."""
    flag = np.asarray(flag, dtype=bool)
    out = {}
    for value, name in ((False, names[0]), (True, names[1])):
        mask = flag == value
        n = int(mask.sum())
        if n == 0:
            continue
        m = torch.from_numpy(mask).to(pos_scores.device)
        out[name] = {'n_edges': n,
                     **evaluate_ranking(pos_scores[m], neg_scores[m], ks=ks)}
    return out


def print_stratified(strat):
    """Compact per-bucket report (NDCG/Hit only -- Precision/Recall duplicate
    Hit in the 1-positive protocol)."""
    width = max([9] + [len(name) for name in strat])
    for name, res in strat.items():
        metrics = '  '.join(f'{k} {v:.4f}' for k, v in res.items()
                            if k.startswith(('NDCG', 'Hit')))
        print(f'  {name:<{width}} ({res["n_edges"]:>7} edges)  {metrics}')


def evaluate_full_ranking(src_emb, dst_emb, edges, known_adj, ks=(10, 20, 50), chunk_size=1024):
    """Full ranking instead of 99 sampled negatives: each eval edge (u, p) ranks p against
    EVERY dst node u has no observed edge to (known_adj: CSR of all observed edges, train +
    val + test; p itself stays in). Ties rank pessimistically, as in evaluate_ranking.

    Per edge, averaged over edges: Hit@k, NDCG@k, MRR. Per user (all of u's eval positives
    ranked together, macro-averaged over users, the usual all-ranking protocol):
    UserRecall@k = |top-k with u's eval positives| / |u's eval positives|, UserNDCG@k.
    Sampled metrics can rank models differently from full ranking (Krichene & Rendle,
    KDD 2020), so this is the check on the sampled table."""
    edges = np.asarray(edges)
    known = known_adj.tocsr()
    n_dst = dst_emb.shape[0]
    ranks = np.empty(len(edges), dtype=np.int64)
    pos = np.empty(len(edges), dtype=np.float64)
    for start in range(0, len(edges), chunk_size):
        e = edges[start:start + chunk_size]
        u = torch.as_tensor(e[:, 0], device=src_emb.device)
        p = torch.as_tensor(e[:, 1], device=src_emb.device)
        s = src_emb[u] @ dst_emb.T                                  # [c, n_dst]
        target = s[torch.arange(len(e), device=s.device), p].clone()
        sub = known[e[:, 0]].tocoo()                                # the chunk's observed items
        s[torch.as_tensor(sub.row, device=s.device), torch.as_tensor(sub.col, device=s.device)] = float('-inf')
        s[torch.arange(len(e), device=s.device), p] = target
        ranks[start:start + len(e)] = (s >= target[:, None]).sum(1).cpu().numpy()
        pos[start:start + len(e)] = target.cpu().numpy()
    out = {'n_edges': len(edges)}
    for k in ks:
        hit = ranks <= min(k, n_dst)
        out[f'Hit@{k}'] = float(hit.mean())
        out[f'NDCG@{k}'] = float((hit / np.log2(ranks + 1)).mean())
    out['MRR'] = float((1.0 / ranks).mean())
    # per user: another eval positive of the same user scoring at least as high is a
    # candidate too, so it pushes this positive down by one
    order = np.argsort(edges[:, 0], kind='stable')
    users = edges[order, 0]
    starts = np.flatnonzero(np.r_[True, users[1:] != users[:-1]])
    bounds = np.r_[starts, len(users)]
    sums = {k: np.zeros(2) for k in ks}
    for i in range(len(starts)):
        idx = order[bounds[i]:bounds[i + 1]]
        sc = pos[idx]
        r = ranks[idx] + (sc[None, :] >= sc[:, None]).sum(1) - 1
        for k in ks:
            top = r <= k
            idcg = (1.0 / np.log2(np.arange(2, min(len(idx), k) + 2))).sum()
            sums[k] += (top.sum() / len(idx), (1.0 / np.log2(r[top] + 1)).sum() / idcg)
    for k in ks:
        out[f'UserRecall@{k}'], out[f'UserNDCG@{k}'] = (sums[k] / len(starts)).tolist()
    out['n_users'] = len(starts)
    return out


def print_full_ranking(res):
    print(f"Full-ranking metrics ({res['n_edges']} edges, {res['n_users']} users; every product the "
          f"user has no observed edge to is a candidate):")
    print('  ' + '  '.join(f'{k} {v:.4f}' for k, v in res.items() if not k.startswith('n_')))


def report_twin_breakdown(pos_scores, neg_scores, link_split, edges):
    """Print (and return) the duplicate-edition breakdown for the given eval edges.
    Imported lazily to keep han_utils free of a link_data dependency at import time."""
    from link_data import twin_in_train
    flag = twin_in_train(link_split, edges)
    strat = evaluate_ranking_by_flag(pos_scores, neg_scores, flag)
    print(f'By duplicate-edition twin in train (identical reviewer sets; '
          f'{100 * flag.mean():.1f}% of eval edges have one):')
    print_stratified(strat)
    return strat