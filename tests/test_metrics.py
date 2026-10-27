"""Ranking metrics against brute-force definitions (HAN review part H5).

    python tests/test_metrics.py      (or: python -m pytest tests)

Scores are drawn from a few distinct values so that ties are common: every metric ranks a tie
pessimistically (the positive counts as below every candidate with an equal score).
"""
import math
import os
import sys

import numpy as np
import scipy.sparse as sp
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from han_utils import (evaluate_full_ranking, evaluate_ranking, evaluate_ranking_by_flag,  # noqa: E402
                       evaluate_ranking_by_user, evaluate_ranking_stratified, score_edges)
from link_data import product_twin_groups, twin_in_train  # noqa: E402

KS = (1, 5, 10, 20)


def _brute_edge(pos, neg, k):
    """Per-edge Hit/NDCG/Precision/Recall@k, one positive against its own negatives."""
    hit = ndcg = 0.0
    for p, row in zip(pos.tolist(), neg.tolist()):
        rank = 1 + sum(1 for x in row if x >= p)
        if rank <= k:
            hit += 1
            ndcg += 1 / math.log2(rank + 1)
    n = len(pos)
    return {'Hit': hit / n, 'NDCG': ndcg / n, 'Precision': hit / n / k, 'Recall': hit / n}


def test_per_edge_metrics():
    rng = np.random.default_rng(0)
    pos = torch.tensor(rng.integers(0, 6, 300), dtype=torch.float32)
    neg = torch.tensor(rng.integers(0, 6, (300, 30)), dtype=torch.float32)
    got = evaluate_ranking(pos, neg, ks=KS)
    for k in KS:
        want = _brute_edge(pos, neg, k)
        for m, v in want.items():
            assert abs(got[f'{m}@{k}'] - v) < 1e-6, (m, k, got[f'{m}@{k}'], v)


def test_per_user_metrics():
    rng = np.random.default_rng(1)
    n_users, n_items, k_neg = 40, 60, 8
    edges, negs = [], []
    for u in range(n_users):
        items = rng.choice(n_items, rng.integers(1, 5), replace=False)
        rest = np.setdiff1d(np.arange(n_items), items)
        for p in items:
            edges.append((u, p))
            negs.append(rng.choice(rest, k_neg, replace=False))    # overlaps across a user's edges
    edges, negs = np.array(edges), np.array(negs)
    item_score = rng.integers(0, 7, n_items).astype(np.float32)     # one score per (user, item): ties
    user_shift = rng.integers(0, 3, n_users)
    score = lambda u, i: float(item_score[i] + user_shift[u])
    pos = torch.tensor([score(u, p) for u, p in edges])
    neg = torch.tensor([[score(u, i) for i in row] for (u, _), row in zip(edges, negs)])
    got = evaluate_ranking_by_user(pos, neg, edges, negs, ks=(1, 3, 10))
    assert got['n_users'] == n_users
    for k in (1, 3, 10):
        sums = np.zeros(4)
        for u in range(n_users):
            mine = edges[:, 0] == u
            positives = set(edges[mine, 1].tolist())
            cands = positives | set(negs[mine].ravel().tolist())
            hits, dcg = 0, 0.0
            for p in positives:
                rank = sum(1 for c in cands if score(u, c) >= score(u, p))
                if rank <= k:
                    hits += 1
                    dcg += 1 / math.log2(rank + 1)
            idcg = sum(1 / math.log2(r + 1) for r in range(1, min(len(positives), k) + 1))
            sums += (hits / k, hits / len(positives), dcg / idcg, float(hits > 0))
        prec, rec, ndcg, hit = sums / n_users
        for m, v in (('Precision', prec), ('Recall', rec), ('NDCG', ndcg), ('Hit', hit)):
            assert abs(got[f'{m}@{k}'] - v) < 1e-9, (m, k, got[f'{m}@{k}'], v)


def test_breakdowns_are_masked_per_edge_metrics():
    rng = np.random.default_rng(2)
    pos = torch.tensor(rng.random(200), dtype=torch.float32)
    neg = torch.tensor(rng.random((200, 20)), dtype=torch.float32)
    deg = rng.integers(0, 30, 200)
    strat = evaluate_ranking_stratified(pos, neg, deg, ks=(10,))
    for name, (lo, hi) in (('deg=0', (0, 0)), ('deg 1-5', (1, 5)), ('deg 6-20', (6, 20)), ('deg>=21', (21, 99))):
        m = (deg >= lo) & (deg <= hi)
        want = _brute_edge(pos[torch.from_numpy(m)], neg[torch.from_numpy(m)], 10)
        assert strat[name]['n_edges'] == m.sum()
        assert abs(strat[name]['Hit@10'] - want['Hit']) < 1e-6
    flag = rng.random(200) < 0.3
    by = evaluate_ranking_by_flag(pos, neg, flag)
    assert by['twin in train']['n_edges'] == flag.sum()
    want = _brute_edge(pos[torch.from_numpy(flag)], neg[torch.from_numpy(flag)], 10)
    assert abs(by['twin in train']['NDCG@10'] - want['NDCG']) < 1e-6
    assert 'twin in train' not in evaluate_ranking_by_flag(pos, neg, np.zeros(200, bool))   # empty group left out


def test_chunked_scores():
    rng = np.random.default_rng(3)
    U, P = torch.randn(50, 8), torch.randn(70, 8)
    src = torch.tensor(rng.integers(0, 50, 2500))
    dst = torch.tensor(rng.integers(0, 70, (2500, 11)))
    want = torch.einsum('nd,nkd->nk', U[src], P[dst])
    assert torch.allclose(score_edges(U, P, src, dst, chunk_size=97), want, atol=1e-5)
    assert torch.allclose(score_edges(U, P, src, dst[:, 0]), want[:, 0], atol=1e-5)


def test_twin_flags():
    # products 0, 1, 2 share one reviewer set; 3 has another; 4 has none
    full = sp.csr_matrix(np.array([[1, 1, 1, 0, 0],
                                   [1, 1, 1, 1, 0],
                                   [0, 0, 0, 1, 0]], dtype=np.float32))
    g = product_twin_groups(full)
    assert g[0] == g[1] == g[2] and len({g[0], g[3], g[4]}) == 3
    train = sp.csr_matrix(np.array([[1, 0, 0, 0, 0],
                                    [0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0]], dtype=np.float32))
    split = {'dst_twin_group': g, 'train_adj_src2dst': train}
    flags = twin_in_train(split, np.array([[0, 2], [1, 2], [2, 3]]))
    assert flags.tolist() == [True, False, False]    # user 0 has twin 0 in train; user 1 only product 3


def test_full_ranking():
    rng = np.random.default_rng(4)
    n_u, n_p = 30, 40
    U = torch.tensor(rng.integers(-2, 3, (n_u, 3)), dtype=torch.float32)     # integer scores: ties
    P = torch.tensor(rng.integers(-2, 3, (n_p, 3)), dtype=torch.float32)
    known = (rng.random((n_u, n_p)) < 0.15).astype(np.float32)
    edges = np.argwhere(known)[rng.permutation(int(known.sum()))[:60]]       # eval edges among the known ones
    got = evaluate_full_ranking(U, P, edges, sp.csr_matrix(known), ks=(1, 5, 10), chunk_size=7)
    S = (U @ P.T).numpy()
    ranks = []
    for u, p in edges.tolist():
        cands = [j for j in range(n_p) if not known[u, j] or j == p]
        ranks.append(sum(1 for j in cands if S[u, j] >= S[u, p]))
    ranks = np.array(ranks)
    for k in (1, 5, 10):
        assert abs(got[f'Hit@{k}'] - (ranks <= k).mean()) < 1e-9
        assert abs(got[f'NDCG@{k}'] - ((ranks <= k) / np.log2(ranks + 1)).mean()) < 1e-9
    assert abs(got['MRR'] - (1 / ranks).mean()) < 1e-9
    # per user: the user's eval positives are ranked together among the same candidates
    users = sorted(set(edges[:, 0].tolist()))
    for k in (1, 5, 10):
        rec = ndcg = 0.0
        for u in users:
            mine = [p for uu, p in edges.tolist() if uu == u]
            cands = [j for j in range(n_p) if not known[u, j] or j in mine]
            r = [sum(1 for j in cands if S[u, j] >= S[u, p]) for p in mine]
            top = [x for x in r if x <= k]
            rec += len(top) / len(mine)
            ndcg += sum(1 / math.log2(x + 1) for x in top) / sum(1 / math.log2(i + 2) for i in range(min(len(mine), k)))
        assert abs(got[f'UserRecall@{k}'] - rec / len(users)) < 1e-9
        assert abs(got[f'UserNDCG@{k}'] - ndcg / len(users)) < 1e-9


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
