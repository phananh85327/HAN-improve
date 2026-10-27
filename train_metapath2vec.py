#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Metapath2vec (Dong et al., KDD 2017) baseline under the shared link-prediction
protocol: meta-path-guided random walks + skip-gram with negative sampling, then
FROZEN embeddings scored by dot product on the same split / 99-negative ranking
eval as HAN and LMSPS.

Walk templates are the same auto-discovered P...P meta-paths HAN consumes
('PUP' repeated gives P-U-P-U-..., templates cycle cleanly because every path
starts and ends at the target type). Walk steps sample uniform neighbors from
build_metapath_graphs.build_base_matrices(), whose U-P matrices contain ONLY the
training split -- val/test interactions can never enter the corpus.

Negatives are drawn within the context node's type (each type occupies a
contiguous global-id block), metapath2vec++-style, so the model cannot cheat by
merely separating node types.

    python train_metapath2vec.py --arch amazon_lp_manual_5 --epochs 10 --evals-per-epoch 4 \
        --patience 6 --seed 1 --save-emb output/AMAZON/v3/m2v_amazon_lp_manual_5_s1.pt
"""

import argparse
import os
import sys
import time

# cap CUDA allocator block-splitting (same as train_han/train_gat): the sparse-grad
# accumulation pattern fragments the pool over thousands of chunked backwards
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:256')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from data_loader import data_loader
from link_data import build_link_split, data_provenance, check_data
from build_metapath_graphs import build_base_matrices
from han_utils import (NODE_TYPE_IDS, bpr_loss, score_edges, evaluate_ranking,
                       set_random_seed, get_n_params,
                       evaluate_ranking_by_user, print_by_user,
                       evaluate_ranking_stratified, print_stratified,
                       report_twin_breakdown)


def generate_walks(base, templates, num_start, shifts, walks_per_node,
                   walk_length, rng):
    """Vectorized meta-path-guided walks. Returns [num_walks, walk_length + 1]
    GLOBAL node ids, -1-padded where a walk hit a neighborless node."""
    all_walks = []
    for template in templates:
        steps = [template[i:i + 2] for i in range(len(template) - 1)]
        for _ in range(walks_per_node):
            cur = np.arange(num_start, dtype=np.int64)      # local ids, type P
            alive = np.ones(num_start, dtype=bool)
            seq = np.full((num_start, walk_length + 1), -1, dtype=np.int64)
            seq[:, 0] = cur + shifts[template[0]]
            for s in range(walk_length):
                pair = steps[s % len(steps)]
                mat = base[pair]
                indptr, indices = mat.indptr, mat.indices
                deg = indptr[cur + 1] - indptr[cur]
                ok = alive & (deg > 0)
                if not ok.any():
                    break
                off = (rng.random(int(ok.sum())) * deg[ok]).astype(np.int64)
                nxt = np.full_like(cur, -1)
                nxt[ok] = indices[indptr[cur[ok]] + off]
                cur, alive = nxt, ok
                seq[ok, s + 1] = cur[ok] + shifts[pair[1]]
            all_walks.append(seq)
    return np.concatenate(all_walks, axis=0)


def main(args):
    if args.seed > 0:
        set_random_seed(args.seed)
    device = 'cuda:{}'.format(args.gpu) if not args.cpu else 'cpu'
    if not args.cpu and args.vram_fraction > 0:
        # headroom for the Windows display stack: overshoot raises a clean CUDA
        # OOM instead of silently spilling to shared memory / bluescreening
        torch.cuda.set_per_process_memory_fraction(args.vram_fraction, args.gpu)
    rng = np.random.default_rng(args.seed)

    if args.paths:
        templates = [p.strip() for p in args.paths.split(',') if p.strip()]
    elif args.arch:
        from arch import archs
        templates = archs[args.arch][0]
    else:
        raise SystemExit('pass --paths "PUP,..." or --arch <arch.py key>')

    src_name, dst_name = args.target_relation.split(',')
    type_ids = NODE_TYPE_IDS[args.dataset]
    for t in templates:
        assert len(t) >= 2 and t[0] == dst_name and t[-1] == dst_name, \
            f'template {t} must start and end at target type {dst_name}'
    warnings = []
    if not any(src_name in t for t in templates):
        msg = (f'WARNING: no template contains {src_name}: {src_name} nodes are never visited, '
               f'their embeddings stay at random init and the ranking metrics will sit at the '
               f'chance floor (~0.10 Hit@10). Add a path through {src_name} (e.g. PUP).')
        print(msg)
        warnings.append(msg)
    run_start = time.time()

    # ---- data: identical split + train-only base matrices ----
    dl = data_loader(f'{args.root}/{args.dataset}')
    link_split = build_link_split(dl, type_ids[src_name], type_ids[dst_name],
                                  seed=args.split_seed, num_neg_eval=args.num_neg_eval)
    data_prov = data_provenance(os.path.join(args.root, args.dataset), dl, link_split)
    check_data(data_prov, args.split_seed, args.allow_legacy_data)
    base = build_base_matrices(dl, args.dataset, link_split)
    shifts = {name: dl.nodes['shift'][tid] for name, tid in type_ids.items()}
    counts = {name: dl.nodes['count'][tid] for name, tid in type_ids.items()}
    num_nodes = dl.nodes['total']

    # ---- walk corpus ----
    print(f'Generating walks: {len(templates)} templates x {args.walks_per_node} '
          f'per product x length {args.walk_length} ...')
    tic = time.time()
    walks = generate_walks(base, templates, counts[dst_name], shifts,
                           args.walks_per_node, args.walk_length, rng)
    visited = np.unique(walks[walks >= 0])
    print(f'  {walks.shape[0]} walks in {time.time() - tic:.1f}s; '
          f'{visited.size}/{num_nodes} nodes visited '
          f'({num_nodes - visited.size} keep their random init)')

    # ---- skip-gram with type-aware negative sampling ----
    emb_in = nn.Embedding(num_nodes, args.dim, sparse=True).to(device)
    emb_out = nn.Embedding(num_nodes, args.dim, sparse=True).to(device)
    nn.init.uniform_(emb_in.weight, -0.5 / args.dim, 0.5 / args.dim)
    nn.init.zeros_(emb_out.weight)
    optimizer = torch.optim.SparseAdam(list(emb_in.parameters())
                                       + list(emb_out.parameters()), lr=args.lr)
    print('# Params:', get_n_params(emb_in) + get_n_params(emb_out))

    # contiguous type blocks -> negatives drawn from the context node's own type
    tids_sorted = sorted(type_ids.values())
    lows = torch.tensor([dl.nodes['shift'][t] for t in tids_sorted], device=device)
    spans = torch.tensor([dl.nodes['count'][t] for t in tids_sorted],
                         device=device, dtype=torch.float32)
    bounds = torch.tensor([dl.nodes['shift'][t] for t in tids_sorted[1:]], device=device)

    def sgns_step(center, context):
        ce = emb_in(center)
        pos = F.logsigmoid((ce * emb_out(context)).sum(-1))
        tix = torch.bucketize(context, bounds, right=True)
        neg = lows[tix].unsqueeze(1) + (
            torch.rand(context.numel(), args.negatives, device=device)
            * spans[tix].unsqueeze(1)).long()
        neg_score = torch.bmm(emb_out(neg), ce.unsqueeze(-1)).squeeze(-1)
        return -(pos + F.logsigmoid(-neg_score).sum(1)).mean()

    walks_t = torch.from_numpy(walks)
    num_walks = walks_t.shape[0]

    # ---- val-based epoch selection (same protocol as HAN/GAT's best-val
    # checkpointing: without it, m2v reports whatever the LAST epoch happens to
    # be, an asymmetry that can only hurt the baseline) ----
    p_lo = shifts[dst_name]
    u_lo = shifts[src_name]
    val_edges = torch.LongTensor(link_split['val_edges']).to(device)
    val_negs = torch.LongTensor(link_split['val_negs']).to(device)

    def current_pu():
        P = emb_in.weight[p_lo:p_lo + counts[dst_name]].detach()
        U = emb_in.weight[u_lo:u_lo + counts[src_name]].detach()
        return P, U

    n_batches = (num_walks + args.batch_walks - 1) // args.batch_walks
    eval_every = max(1, n_batches // max(1, args.evals_per_epoch))

    best_val, best_epoch, best_P, best_U, wait = float('inf'), -1, None, None, 0
    stop = False
    for epoch in range(args.epochs):
        if stop:
            break
        tic = time.time()
        perm = torch.randperm(num_walks)
        total, nb = 0.0, 0
        for bi, start in enumerate(tqdm(range(0, num_walks, args.batch_walks),
                                        leave=False)):
            w = walks_t[perm[start:start + args.batch_walks]].to(device)
            optimizer.zero_grad()
            batch_loss, nparts = 0.0, 0
            for off in range(1, args.window + 1):
                a = w[:, :-off].reshape(-1)
                b = w[:, off:].reshape(-1)
                m = (a >= 0) & (b >= 0)
                if not m.any():
                    continue
                # both directions
                aa = torch.cat([a[m], b[m]])
                bb = torch.cat([b[m], a[m]])
                n_pairs = aa.numel()
                # chunked gradient accumulation: ONE offset's full backward peaks
                # at ~5 GiB of transients (the bmm gradient plus the sparse
                # embedding-gradient values, both [n_pairs x negatives, dim]) --
                # at batch-walks 4096 an offset is 163k pairs, a guaranteed
                # OOM/WDDM-spill on a 12 GB card. Each chunk's loss is scaled by
                # n_chunk/n_pairs, so the SUM of chunk losses (and the accumulated
                # gradient) equals the offset's mean EXACTLY -- same math, bounded
                # peak. optimizer.step() still runs once per batch.
                for s in range(0, n_pairs, args.pair_chunk):
                    ca = aa[s:s + args.pair_chunk]
                    cb = bb[s:s + args.pair_chunk]
                    loss = sgns_step(ca, cb) * (ca.numel() / n_pairs)
                    loss.backward()
                    batch_loss += float(loss)
                nparts += 1
            if nparts:
                optimizer.step()
                total += batch_loss
                nb += 1

            # ---- validation, --evals-per-epoch times per pass over the corpus.
            # SGNS peaks against the RANKING objective within the first epoch or
            # two (sgns loss keeps falling long after val bpr starts rising), so
            # once-per-epoch selection can only land on a coarse grid around that
            # peak -- a finer grid gives the baseline its best honest shot.
            # --patience/--min-delta are counted in EVALUATIONS, not epochs.
            if (bi + 1) % eval_every and (bi + 1) != n_batches:
                continue
            with torch.no_grad():
                P_emb, U_emb = current_pu()
                pos = score_edges(U_emb, P_emb, val_edges[:, 0], val_edges[:, 1])
                neg = score_edges(U_emb, P_emb, val_edges[:, 0], val_negs)
                val_loss = bpr_loss(pos, neg).item()
            progress = epoch + (bi + 1) / n_batches
            print(f'Epoch {progress:.2f}, sgns loss {total / max(nb, 1):.4f}, '
                  f'val bpr {val_loss:.4f} ({time.time() - tic:.0f}s into epoch)')
            total, nb = 0.0, 0
            if val_loss <= best_val:
                # same rule as train_han/train_gat/main_path: any improvement
                # updates the best snapshot, only an improvement > --min-delta
                # resets patience
                wait = 0 if best_val - val_loss > args.min_delta else wait + 1
                best_val, best_epoch = val_loss, progress
                best_P, best_U = P_emb.cpu().clone(), U_emb.cpu().clone()
            else:
                wait += 1
            if wait >= args.patience:
                print(f'Early stop at epoch {progress:.2f} (no val improvement > '
                      f'{args.min_delta} for {args.patience} evaluations)')
                stop = True
                break
    print(f'Best epoch {best_epoch:.2f}, val bpr {best_val:.4f}')
    train_time_s = time.time() - run_start
    if best_P is None:
        raise SystemExit('no validation evaluation ever improved on inf (val bpr NaN, or '
                         '--epochs 0); nothing to evaluate')

    # ---- frozen-embedding evaluation of the BEST-val epoch, identical protocol ----
    with torch.no_grad():
        P_emb = best_P.to(device)
        U_emb = best_U.to(device)
        test_edges = torch.LongTensor(link_split['test_edges']).to(device)
        test_negs = torch.LongTensor(link_split['test_negs']).to(device)
        pos_scores = score_edges(U_emb, P_emb, test_edges[:, 0], test_edges[:, 1])
        neg_scores = score_edges(U_emb, P_emb, test_edges[:, 0], test_negs)
        results = evaluate_ranking(pos_scores, neg_scores)

    print('Test ranking metrics:')
    for k, v in results.items():
        print(f'  {k}: {v:.4f}')
    deg = np.asarray(link_split['train_adj_src2dst'].sum(axis=1)).ravel()
    strat = evaluate_ranking_stratified(pos_scores, neg_scores,
                                        deg[link_split['test_edges'][:, 0]])
    print('By user training degree:')
    print_stratified(strat)
    twin = report_twin_breakdown(pos_scores, neg_scores, link_split, link_split['test_edges'])
    print_by_user(evaluate_ranking_by_user(
        pos_scores, neg_scores, link_split['test_edges'], link_split['test_negs']))

    if args.save_emb:
        save_path = args.save_emb
        if save_path == 'auto':
            save_path = f'./output/{args.dataset}/m2v_emb.pt'
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        torch.save({'model': 'metapath2vec', 'dataset': args.dataset,
                    'target_relation': args.target_relation, 'paths': templates,
                    'split_seed': args.split_seed, 'P_emb': P_emb.cpu(),
                    'U_emb': U_emb.cpu(), 'metrics': results, 'metrics_twin': twin,
                    'best_val_loss': best_val, 'best_epoch': best_epoch,
                    'epochs_run': args.epochs if not stop else int(best_epoch) + 1,
                    'train_time_s': float(train_time_s), 'warnings': warnings,
                    'data': data_prov, 'config': vars(args)}, save_path)
        print(f'Embeddings saved to {save_path} (serve with recommend.py)')
    return results


def parse_args():
    ap = argparse.ArgumentParser(
        description='Metapath2vec baseline on the shared link-prediction protocol.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--root', type=str, default='data')
    ap.add_argument('--dataset', type=str, default='AMAZON')
    ap.add_argument('--target-relation', type=str, default='U,P')
    ap.add_argument('--paths', type=str, default='',
                    help='comma-separated walk templates (overrides --arch)')
    ap.add_argument('--arch', type=str, default='',
                    help='arch.py key holding the discovered path list')
    ap.add_argument('--split-seed', type=int, default=1,
                    help='must match the search --split-seed (the LMSPS fleets use 1)')
    ap.add_argument('--allow-legacy-data', action='store_true',
                    help='run on data without meta.json (from before the AMAZON rebuild) instead of stopping')
    ap.add_argument('--num-neg-eval', type=int, default=99)
    ap.add_argument('--dim', type=int, default=512)
    ap.add_argument('--walks-per-node', type=int, default=5,
                    help='walks started per product PER template')
    ap.add_argument('--walk-length', type=int, default=20, help='steps per walk')
    ap.add_argument('--window', type=int, default=3)
    ap.add_argument('--negatives', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--evals-per-epoch', type=int, default=1,
                    help='validation checks per pass over the walk corpus. The best epoch '
                         'has been 0-1 in every run so far, so 1 check/epoch selects on a '
                         'very coarse grid around the peak; 4 costs a few seconds each and '
                         'lets the baseline keep its true best point. Raise --patience with it')
    ap.add_argument('--patience', type=int, default=2,
                    help='stop after this many EVALUATIONS without a val improvement > '
                         '--min-delta (evaluations, not epochs -- see --evals-per-epoch). '
                         'SGNS overfits the ranking objective fast at lr 0.01, so later '
                         'epochs are wasted compute')
    ap.add_argument('--min-delta', type=float, default=5e-4,
                    help='minimum val-bpr improvement that resets patience (same rule as '
                         'train_han/train_gat: smaller improvements still update the best '
                         'snapshot but count toward patience). 0 restores strict behavior')
    ap.add_argument('--vram-fraction', type=float, default=0.9,
                    help='cap the CUDA allocator at this fraction of total VRAM (headroom for the '
                         'display driver; overshoot = clean CUDA OOM, not a spill/bluescreen). 0 disables')
    ap.add_argument('--batch-walks', type=int, default=1024,
                    help='walks per optimizer step. Sparse grads of ALL pairs in a batch '
                         'accumulate until step(): ~1.7 GB at 1024 walks but ~6.7 GB at the old '
                         'default 4096, which cannot fit a 12 GB card alongside the 1.2 GB '
                         'tables + 2.5 GB SparseAdam state (it only ever ran by spilling to '
                         'shared memory). NOTE: changing this changes optimization (4x more, '
                         'smaller steps than pre-2026-09-09 runs)')
    ap.add_argument('--pair-chunk', type=int, default=50000,
                    help='center-context pairs per backward inside a batch. Bounds peak VRAM '
                         '(each chunk holds ~pair_chunk x negatives x dim transients in backward); '
                         'chunk losses are scaled so gradients match the unchunked mean exactly')
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--paper-walks', action='store_true',
                    help='walk-corpus settings of the metapath2vec baseline in the HAN paper '
                         '(Wang et al. 2019, Sec. 5.3: walk length 100, window 5, 5 negatives) with '
                         '10 walks per node per template instead of the paper\'s 40 (one epoch is '
                         '~2 h on a 12 GB card at 10; 40 would be ~8 h). Any of --walk-length, '
                         '--walks-per-node, --window, --negatives passed explicitly keeps its value. '
                         'Default (len 20, 5 walks, window 3) is the fast sweep setting; the paper '
                         'used dim 64/128 -- dim stays 512 here so every model shares one width')
    ap.add_argument('--save-emb', type=str, default='auto')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args()
    if args.paper_walks:
        args = apply_paper_walks(args, sys.argv[1:])
    return args


PAPER_WALKS = {'walk_length': 100, 'walks_per_node': 10, 'window': 5, 'negatives': 5}


def apply_paper_walks(args, argv):
    """Set PAPER_WALKS on args unless the flag was passed explicitly (explicit wins)."""
    explicit = {a.split('=')[0] for a in argv if a.startswith('--')}
    applied = {}
    for key, val in PAPER_WALKS.items():
        if '--' + key.replace('_', '-') in explicit:
            continue
        setattr(args, key, val)
        applied[key] = val
    print('--paper-walks: ' + ', '.join(f'{k}={v}' for k, v in applied.items())
          + (' (explicitly passed flags kept)' if len(applied) < len(PAPER_WALKS) else ''))
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    main(args)
