#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plain GAT baseline -- NO meta-paths. The heterogeneity-blind comparison used in
the HAN paper itself: one homogeneous multi-layer GAT over the union of ALL raw
edges (P-P also-bought, C-C hierarchy, P-C membership, and the TRAIN-ONLY U-P
review edges), with per-type input projections to reconcile feature dimensions.
User embeddings come from the same GAT (no shallow tower) -- contrasting with
HAN/LMSPS is the point.

Same split, BPR training, and 99-negative ranking eval as every other model;
saves the shared embedding format for recommend.py.

    python train_gat.py --dataset AMAZON --split-seed 0

VRAM: ~8M edges x 8 heads x 2 layers peaks around 8-10 GB on the default config,
close to a 12 GB card's limit; sustained near-limit pressure is what killed long
runs with sporadic CUDA illegal-memory-access errors. --num-heads 4 halves every
per-edge tensor and is the recommended first lever (then --in-dim 128). The
best-val checkpoint path is printed at startup; a crashed run can be salvaged
without retraining (use the SAME model flags the run was started with):
    python train_gat.py --eval-only output/AMAZON/gat_<hex>.pkl
"""

import argparse
import gc
import os
import time
import uuid

# cap CUDA allocator block-splitting: multi-hour full-graph runs on a near-full
# card fragment the pool otherwise (must be set before the first CUDA allocation;
# a value already set outside wins)
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:256')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from data_loader import data_loader
from link_data import build_link_split, TrainEdgeSampler
from model_han import GATLayer
from han_utils import (NODE_TYPE_IDS, bpr_loss, score_edges, evaluate_ranking,
                       set_random_seed, get_n_params,
                       evaluate_ranking_by_user, print_by_user,
                       evaluate_ranking_stratified, print_stratified)


class HomoGAT(nn.Module):
    """Per-type linear input projections + stacked homogeneous GAT layers over
    the union graph. Returns embeddings for ALL nodes in global-id order."""

    def __init__(self, type_dims, in_dim, hidden, num_heads, feat_drop, attn_drop,
                 num_layers=2, grad_ckpt=False):
        super().__init__()
        self.grad_ckpt = grad_ckpt
        self.projs = nn.ModuleList([nn.Linear(d, in_dim) for d in type_dims])
        dims = [in_dim] + [hidden] * num_layers
        self.gats = nn.ModuleList([
            GATLayer(dims[i], hidden // num_heads, num_heads,
                     feat_drop=feat_drop, attn_drop=attn_drop)
            for i in range(num_layers)
        ])

    def forward(self, edges, type_feats):
        x = torch.cat([proj(f) for proj, f in zip(self.projs, type_feats)], dim=0)
        for gat in self.gats:
            if self.grad_ckpt and self.training and torch.is_grad_enabled():
                # recompute this layer's [E, heads] internals during backward --
                # bit-identical results (RNG preserved), several GB less peak VRAM
                x = checkpoint(gat, edges, x, use_reentrant=False)
            else:
                x = gat(edges, x)
        return x                                            # [N_total, hidden]


def build_union_edges(dl, type_ids, src_name, dst_name, link_split, device):
    """All context relations on GLOBAL ids + train-only target edges + self-loops.
    Every relation in link.dat already has its reverse stored, so no transposing."""
    target_pair = {type_ids[src_name], type_ids[dst_name]}
    srcs, dsts = [], []
    for r, (h, t) in dl.links['meta'].items():
        if {h, t} == target_pair:
            continue                                        # replaced by train split
        coo = dl.links['data'][r].tocoo()
        srcs.append(coo.row.astype(np.int64))
        dsts.append(coo.col.astype(np.int64))

    tr = link_split['train_adj_src2dst'].tocoo()
    u = tr.row.astype(np.int64) + dl.nodes['shift'][type_ids[src_name]]
    p = tr.col.astype(np.int64) + dl.nodes['shift'][type_ids[dst_name]]
    srcs += [u, p]
    dsts += [p, u]

    loops = np.arange(dl.nodes['total'], dtype=np.int64)
    srcs.append(loops)
    dsts.append(loops)

    src_np = np.concatenate(srcs)
    dst_np = np.concatenate(dsts)
    # mutual also-bought/hierarchy pairs are stored in both directions, so the raw
    # union contains duplicate edges; they double-count in the attention softmax
    # and inflate every per-edge tensor, so keep one copy of each
    _, keep = np.unique(src_np * np.int64(dl.nodes['total']) + dst_np, return_index=True)
    src = torch.from_numpy(src_np[keep]).to(device)
    dst = torch.from_numpy(dst_np[keep]).to(device)
    print(f'Union graph: {dl.nodes["total"]} nodes, {src.numel()} edges '
          f'(deduped from {src_np.size}, incl. self-loops)')
    return (src, dst)


def main(args):
    if args.seed > 0:
        set_random_seed(args.seed)
    device = 'cuda:{}'.format(args.gpu) if not args.cpu else 'cpu'
    if not args.cpu and args.vram_fraction > 0:
        # headroom for the Windows display stack: overshoot raises a clean CUDA
        # OOM instead of a driver bluescreen (see train_han.py)
        torch.cuda.set_per_process_memory_fraction(args.vram_fraction, args.gpu)

    src_name, dst_name = args.target_relation.split(',')
    type_ids = NODE_TYPE_IDS[args.dataset]

    dl = data_loader(f'{args.root}/{args.dataset}')
    link_split = build_link_split(dl, type_ids[src_name], type_ids[dst_name],
                                  seed=args.split_seed, num_neg_eval=args.num_neg_eval)

    edges = build_union_edges(dl, type_ids, src_name, dst_name, link_split, device)

    n_types = len(dl.nodes['count'])
    type_feats = []
    for t in range(n_types):
        attr = dl.nodes['attr'][t]
        assert attr is not None, f'node type {t} has no attributes'
        type_feats.append(torch.FloatTensor(attr).to(device))
    p_lo = dl.nodes['shift'][type_ids[dst_name]]
    u_lo = dl.nodes['shift'][type_ids[src_name]]
    num_p = dl.nodes['count'][type_ids[dst_name]]
    num_u = dl.nodes['count'][type_ids[src_name]]

    train_sampler = TrainEdgeSampler(link_split['train_edges'],
                                     link_split['train_adj_src2dst'], seed=args.seed)
    val_edges = torch.LongTensor(link_split['val_edges']).to(device)
    val_negs = torch.LongTensor(link_split['val_negs']).to(device)
    test_edges = torch.LongTensor(link_split['test_edges']).to(device)
    test_negs = torch.LongTensor(link_split['test_negs']).to(device)

    model = HomoGAT([f.shape[1] for f in type_feats], args.in_dim, args.hidden,
                    args.num_heads, args.feat_drop, args.attn_drop,
                    num_layers=args.layers, grad_ckpt=not args.no_grad_ckpt).to(device)
    print(model)
    print('# Params:', get_n_params(model))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    checkpt_folder = f'./output/{args.dataset}/'
    os.makedirs(checkpt_folder, exist_ok=True)
    checkpt_file = checkpt_folder + 'gat_' + uuid.uuid4().hex
    if args.eval_only:
        print(f'--eval-only: skipping training, loading {args.eval_only}')
        args.stage = 0
    else:
        print(f'Checkpoint: {checkpt_file}.pkl (a crashed run can be salvaged with --eval-only)')

    best_epoch, best_val_loss, wait = -1, float('inf'), 0
    train_times = []
    iter_num = train_sampler.num_batches(args.batch_size)

    for epoch in tqdm(range(args.stage)):
        gc.collect()
        model.train()
        start = time.time()
        total = 0.0
        for _ in range(iter_num):
            src, pos_dst, neg_dst = train_sampler.sample_batch(args.batch_size)
            optimizer.zero_grad()
            x = model(edges, type_feats)
            src_emb = x[src.to(device) + u_lo]
            pos_emb = x[pos_dst.to(device) + p_lo]
            neg_emb = x[neg_dst.to(device) + p_lo]
            loss = bpr_loss((src_emb * pos_emb).sum(-1), (src_emb * neg_emb).sum(-1))
            loss.backward()
            optimizer.step()
            total += loss.item()
        train_times.append(time.time() - start)
        # release the last training batch's tensors before the val forward --
        # x alone is [N, hidden] ~ 620 MB and the card runs near its limit
        x = src_emb = pos_emb = neg_emb = loss = None
        if not args.cpu:
            torch.cuda.empty_cache()

        with torch.no_grad():
            model.eval()
            x = model(edges, type_feats)
            P_emb = x[p_lo:p_lo + num_p]
            U_emb = x[u_lo:u_lo + num_u]
            pos_scores = score_edges(U_emb, P_emb, val_edges[:, 0], val_edges[:, 1])
            neg_scores = score_edges(U_emb, P_emb, val_edges[:, 0], val_negs)
            val_loss = bpr_loss(pos_scores, neg_scores).item()
            x = P_emb = U_emb = pos_scores = neg_scores = None  # free before the next train epoch

        print(f'Epoch {epoch}, train bpr {total / iter_num:.4f}, val bpr {val_loss:.4f}')
        if val_loss <= best_val_loss:
            # any improvement updates the best checkpoint, but only an improvement
            # bigger than --min-delta resets patience (see train_han.py)
            wait = 0 if best_val_loss - val_loss > args.min_delta else wait + 1
            best_epoch, best_val_loss = epoch, val_loss
            torch.save(model.state_dict(), f'{checkpt_file}.pkl')
        else:
            wait += 1
        if wait >= args.patience:
            print(f'Early stop at epoch {epoch} (no val improvement > {args.min_delta} '
                  f'for {args.patience} epochs)')
            break

    if train_times:
        print('average train times', sum(train_times) / len(train_times))
        print(f'Best epoch {best_epoch}, val bpr {best_val_loss:.4f}')

    ckpt_path = args.eval_only or f'{checkpt_file}.pkl'
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    model = model.to(device)
    if not args.cpu:
        torch.cuda.empty_cache()

    with torch.no_grad():
        model.eval()
        x = model(edges, type_feats)
        P_emb = x[p_lo:p_lo + num_p]
        U_emb = x[u_lo:u_lo + num_u]
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
    print_by_user(evaluate_ranking_by_user(
        pos_scores, neg_scores, link_split['test_edges'], link_split['test_negs']))

    if args.save_emb:
        save_path = args.save_emb
        if save_path == 'auto':
            save_path = f'./output/{args.dataset}/gat_emb.pt'
        torch.save({'model': 'GAT', 'dataset': args.dataset,
                    'target_relation': args.target_relation, 'paths': [],
                    'split_seed': args.split_seed, 'P_emb': P_emb.cpu(),
                    'U_emb': U_emb.cpu(), 'metrics': results,
                    'config': vars(args)}, save_path)
        print(f'Embeddings saved to {save_path} (serve with recommend.py)')
    if not args.eval_only:
        os.remove(f'{checkpt_file}.pkl')
    return results


def parse_args():
    ap = argparse.ArgumentParser(
        description='Homogeneous GAT baseline (no meta-paths) on the shared protocol.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--root', type=str, default='data')
    ap.add_argument('--dataset', type=str, default='AMAZON')
    ap.add_argument('--target-relation', type=str, default='U,P')
    ap.add_argument('--split-seed', type=int, default=0)
    ap.add_argument('--num-neg-eval', type=int, default=99)
    ap.add_argument('--in-dim', type=int, default=256,
                    help='per-type projection dim before the first GAT layer')
    ap.add_argument('--hidden', type=int, default=512)
    ap.add_argument('--num-heads', type=int, default=8)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--feat-drop', type=float, default=0.2)
    ap.add_argument('--attn-drop', type=float, default=0.2)
    ap.add_argument('--no-grad-ckpt', action='store_true',
                    help='disable gradient checkpointing of the GAT layers. Checkpointing is ON '
                         'by default: several GB less peak VRAM (~+30%% step time), bit-identical results')
    ap.add_argument('--vram-fraction', type=float, default=0.9,
                    help='cap the CUDA allocator at this fraction of total VRAM (headroom for the '
                         'display driver; overshoot = clean CUDA OOM, not a bluescreen). 0 disables')
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--weight-decay', type=float, default=0.0)
    ap.add_argument('--stage', type=int, default=200, help='max epochs')
    ap.add_argument('--patience', type=int, default=20)
    ap.add_argument('--min-delta', type=float, default=5e-4,
                    help='minimum val-bpr improvement that resets patience. Smaller improvements '
                         'still update the best checkpoint but count toward patience. '
                         '0 restores the strict any-improvement-resets behavior')
    ap.add_argument('--batch-size', type=int, default=100000)
    ap.add_argument('--save-emb', type=str, default='auto')
    ap.add_argument('--eval-only', type=str, default='',
                    help='path to a saved gat_*.pkl checkpoint: skip training, just evaluate and save embeddings')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--seed', type=int, default=1)
    return ap.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)
    main(args)
