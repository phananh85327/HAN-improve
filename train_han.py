#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train/evaluate HAN on link prediction with LMSPS-discovered meta-paths.

Same edge split as the LMSPS search (build_link_split with --split-seed), BPR loss with
TrainEdgeSampler, and 99-sampled-negative Precision/Recall/NDCG/Hit@K evaluation, like every
model in this repo.

Prerequisites:
  1. The meta-path sets in arch.py (or --paths): the LMSPS Fleet v3 sets, searched with
     --tgt-ended-paths so every path starts and ends at a product.
  2. Materialize the paths as graphs:
       python build_metapath_graphs.py --arch <key> --graph-dir <dir> --skip-existing
  3. This script (run_all.bat runs every set; README "Run flow"):
       python train_han.py --arch <key> --graph-dir <dir> --stage 150 --patience 10 \
           --beta-warmup 15 --seed 1 --save-emb output/AMAZON/v3/han_<key>_s1.pt
"""

import argparse
import datetime
import gc
import json
import os
import sys
import time
import uuid

# cap CUDA allocator block-splitting (same as train_gat.py): multi-hour full-graph
# runs near the VRAM limit fragment the pool otherwise, and sustained near-limit
# pressure is what escalates into Windows display-driver (dxgkrnl) crashes
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:256')

import numpy as np
import scipy.sparse as sp
import torch
from tqdm import tqdm

from data_loader import data_loader
from link_data import build_link_split, TrainEdgeSampler, data_provenance, check_data
from model_han import HAN_LP, ShallowNodeTower, UserEmbeddingTower
from han_utils import (NODE_TYPE_IDS, bpr_loss, score_edges, evaluate_ranking,
                       evaluate_ranking_stratified, print_stratified,
                       evaluate_ranking_by_user, print_by_user,
                       set_random_seed, get_n_params, report_twin_breakdown)


def graph_sidecar(graph_dir, path, topk, split_seed):
    """The .meta.json build_metapath_graphs.py writes next to a graph ({} if missing)."""
    meta_file = os.path.join(graph_dir, f'{path}_top{topk}_split{split_seed}.meta.json')
    if not os.path.isfile(meta_file):
        return {}
    with open(meta_file, encoding='utf-8') as f:
        return json.load(f)


def load_metapath_graphs(graph_dir, paths, topk, split_seed, num_nodes, device,
                         data_hash, allow_legacy=False):
    """One (src, dst) edge list per path. Graph files are found by name, which does not
    change with the data, so each graph's .meta.json must carry this run's training-edge
    fingerprint (data_hash): a graph built from other data or another split stops the
    run. allow_legacy: a graph from before the fingerprint existed only warns."""
    graphs = []
    loop = torch.arange(num_nodes, dtype=torch.long)
    rebuild = 'rebuild it with build_metapath_graphs.py (same --split-seed/--topk)'
    for p in paths:
        stem = os.path.join(graph_dir, f'{p}_top{topk}_split{split_seed}')
        fname = stem + '.npz'
        if not os.path.isfile(fname):
            raise SystemExit(f'missing {fname} -- run build_metapath_graphs.py first '
                             f'with the same --paths/--topk/--split-seed')
        adj = sp.load_npz(fname).tocoo()
        if adj.shape != (num_nodes, num_nodes):
            raise SystemExit(f'{fname} has {adj.shape[0]} nodes but the data has {num_nodes} '
                             f'products: it was built from other data -- {rebuild}')
        meta = graph_sidecar(graph_dir, p, topk, split_seed)
        built_on = meta.get('train_edges_hash')
        if built_on != data_hash:
            if built_on is not None:
                raise SystemExit(f'{fname} was built from other training edges (fingerprint '
                                 f'{built_on}, this run {data_hash}): other data or another '
                                 f'split -- {rebuild}')
            msg = f'{fname} does not record which data it was built from -- {rebuild}'
            if not allow_legacy:
                raise SystemExit(msg)
            print('  WARNING: ' + msg)
        # (src, dst) edge list with self-loops appended; pure torch tensors, so the
        # CPU-only dgl wheel in this env is never asked to move a graph to CUDA
        src = torch.cat([torch.from_numpy(adj.row.astype(np.int64)), loop]).to(device)
        dst = torch.cat([torch.from_numpy(adj.col.astype(np.int64)), loop]).to(device)
        graphs.append((src, dst))
        # tie-break mode from the sidecar written by build_metapath_graphs.py
        if meta:
            prov = f"tie-break {meta.get('tie_break')}"
            if meta.get('tie_seed') is not None:
                prov += f" (seed {meta['tie_seed']})"
        else:
            prov = ('WARNING: no .meta.json -- legacy graph built with the id tie-break '
                    '(low-id products become hubs); rebuild with build_metapath_graphs.py')
        print(f'  {p}: {src.numel()} edges (incl. self-loops); {prov}')
    return graphs


# Wang et al., WWW-2019, Sec. 5.3: Adam, lr 0.005, L2 0.001, semantic attention
# vector 128-d (already the model default), 8 heads, attention dropout 0.6, early
# stopping with patience 100, and a 64-d final embedding (8 heads x 8). The paper
# has no beta warmup, no temperature and no min-delta rule.
PAPER_CONFIG = {
    'lr': 0.005,
    'weight_decay': 0.001,
    'feat_drop': 0.6,
    'attn_drop': 0.6,
    'hidden': 64,
    'num_heads': 8,
    'patience': 100,
    'min_delta': 0.0,
    'beta_warmup': 0,
    'semantic_tau': 1.0,
}


def apply_paper_config(args, argv):
    """Overwrite PAPER_CONFIG keys on args unless the user passed that flag
    explicitly on the command line (explicit flags always win)."""
    explicit = {a.split('=')[0] for a in argv if a.startswith('--')}
    applied = {}
    for key, val in PAPER_CONFIG.items():
        flag = '--' + key.replace('_', '-')
        if flag in explicit:
            continue
        setattr(args, key, val)
        applied[key] = val
    print('--paper-config: ' + ', '.join(f'{k}={v}' for k, v in applied.items())
          + (' (explicitly passed flags kept)' if len(applied) < len(PAPER_CONFIG) else ''))
    return args


def main(args):
    if args.seed > 0:
        set_random_seed(args.seed)
    device = 'cuda:{}'.format(args.gpu) if not args.cpu else 'cpu'
    if not args.cpu and args.vram_fraction > 0:
        # leave headroom for the Windows display stack (WDDM): running the card at
        # 11.6/12 GB makes pressure surface as driver bluescreens instead of a
        # clean, catchable CUDA OOM
        torch.cuda.set_per_process_memory_fraction(args.vram_fraction, args.gpu)

    if args.paths:
        paths = [p.strip() for p in args.paths.split(',') if p.strip()]
    elif args.arch:
        from arch import archs
        paths = archs[args.arch][0]
    else:
        raise SystemExit('pass --paths "PUP,PP,..." or --arch <arch.py key>')

    src_name, dst_name = args.target_relation.split(',')
    type_ids = NODE_TYPE_IDS[args.dataset]
    for p in paths:
        assert len(p) >= 2 and p[0] == dst_name and p[-1] == dst_name, \
            f'path {p} must start and end at target type {dst_name}'

    # ---- data: same loader + same split as the LMSPS pipeline ----
    dl = data_loader(f'{args.root}/{args.dataset}')
    link_split = build_link_split(dl, type_ids[src_name], type_ids[dst_name],
                                  seed=args.split_seed, num_neg_eval=args.num_neg_eval)
    data_prov = data_provenance(os.path.join(args.root, args.dataset), dl, link_split)
    check_data(data_prov, args.split_seed, args.allow_legacy_data)

    p_feat = torch.FloatTensor(dl.nodes['attr'][type_ids[dst_name]]).to(device)
    if args.user_tower == 'feats':
        src_raw = dl.nodes['attr'][type_ids[src_name]]
        if src_raw is None:
            # a dense identity fallback would be [num_src, num_src] -- ~117 GB at
            # AMAZON's user count -- so refuse instead of crashing the machine
            raise SystemExit(f'node type {src_name} has no attributes; --user-tower feats '
                             f'needs them -- use --user-tower embedding instead')
        src_attr = torch.FloatTensor(src_raw)
        src_attr_dev = src_attr.to(device)

    graph_dir = args.graph_dir or os.path.join(args.root, args.dataset, 'metapath_graphs')
    print(f'Loading meta-path graphs from {graph_dir}:')
    graphs = load_metapath_graphs(graph_dir, paths, args.topk, args.split_seed,
                                  link_split['num_dst'], device,
                                  data_prov['train_edges_hash'], args.allow_legacy_data)

    train_sampler = TrainEdgeSampler(link_split['train_edges'],
                                     link_split['train_adj_src2dst'], seed=args.seed)
    val_edges = torch.LongTensor(link_split['val_edges']).to(device)
    val_negs = torch.LongTensor(link_split['val_negs']).to(device)
    test_edges = torch.LongTensor(link_split['test_edges']).to(device)
    test_negs = torch.LongTensor(link_split['test_negs']).to(device)

    # ---- model ----
    model = HAN_LP(len(paths), p_feat.shape[1], args.hidden, args.num_heads,
                   args.feat_drop, args.attn_drop,
                   semantic_tau=args.semantic_tau,
                   grad_ckpt=not args.no_grad_ckpt,
                   symmetric_init=not args.independent_init).to(device)
    if args.user_tower == 'embedding':
        tower = UserEmbeddingTower(link_split['num_src'], args.hidden).to(device)
    else:
        tower = ShallowNodeTower(src_attr.shape[1], args.hidden).to(device)

    def user_emb_batch(idx):
        return tower(idx) if args.user_tower == 'embedding' else tower(src_attr_dev[idx])

    def user_emb_all():
        return tower.all_users() if args.user_tower == 'embedding' else tower(src_attr_dev)
    print(model)
    print("# Params:", get_n_params(model) + get_n_params(tower))
    optimizer = torch.optim.Adam(list(model.parameters()) + list(tower.parameters()),
                                 lr=args.lr, weight_decay=args.weight_decay)

    checkpt_folder = f'./output/{args.dataset}/'
    os.makedirs(checkpt_folder, exist_ok=True)
    checkpt_file = checkpt_folder + 'han_' + uuid.uuid4().hex

    best_epoch, best_val_loss, wait = -1, float('inf'), 0
    train_times = []
    epochs_run = 0
    iter_num = train_sampler.num_batches(args.batch_size)
    assert args.beta_warmup < args.stage, '--beta-warmup must leave post-warmup epochs to train/checkpoint in'

    for epoch in tqdm(range(args.stage)):
        epochs_run = epoch + 1
        gc.collect()
        # warmup: freeze beta uniform so every GAT learns before attention judges;
        # applies to this epoch's train steps AND its val forward
        warmup = epoch < args.beta_warmup
        model.semantic.force_uniform = warmup
        model.train(); tower.train()
        start = time.time()
        total = 0.
        for _ in range(iter_num):
            src, pos_dst, neg_dst = train_sampler.sample_batch(args.batch_size)
            optimizer.zero_grad()
            # full-graph forward: GAT embeddings depend on neighbors, so the whole
            # target-type graph is encoded once per step (90k nodes is cheap on GPU)
            dst_emb, _ = model(graphs, p_feat)
            src_emb = user_emb_batch(src.to(device))
            pos_emb = dst_emb[pos_dst.to(device)]
            neg_emb = dst_emb[neg_dst.to(device)]
            loss = bpr_loss((src_emb * pos_emb).sum(-1), (src_emb * neg_emb).sum(-1))
            loss.backward()
            optimizer.step()
            total += loss.item()
        train_times.append(time.time() - start)
        train_loss = total / iter_num

        with torch.no_grad():
            model.eval(); tower.eval()
            dst_emb, beta = model(graphs, p_feat)
            src_emb = user_emb_all()
            pos_scores = score_edges(src_emb, dst_emb, val_edges[:, 0], val_edges[:, 1])
            neg_scores = score_edges(src_emb, dst_emb, val_edges[:, 0], val_negs)
            val_loss = bpr_loss(pos_scores, neg_scores).item()

        # log beta every epoch: a healthy run keeps several paths alive; a fast
        # one-hot collapse means the other paths' GATs stop receiving gradient
        beta_str = ' '.join(f'{p}:{b:.3f}' for p, b in zip(paths, beta.tolist()))
        print(f'Epoch {epoch}, train bpr {train_loss:.4f}, val bpr {val_loss:.4f} | beta {beta_str}'
              + (' (beta warmup)' if warmup else ''))
        if warmup:
            continue  # uniform-beta epochs are not eligible as the best checkpoint:
                      # the final model fuses with the LEARNED beta, so saving a
                      # warmup state would evaluate a model that was never trained
        if val_loss <= best_val_loss:
            # any improvement updates the best checkpoint, but only an improvement
            # bigger than --min-delta resets patience: val bpr creeping down by
            # ~1e-4/epoch otherwise defeats early stopping entirely (a 200-epoch
            # run where the last 100 epochs buy < 0.004)
            wait = 0 if best_val_loss - val_loss > args.min_delta else wait + 1
            best_epoch, best_val_loss = epoch, val_loss
            torch.save({'model': model.state_dict(), 'tower': tower.state_dict()},
                       f'{checkpt_file}.pkl')
        else:
            wait += 1
        if wait >= args.patience:
            print(f'Early stop at epoch {epoch} (no val improvement > {args.min_delta} '
                  f'for {args.patience} epochs)')
            break

    print('average train times', sum(train_times) / len(train_times))
    print(f'Best epoch {best_epoch}, val bpr {best_val_loss:.4f}')

    model.semantic.force_uniform = False  # test always fuses with the learned beta
    ckpt = torch.load(f'{checkpt_file}.pkl', map_location='cpu')
    model.load_state_dict(ckpt['model']); tower.load_state_dict(ckpt['tower'])
    model = model.to(device); tower = tower.to(device)

    with torch.no_grad():
        model.eval(); tower.eval()
        dst_emb, beta = model(graphs, p_feat)
        src_emb = user_emb_all()
        pos_scores = score_edges(src_emb, dst_emb, test_edges[:, 0], test_edges[:, 1])
        neg_scores = score_edges(src_emb, dst_emb, test_edges[:, 0], test_negs)
        results = evaluate_ranking(pos_scores, neg_scores)

    print('Semantic attention over meta-paths:')
    for p, b in zip(paths, beta.tolist()):
        print(f'  {p}: {b:.4f}')
    print('Test ranking metrics:')
    for k, v in results.items():
        print(f'  {k}: {v:.4f}')
    print_by_user(evaluate_ranking_by_user(
        pos_scores, neg_scores, link_split['test_edges'], link_split['test_negs']))
    deg = np.asarray(link_split['train_adj_src2dst'].sum(axis=1)).ravel()
    strat = evaluate_ranking_stratified(pos_scores, neg_scores,
                                        deg[link_split['test_edges'][:, 0]])
    print('By user training degree:')
    print_stratified(strat)
    twin = report_twin_breakdown(pos_scores, neg_scores, link_split, link_split['test_edges'])

    if args.save_emb:
        save_path = args.save_emb
        if save_path == 'auto':
            save_path = f'./output/{args.dataset}/han_emb.pt'
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        torch.save({'model': 'HAN', 'dataset': args.dataset,
                    'target_relation': args.target_relation, 'paths': paths,
                    'beta': beta.detach().cpu(), 'split_seed': args.split_seed,
                    'P_emb': dst_emb.detach().cpu(), 'U_emb': src_emb.detach().cpu(),
                    'user_tower': args.user_tower,
                    'metrics': results, 'metrics_twin': twin,
                    'best_val_loss': best_val_loss, 'best_epoch': best_epoch,
                    'epochs_run': epochs_run, 'train_time_s': float(sum(train_times)),
                    'data': data_prov,
                    'graphs': {p: graph_sidecar(graph_dir, p, args.topk, args.split_seed) for p in paths},
                    'config': vars(args)}, save_path)
        print(f'Embeddings saved to {save_path} (serve with recommend.py)')
    os.remove(f'{checkpt_file}.pkl')
    return results


def parse_args():
    ap = argparse.ArgumentParser(
        description='HAN link prediction with LMSPS-discovered meta-paths.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--root', type=str, default='data')
    ap.add_argument('--dataset', type=str, default='AMAZON')
    ap.add_argument('--target-relation', type=str, default='U,P')
    ap.add_argument('--paths', type=str, default='',
                    help='comma-separated meta-paths (overrides --arch)')
    ap.add_argument('--arch', type=str, default='',
                    help='arch.py key holding the path list, e.g. amazon_lp_han')
    ap.add_argument('--topk', type=int, default=20,
                    help='must match build_metapath_graphs.py --topk')
    ap.add_argument('--graph-dir', type=str, default='',
                    help='directory holding <path>_top<k>_split<s>.npz (default '
                         '<root>/<dataset>/metapath_graphs). Graph files are per path, so one '
                         'directory with every path built (e.g. data/AMAZON/metapath_graphs_all_unique) '
                         'serves every --arch without renaming folders')
    ap.add_argument('--split-seed', type=int, default=1,
                    help='must match the search and build --split-seed (the LMSPS fleets use 1)')
    ap.add_argument('--allow-legacy-data', action='store_true',
                    help='run on data without meta.json (from before the AMAZON rebuild), and on '
                         'graphs that do not record their data, instead of stopping')
    ap.add_argument('--num-neg-eval', type=int, default=99)
    ap.add_argument('--hidden', type=int, default=512)
    ap.add_argument('--num-heads', type=int, default=8)
    ap.add_argument('--feat-drop', type=float, default=0.2)
    ap.add_argument('--attn-drop', type=float, default=0.2)
    ap.add_argument('--semantic-tau', type=float, default=1.0,
                    help='temperature on the semantic-attention softmax. 1.0 = paper-faithful. '
                         'If the per-epoch beta log shows an early one-hot collapse (softmax '
                         'saturation starves the other paths of gradient), retry with 4-8')
    ap.add_argument('--beta-warmup', type=int, default=0,
                    help='train the first N epochs with beta FROZEN UNIFORM so every path\'s GAT '
                         'learns before the semantic attention judges it -- counters the '
                         'rich-get-richer collapse where paths scored low at init never receive '
                         'gradient. Best-checkpoint/early-stop tracking starts after warmup. '
                         '0 = off (paper-faithful)')
    ap.add_argument('--independent-init', action='store_true',
                    help='start every path\'s GAT from its own random weights (the behaviour before '
                         '2026-10-27). By default all GATs start from the same weights, so the path '
                         'weights and the result do not depend on the order of the paths in the list')
    ap.add_argument('--no-grad-ckpt', action='store_true',
                    help='disable gradient checkpointing of the per-path GATs. Checkpointing is ON '
                         'by default: it cuts peak training VRAM by several GB (recomputes GAT '
                         'internals in backward, ~+30%% step time) with bit-identical results')
    ap.add_argument('--vram-fraction', type=float, default=0.9,
                    help='cap the CUDA allocator at this fraction of total VRAM, leaving headroom '
                         'for the Windows display driver -- overshoot then raises a clean CUDA OOM '
                         'instead of crashing the machine. 0 disables the cap')
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--weight-decay', type=float, default=0.0)
    ap.add_argument('--stage', type=int, default=200, help='max epochs')
    ap.add_argument('--save-emb', type=str, default='auto',
                    help="where to save P/U embeddings for recommend.py ('auto' = ./output/<dataset>/han_emb.pt, '' disables)")
    ap.add_argument('--user-tower', type=str, default='embedding', choices=['embedding', 'feats'],
                    help="user-side encoder: 'embedding' = free per-user trained vectors (same per-user "
                         "capacity as metapath2vec/GAT -- use this for model comparisons); 'feats' = "
                         "linear projection of the user hash features (inductive; original behavior)")
    ap.add_argument('--patience', type=int, default=20)
    ap.add_argument('--min-delta', type=float, default=5e-4,
                    help='minimum val-bpr improvement that resets patience. Smaller improvements '
                         'still update the best checkpoint but count toward patience, so the run '
                         'stops once progress is noise-level instead of grinding to --stage. '
                         '0 restores the strict any-improvement-resets behavior')
    ap.add_argument('--batch-size', type=int, default=100000,
                    help='edges per BPR step (each step does one full-graph forward)')
    ap.add_argument('--paper-config', action='store_true',
                    help='use the hyperparameters of Wang et al. (WWW-2019) Sec. 5.3: lr 0.005, '
                         'weight-decay 0.001, feat/attn dropout 0.6, hidden 64 (8 heads x 8), '
                         'patience 100, min-delta 0, no beta warmup, tau 1. Any of these flags '
                         'passed explicitly on the command line keeps its explicit value. '
                         'Without this flag the defaults above are the repo\'s own tuning')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args()
    if args.paper_config:
        args = apply_paper_config(args, sys.argv[1:])
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    main(args)
