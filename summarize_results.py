#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Collect every saved embedding blob (*_emb*.pt written by train_han.py,
train_metapath2vec.py, train_gat.py) into one results table.

Per blob: model, arch key (config.arch, else the path list), split seed, data
(the training-edge fingerprint the run recorded; '-' = saved before runs recorded
it), run seed, preset flags (--paper-config / --paper-walks), best epoch, best VAL
BPR, and per-edge Hit@10 / NDCG@10 overall, on "twin in train" test edges and on
"no twin in train" edges (see link_data.product_twin_groups). Blobs written
before 2026-10-23 lack metrics_twin / best_val_loss and show '-'; rerun
eval_emb.py on them if those columns are needed. A table that mixes data
versions prints a warning: rows are comparable only within one version.
With --full-ranking (needs the data under --root) every run is also ranked against
all products instead of 99 sampled negatives: hit10_full, ndcg10_full and
recall20_full (per user). Sampled metrics flatter popularity-like models.

Model selection rule for the thesis: pick the best arch per group by VAL BPR
(lower is better), never by a test metric. The --group view aggregates runs with
the same (model, arch, preset, variant, split, data) over run seeds as mean +/- std;
'variant' lists the settings that differ between runs of one model (e.g.
beta_warmup=0 vs 15), so different settings are never averaged as if they were seeds.

    python summarize_results.py                          # output/AMAZON/*.pt
    python summarize_results.py output/AMAZON/han_*.pt --group --csv logs/results.csv
"""

import argparse
import csv
import glob
import os
import statistics
import sys

import torch


def load_blob(path):
    try:
        return torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    except TypeError:  # torch < 2.1: no mmap argument
        return torch.load(path, map_location='cpu')


def row_from_blob(path, blob):
    cfg = blob.get('config') or {}
    arch = cfg.get('arch') or ''
    paths = blob.get('paths') or []
    if not arch:
        arch = ','.join(paths) if paths else '-'
    preset = []
    if cfg.get('paper_config'):
        preset.append('paper-config')
    if cfg.get('paper_walks'):
        preset.append('paper-walks')
    m = blob.get('metrics') or {}
    tw = blob.get('metrics_twin') or {}
    twin = tw.get('twin in train', {})
    notwin = tw.get('no twin in train', {})
    bv = blob.get('best_val_loss')
    be = blob.get('best_epoch')
    return {
        'file': os.path.basename(path),
        'model': blob.get('model', '?'),
        'arch': arch,
        'n_paths': len(paths),
        'preset': '+'.join(preset) or '-',
        'split': blob.get('split_seed', '-'),
        'data': (blob.get('data') or {}).get('train_edges_hash') or '-',
        'seed': cfg.get('seed', '-'),
        'best_epoch': be if be is not None else '-',
        'val_bpr': bv if bv is not None else None,
        'hit10': m.get('Hit@10'),
        'ndcg10': m.get('NDCG@10'),
        'hit10_twin': twin.get('Hit@10'),
        'hit10_notwin': notwin.get('Hit@10'),
        'ndcg10_notwin': notwin.get('NDCG@10'),
        'n_twin': twin.get('n_edges'),
        'n_notwin': notwin.get('n_edges'),
        'warnings': '; '.join(blob.get('warnings') or []),
    }


# config entries that do not change what a run computes, or that the table shows elsewhere
NOT_A_VARIANT = {'seed', 'save_emb', 'gpu', 'cpu', 'vram_fraction', 'root', 'graph_dir', 'dataset',
                 'arch', 'paths', 'eval_only', 'no_grad_ckpt', 'allow_legacy_data', 'split_seed',
                 'paper_config', 'paper_walks'}


def add_variants(rows, cfgs):
    """'variant' per row: the config entries that differ between runs of the same model and
    preset (e.g. beta_warmup=0 vs 15), so different settings are never averaged as seeds."""
    by_model = {}
    for r, cfg in zip(rows, cfgs):
        by_model.setdefault((r['model'], r['preset']), []).append((r, cfg))
    for members in by_model.values():
        keys = sorted({k for _, cfg in members for k in cfg} - NOT_A_VARIANT)
        differ = [k for k in keys if len({repr(cfg.get(k)) for _, cfg in members}) > 1]
        for r, cfg in members:
            r['variant'] = ', '.join(f'{k}={cfg.get(k)}' for k in differ) or '-'


def add_full_ranking(rows, blobs, root, device):
    """hit10_full / ndcg10_full / recall20_full per row, from the blob's embeddings and the
    split rebuilt from the data under root; '-' for a blob trained on other data."""
    from data_loader import data_loader
    from link_data import build_link_split, edges_fingerprint
    from han_utils import NODE_TYPE_IDS, evaluate_full_ranking
    splits = {}
    for r, blob in zip(rows, blobs):
        key = (blob['dataset'], blob.get('target_relation', 'U,P'), blob['split_seed'])
        if key not in splits:
            src, dst = key[1].split(',')
            tids = NODE_TYPE_IDS[key[0]]
            ls = build_link_split(data_loader(f'{root}/{key[0]}'), tids[src], tids[dst], seed=key[2])
            splits[key] = (ls, edges_fingerprint(ls['train_edges']))
        ls, fp = splits[key]
        same_data = r['data'] == fp if r['data'] != '-' else \
            (blob['P_emb'].shape[0], blob['U_emb'].shape[0]) == (ls['num_dst'], ls['num_src'])
        if not same_data:
            print(f"full ranking skipped for {r['file']}: trained on other data", file=sys.stderr)
            continue
        with torch.no_grad():
            res = evaluate_full_ranking(blob['U_emb'].detach().float().to(device),
                                        blob['P_emb'].detach().float().to(device),
                                        ls['test_edges'], ls['full_adj_src2dst'], ks=(10, 20))
        r['hit10_full'], r['ndcg10_full'], r['recall20_full'] = res['Hit@10'], res['NDCG@10'], res['UserRecall@20']


def fmt(v, nd=4):
    if v is None or v == '-':
        return '-'
    if isinstance(v, float):
        return f'{v:.{nd}f}'
    return str(v)


COLS = ['model', 'arch', 'preset', 'variant', 'split', 'data', 'seed', 'best_epoch', 'val_bpr',
        'hit10', 'ndcg10', 'hit10_twin', 'hit10_notwin', 'ndcg10_notwin', 'file']


def print_markdown(rows, cols):
    print('| ' + ' | '.join(cols) + ' |')
    print('|' + '---|' * len(cols))
    for r in rows:
        print('| ' + ' | '.join(fmt(r.get(c)) for c in cols) + ' |')


def grouped(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r['model'], r['arch'], r['preset'], r['variant'], r['split'], r['data']), []).append(r)
    out = []
    for (model, arch, preset, variant, split, data), rs in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0]))):
        g = {'model': model, 'arch': arch, 'preset': preset, 'variant': variant, 'split': split,
             'data': data, 'n_runs': len(rs), 'seeds': ','.join(str(r['seed']) for r in rs)}
        for k in ('val_bpr', 'hit10', 'ndcg10', 'hit10_twin', 'hit10_notwin', 'ndcg10_notwin',
                  'hit10_full', 'ndcg10_full', 'recall20_full'):
            vals = [r[k] for r in rs if isinstance(r.get(k), float)]
            if not vals:
                g[k] = '-'
            elif len(vals) == 1:
                g[k] = f'{vals[0]:.4f}'
            else:
                g[k] = f'{statistics.mean(vals):.4f} +/- {statistics.stdev(vals):.4f}'
        out.append(g)
    return out


GCOLS = ['model', 'arch', 'preset', 'variant', 'split', 'data', 'n_runs', 'seeds', 'val_bpr', 'hit10',
         'ndcg10', 'hit10_twin', 'hit10_notwin', 'ndcg10_notwin']


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('globs', nargs='*', default=['output/AMAZON/*.pt'],
                    help='embedding blobs (globs ok)')
    ap.add_argument('--group', action='store_true',
                    help='aggregate over run seeds per (model, arch, preset, split)')
    ap.add_argument('--csv', type=str, default='', help='also write the per-run table to this CSV')
    ap.add_argument('--full-ranking', action='store_true',
                    help='add full-ranking columns (every product as a candidate); needs the data '
                         'under --root; seconds per run on a GPU, a minute or two on a CPU')
    ap.add_argument('--root', type=str, default='data')
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--sort', type=str, default='val_bpr',
                    choices=['val_bpr', 'hit10', 'hit10_notwin', 'model', 'arch'],
                    help='sort key for the per-run table (val_bpr ascending, metrics descending)')
    args = ap.parse_args()

    paths = []
    for g in args.globs:
        paths.extend(sorted(glob.glob(g)))
    if not paths:
        sys.exit(f'no files match {args.globs}')

    rows, blobs = [], []
    for p in paths:
        try:
            blob = load_blob(p)
            rows.append(row_from_blob(p, blob))
            blobs.append(blob)
        except Exception as exc:  # a corrupt or foreign .pt must not kill the table
            print(f'skip {p}: {exc}', file=sys.stderr)
    add_variants(rows, [b.get('config') or {} for b in blobs])
    cols, gcols = COLS, GCOLS
    if args.full_ranking:
        device = 'cpu' if args.cpu or not torch.cuda.is_available() else f'cuda:{args.gpu}'
        add_full_ranking(rows, blobs, args.root, device)
        full = ['hit10_full', 'ndcg10_full', 'recall20_full']
        cols = COLS[:-1] + full + COLS[-1:]
        gcols = GCOLS + full

    def key(r):
        v = r.get(args.sort)
        if args.sort in ('model', 'arch'):
            return (0, str(v))
        if not isinstance(v, float):
            return (1, 0.0)
        return (0, v if args.sort == 'val_bpr' else -v)

    rows.sort(key=lambda r: (r['model'], key(r)))
    versions = {}
    for r in rows:
        versions[(r['split'], r['data'])] = versions.get((r['split'], r['data']), 0) + 1
    print(f'{len(rows)} runs; data (split, training-edge fingerprint): '
          + ', '.join(f'split {s} {d} x{n}' for (s, d), n in versions.items()))
    if len(versions) > 1:
        print(f'WARNING: these runs used {len(versions)} different data versions or splits '
              f"('-' = saved before runs recorded their data); compare rows only within one")
    print()
    print_markdown(rows, cols)
    bad = [r for r in rows if r['warnings']]
    if bad:
        print('\nruns with warnings:')
        for r in bad:
            print(f"  {r['file']}: {r['warnings']}")

    if args.group:
        print('\nGrouped over run seeds (mean +/- std where > 1 run):\n')
        print_markdown(grouped(rows), gcols)

    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or '.', exist_ok=True)
        with open(args.csv, 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)), restval='-')
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f'\nwrote {args.csv}')


if __name__ == '__main__':
    main()
