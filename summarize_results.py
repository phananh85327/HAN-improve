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

Model selection rule for the thesis: pick the best arch per group by VAL BPR
(lower is better), never by a test metric. The --group view aggregates runs with
the same (model, arch, preset, split, data) over run seeds as mean +/- std.

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


def fmt(v, nd=4):
    if v is None or v == '-':
        return '-'
    if isinstance(v, float):
        return f'{v:.{nd}f}'
    return str(v)


COLS = ['model', 'arch', 'preset', 'split', 'data', 'seed', 'best_epoch', 'val_bpr',
        'hit10', 'ndcg10', 'hit10_twin', 'hit10_notwin', 'ndcg10_notwin', 'file']


def print_markdown(rows, cols):
    print('| ' + ' | '.join(cols) + ' |')
    print('|' + '---|' * len(cols))
    for r in rows:
        print('| ' + ' | '.join(fmt(r.get(c)) for c in cols) + ' |')


def grouped(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r['model'], r['arch'], r['preset'], r['split'], r['data']), []).append(r)
    out = []
    for (model, arch, preset, split, data), rs in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0]))):
        g = {'model': model, 'arch': arch, 'preset': preset, 'split': split, 'data': data,
             'n_runs': len(rs), 'seeds': ','.join(str(r['seed']) for r in rs)}
        for k in ('val_bpr', 'hit10', 'ndcg10', 'hit10_twin', 'hit10_notwin', 'ndcg10_notwin'):
            vals = [r[k] for r in rs if isinstance(r.get(k), float)]
            if not vals:
                g[k] = '-'
            elif len(vals) == 1:
                g[k] = f'{vals[0]:.4f}'
            else:
                g[k] = f'{statistics.mean(vals):.4f} +/- {statistics.stdev(vals):.4f}'
        out.append(g)
    return out


GCOLS = ['model', 'arch', 'preset', 'split', 'data', 'n_runs', 'seeds', 'val_bpr', 'hit10',
         'ndcg10', 'hit10_twin', 'hit10_notwin', 'ndcg10_notwin']


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('globs', nargs='*', default=['output/AMAZON/*.pt'],
                    help='embedding blobs (globs ok)')
    ap.add_argument('--group', action='store_true',
                    help='aggregate over run seeds per (model, arch, preset, split)')
    ap.add_argument('--csv', type=str, default='', help='also write the per-run table to this CSV')
    ap.add_argument('--sort', type=str, default='val_bpr',
                    choices=['val_bpr', 'hit10', 'hit10_notwin', 'model', 'arch'],
                    help='sort key for the per-run table (val_bpr ascending, metrics descending)')
    args = ap.parse_args()

    paths = []
    for g in args.globs:
        paths.extend(sorted(glob.glob(g)))
    if not paths:
        sys.exit(f'no files match {args.globs}')

    rows = []
    for p in paths:
        try:
            rows.append(row_from_blob(p, load_blob(p)))
        except Exception as exc:  # a corrupt or foreign .pt must not kill the table
            print(f'skip {p}: {exc}', file=sys.stderr)

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
    print_markdown(rows, COLS)
    bad = [r for r in rows if r['warnings']]
    if bad:
        print('\nruns with warnings:')
        for r in bad:
            print(f"  {r['file']}: {r['warnings']}")

    if args.group:
        print('\nGrouped over run seeds (mean +/- std where > 1 run):\n')
        print_markdown(grouped(rows), GCOLS)

    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or '.', exist_ok=True)
        with open(args.csv, 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f'\nwrote {args.csv}')


if __name__ == '__main__':
    main()
