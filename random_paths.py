#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Seeded random meta-path generator for the ablation row "HAN with RANDOM valid
paths": if automatically DISCOVERED paths carry real signal, HAN trained on them
must beat HAN trained on schema-valid random paths of the same count and length
range. Sampling walks the SCHEMA graph only (never the data), so there are no
split or leakage concerns here.

    python random_paths.py --num 5 --min-len 4 --max-len 6 --seed 0

prints a --paths string plus a ready-to-paste arch.py entry. Then materialize
and train exactly like any other path set:

    python build_metapath_graphs.py --dataset AMAZON --paths "<printed>" --topk 20 --split-seed 0
    python train_han.py --dataset AMAZON --paths "<printed>" --split-seed 0
"""

import argparse
import random

# one-hop reachability between node-type letters; mirrors
# build_metapath_graphs.build_base_matrices (U connects only to P)
SCHEMA = {'P': ['P', 'C', 'U'], 'C': ['P', 'C'], 'U': ['P']}
TARGET = 'P'


def random_path(rng, length):
    """Uniform random schema walk of `length` letters, rejection-sampled until
    it ends back at the target type (required for a HAN neighbor graph)."""
    while True:
        p = [TARGET]
        for _ in range(length - 1):
            p.append(rng.choice(SCHEMA[p[-1]]))
        if p[-1] == TARGET:
            return ''.join(p)


def main():
    ap = argparse.ArgumentParser(
        description='Random schema-valid meta-paths (ablation baseline).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--num', type=int, default=5, help='number of paths')
    ap.add_argument('--min-len', type=int, default=4, help='letters incl. both endpoints')
    ap.add_argument('--max-len', type=int, default=6)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    seen, paths = set(), []
    while len(paths) < args.num:
        p = random_path(rng, rng.randint(args.min_len, args.max_len))
        canon = min(p, p[::-1])   # a path and its reverse materialize the same graph
        if canon in seen:
            continue
        seen.add(canon)
        paths.append(p)

    print('--paths "' + ','.join(paths) + '"')
    print()
    print('arch.py entry:')
    print(f'    "amazon_lp_random{args.seed}": [')
    print(f'        {paths!r}, []],')


if __name__ == '__main__':
    main()