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

    python build_metapath_graphs.py --dataset AMAZON --paths "<printed>" --topk 20 --split-seed 1
    python train_han.py --dataset AMAZON --paths "<printed>" --split-seed 1

The fairer null for a search is a random pick from the search's OWN candidates: with
--from-record, every draw takes --size of the candidate paths stored in an LMSPS search
record (the 36 the fleets chose from), and --draws such sets are printed as arch.py
entries <prefix>_d1, _d2, ... Report the spread over the draws, never a single draw:

    python random_paths.py --from-record "<LMSPS>/hgb/logs/Fleet v3/final_ns4/search_ns4_seed1.json" \
        --size 4 --draws 5 --seed 0 --prefix amazon_lp_v3_rand4
"""

import argparse
import json
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


def draws_from_pool(pool, size, n_draws, seed):
    """n_draws different random sets of `size` paths from `pool`, each sorted."""
    rng = random.Random(seed)
    seen, out = set(), []
    while len(out) < n_draws:
        draw = tuple(sorted(rng.sample(sorted(pool), size)))
        if draw not in seen:
            seen.add(draw)
            out.append(draw)
    return out


def main():
    ap = argparse.ArgumentParser(
        description='Random schema-valid meta-paths (ablation baseline).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--num', type=int, default=5, help='number of paths')
    ap.add_argument('--min-len', type=int, default=4, help='letters incl. both endpoints')
    ap.add_argument('--max-len', type=int, default=6)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--from-record', type=str, default='',
                    help='draw from the candidates of this LMSPS search record instead of the schema')
    ap.add_argument('--size', type=int, default=4, help='paths per draw (--from-record)')
    ap.add_argument('--draws', type=int, default=5, help='number of draws (--from-record)')
    ap.add_argument('--prefix', type=str, default='amazon_lp_rand', help='arch.py key stem (--from-record)')
    args = ap.parse_args()

    if args.from_record:
        with open(args.from_record, encoding='utf-8') as f:
            pool = json.load(f)['candidates']
        print(f'    # random draws of {args.size} from the {len(pool)} candidates of {args.from_record} '
              f'(seed {args.seed}); report the spread over the draws')
        for i, draw in enumerate(draws_from_pool(pool, args.size, args.draws, args.seed), 1):
            print(f'    "{args.prefix}_d{i}": [\n        {list(draw)!r}, []],')
        return

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