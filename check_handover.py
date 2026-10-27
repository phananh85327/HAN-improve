#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Is this repo's data the data an LMSPS search ran on?

The meta-path sets in arch.py come from LMSPS searches. Each search record
(train_search.py --search-json) stores the node counts and a fingerprint of its
training edges, which depends on the data and on the split seed. This script
rebuilds the same split from this repo's data and compares. Run it once before
training on a new set of arch.py keys, against any one record of the fleet that
produced them:

    python check_handover.py "../LMSPS/hgb/logs/Fleet v3/final_ns4/search_ns4_seed1.json"

Exit code 0 = same data and split, 1 = different, 2 = the record cannot tell
(written before searches recorded their data).
"""

import argparse
import json
import os
import sys

from data_loader import data_loader
from link_data import build_link_split, data_provenance
from han_utils import NODE_TYPE_IDS


def compare(record, here):
    """Mismatches between a search record's 'data' entry and this repo's data_provenance."""
    theirs = record['data']
    diffs = []
    ours = {str(t): c for t, c in here['node_counts'].items()}      # JSON keys are strings
    searched = {str(t): c for t, c in theirs['node_counts'].items()}
    if searched != ours:
        diffs.append(f"node counts per type: search {searched}, here {ours}")
    if theirs['num_target_edges'] != here['num_target_edges']:
        diffs.append(f"target edges: search {theirs['num_target_edges']}, here {here['num_target_edges']}")
    if theirs['train_edges_hash'] != here['train_edges_hash']:
        diffs.append(f"training-edge fingerprint: search {theirs['train_edges_hash']}, "
                     f"here {here['train_edges_hash']}")
    return diffs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('record', help='an LMSPS search record (JSON written by train_search.py --search-json)')
    ap.add_argument('--root', type=str, default='data')
    args = ap.parse_args()

    with open(args.record, encoding='utf-8') as f:
        record = json.load(f)
    if not record.get('data'):
        print(f'{args.record} does not record its data (written before searches did): cannot check')
        return 2
    dataset = record.get('dataset', 'AMAZON')
    split_seed = record.get('split_seed', record.get('args', {}).get('split_seed'))
    src_name, dst_name = record.get('target_relation', 'U,P').split(',')
    tids = NODE_TYPE_IDS[dataset]
    data_dir = os.path.join(args.root, dataset)
    dl = data_loader(data_dir)
    link_split = build_link_split(dl, tids[src_name], tids[dst_name], seed=split_seed)
    here = data_provenance(data_dir, dl, link_split)

    diffs = compare(record, here)
    if diffs:
        print(f'DIFFERENT: {data_dir} with split {split_seed} is not the data the search ran on')
        for d in diffs:
            print(f'  {d}')
        print('Copy the data the search used into this repo (and rebuild the graphs) before training.')
        return 1
    print(f"SAME: {data_dir} with split {split_seed} is the search's data "
          f"(training-edge fingerprint {here['train_edges_hash']})")
    return 0


if __name__ == '__main__':
    sys.exit(main())
