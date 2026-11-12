#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Recommendation API: the queries of recommend.py over HTTP, as JSON, for a front end.

Serves saved results (the .pt files of train_han.py, train_gat.py, train_metapath2vec.py and
popularity_baseline.py) with the same scoring as recommend.py: a user's products by dot product,
similar products by cosine. Python's standard library on top of numpy and torch, so there is
nothing new to install.

    python recommend_api.py --users-csv <dir>/filtered_users.csv --products-csv <dir>/filtered_products.csv
    python recommend_api.py --emb "output/AMAZON/v3/han_*_s1.pt" --port 8000

The category descriptions come from filtered_categories.csv and filtered_category_paths.csv, read by
default from the folder of --products-csv (--categories-csv and --category-paths-csv name others);
without them every product's category is null. They are for display only: no model reads them.

Without --emb it serves the representative results of README "Example recommendations" (HAN and
metapath2vec at both sizes, plus GAT). It listens on 127.0.0.1, this machine only, unless --host
says otherwise, and allows calls from any origin (--cors) so a front end on another port can use it.

Endpoints (GET; JSON; an error is {"error": "..."} with status 400 or 404):
  /api/health                                    status, the number of models, products with a category
  /api/models                                    the models: name, model, kind of path set, paths, test metrics
  /api/users?min_reviewed=6&max_reviewed=20&limit=20&seed=0
                                                 a sample of users with that many reviewed products
  /api/users/<customer id>                       the products the user reviewed
  /api/users/<customer id>/recommendations?models=<name>,<name>&k=10&exclude_seen=true
  /api/products?q=<words>&limit=20               products whose ASIN or title contains every word
  /api/products/<ASIN>                           one product, with all its category paths
  /api/products/<ASIN>/similar?models=<name>,<name>&k=10

A model name is its file name without .pt (see /api/models); without models= the first model is
used. <customer id> may also be a user index, as in recommend.py. exclude_seen (default true)
hides every product the user reviewed, held-out test purchases included, so a list shows what the
model would recommend next, not whether it found what the user bought.

Every product in an answer is {asin, title, group, sales_rank, avg_rating, reviewers, category}:
the title, product group, Amazon sales rank (lower sells better) and average customer rating come
from --products-csv (null when it is not given or the value is unknown); reviewers is the number of
customers in the data who reviewed the product, held-out test purchases included; category is its
first Amazon category path, a list of names from the root down (null when it has none or the
category files are off). /api/products/<ASIN> also gives product.categories, every path in order.
"""

import argparse
import collections
import csv
import glob
import itertools
import json
import os
import random
import threading
import time
from array import array
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlsplit

import numpy as np
import torch

from recommend import (CATEGORY_TYPE, PRODUCT_TYPE, USER_TYPE, load_node_maps, load_seen,
                       load_user_ids, topk_products)

# the representative results (README "Example recommendations"): HAN and metapath2vec at 6 paths,
# then at 4 paths, then GAT. Each model's random draw is the median by its own validation BPR.
DEFAULT_EMB = [f'output/AMAZON/v3/{name}.pt' for name in (
    'han_amazon_lp_v3_f6_a1_s1', 'han_amazon_lp_manual_6_s1', 'han_amazon_lp_rand6_d3_s1',
    'm2v_amazon_lp_v3_f6_a1_s1', 'm2v_amazon_lp_manual_6_s1', 'm2v_amazon_lp_rand6_d2_s1',
    'han_amazon_lp_v3_f4_a1_s1', 'han_amazon_lp_manual_5_s1', 'han_amazon_lp_rand4_d4_s1',
    'm2v_amazon_lp_v3_f4_a1_s1', 'm2v_amazon_lp_manual_5_s1', 'm2v_amazon_lp_rand4_d2_s1',
    'gat_union_s1')]
MAX_K = 100


class ApiError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status
        self.message = message


def load_blob(path):
    """Memory-mapped, as in summarize_results.py: the embeddings are read when first used."""
    try:
        return torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    except TypeError:  # torch < 2.1: no mmap / weights_only arguments
        return torch.load(path, map_location='cpu')


def positive(value, cast):
    """A number from a CSV cell, or None when it is empty, not a number, or not above 0 (the data's
    way of saying unknown: 3 products have sales rank 0)."""
    try:
        number = cast(float(value))
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def load_products(products_csv):
    """ASIN -> {title, group, sales_rank, avg_rating} from filtered_products.csv ({} without the
    file). Sales rank and rating are Amazon's, as in the 2006 metadata."""
    products = {}
    if products_csv:
        with open(products_csv, 'r', encoding='utf-8-sig', newline='') as fh:
            for row in csv.DictReader(fh):
                asin = (row.get('ASIN') or '').strip()
                if asin:
                    products[asin] = {'title': (row.get('Title') or '').strip(),
                                      'group': (row.get('Group') or '').strip(),
                                      'sales_rank': positive(row.get('SalesRank'), int),
                                      'avg_rating': positive(row.get('AvgRating'), float)}
    return products


CATEGORIES_FILE = 'filtered_categories.csv'
CATEGORY_PATHS_FILE = 'filtered_category_paths.csv'


def category_files(products_csv, categories_csv='', category_paths_csv=''):
    """The two category files: the ones given, else those next to --products-csv. Returns
    (categories_csv, category_paths_csv, None), or (None, None, why categories are off)."""
    folder = os.path.dirname(os.path.abspath(products_csv)) if products_csv else ''
    files = [(given or (folder and os.path.join(folder, default)), flag)
             for given, default, flag in ((categories_csv, CATEGORIES_FILE, '--categories-csv'),
                                          (category_paths_csv, CATEGORY_PATHS_FILE, '--category-paths-csv'))]
    for path, flag in files:
        if not path:
            return None, None, f'no {flag} or --products-csv'
        if not os.path.isfile(path):
            return None, None, f'no {path}'
    return files[0][0], files[1][0], None


def load_category_paths(categories_csv, category_paths_csv, keep):
    """ASIN -> its Amazon category paths, each a tuple of names from the root down, for the ASINs in
    `keep` (filtered_categories.csv: ID, Name; filtered_category_paths.csv: ASIN, No, Seq,
    CategoryID, one row per level). A product's paths follow No and each path follows Seq. A level
    without a name is dropped (the raw data lists some categories by number only), and so is a path
    left with no names or with the same names as an earlier path. Display only: no model reads it."""
    names, index = [''], {}                           # names[0] = '' for an id the file lacks
    with open(categories_csv, 'r', encoding='utf-8-sig', newline='') as fh:
        for row in csv.DictReader(fh):
            index[(row.get('ID') or '').strip()] = len(names)
            names.append((row.get('Name') or '').strip())
    # one packed integer per row, (No * SEQS + Seq) * NAMES + name index, in an array per product:
    # 8 bytes a row instead of a tuple of three objects, so reading the 2.3 million rows of the
    # full data costs ~30 MB on top of the result (~45 MB) instead of ~170 MB, and sorting the
    # integers sorts the rows by No, then Seq
    SEQS, NAMES = 1000, len(names)
    levels = collections.defaultdict(lambda: array('q'))
    with open(category_paths_csv, 'r', encoding='utf-8-sig', newline='') as fh:
        for row in csv.DictReader(fh):
            asin = (row.get('ASIN') or '').strip()
            if asin not in keep:
                continue
            try:
                no, seq = int(row['No']), int(row['Seq'])
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= no and 0 <= seq < SEQS:
                levels[asin].append((no * SEQS + seq) * NAMES + index.get((row.get('CategoryID') or '').strip(), 0))
    paths = {}
    for asin, packed in levels.items():
        kept, seen = [], set()
        for _, group in itertools.groupby(sorted(packed), key=lambda code: code // (SEQS * NAMES)):
            path = tuple(names[code % NAMES] for code in group if names[code % NAMES])
            if path and path not in seen:
                seen.add(path)
                kept.append(path)
        if kept:
            paths[asin] = tuple(kept)
    return paths


def set_kind(name, paths):
    """The kind of meta-path set a result was trained on, from its file name."""
    if not paths:
        return 'none'               # GAT and popularity use no meta-paths
    if '_v3_' in name:
        return 'automatic (LMSPS)'
    if '_manual_' in name:
        return 'manual'
    if '_rand' in name:
        return 'random'
    return 'other'


class Catalog:
    """Products, users and who reviewed what: shared by every model."""

    def __init__(self, root, dataset, products_csv, users_csv, categories_csv=None, category_paths_csv=None):
        self.asin_to_pid, self.pid_to_asin, counts = load_node_maps(root, dataset)
        self.num_users = counts[USER_TYPE]
        self.products = load_products(products_csv)
        start = time.perf_counter()
        self.paths = (load_category_paths(categories_csv, category_paths_csv, self.asin_to_pid)
                      if categories_csv and category_paths_csv else {})
        self.category_seconds = time.perf_counter() - start
        self.cid_to_uid = load_user_ids(users_csv) if users_csv else {}
        self.uid_to_cid = {uid: cid for cid, uid in self.cid_to_uid.items()}
        self.seen = load_seen(root, dataset, counts[PRODUCT_TYPE], counts[CATEGORY_TYPE])
        self.reviewers = collections.Counter(p for items in self.seen.values() for p in items)

    def product(self, pid):
        asin = self.pid_to_asin[pid]
        details = self.products.get(asin, {})
        paths = self.paths.get(asin)
        return {'asin': asin, 'title': details.get('title', ''), 'group': details.get('group', ''),
                'sales_rank': details.get('sales_rank'), 'avg_rating': details.get('avg_rating'),
                'reviewers': self.reviewers.get(pid, 0), 'category': list(paths[0]) if paths else None}

    def categories(self, pid):
        """Every category path of a product, in order; [] when it has none."""
        return [list(path) for path in self.paths.get(self.pid_to_asin[pid], ())]

    def user_index(self, key):
        """A customer id, or a user index as recommend.py accepts it."""
        if key.isdigit() and int(key) < self.num_users and key not in self.cid_to_uid:
            return int(key)
        if key not in self.cid_to_uid:
            raise ApiError(404, f'user {key!r} not found (start the API with --users-csv, or use a user index)')
        return self.cid_to_uid[key]

    def user(self, uid):
        return {'id': self.uid_to_cid.get(uid, str(uid)), 'index': uid,
                'reviewed': len(self.seen.get(uid, ()))}


class Models:
    """The served results by name; each loaded once, memory-mapped."""

    def __init__(self, paths, catalog):
        self.catalog = catalog
        self.paths = {}
        for path in paths:
            name = os.path.splitext(os.path.basename(path))[0]
            if name in self.paths:
                raise SystemExit(f'two results named {name}: {self.paths[name]} and {path}')
            self.paths[name] = path
        self.loaded = {}
        self.lock = threading.Lock()

    def get(self, name):
        if name not in self.paths:
            raise ApiError(404, f'model {name!r} not served; see /api/models')
        with self.lock:
            if name not in self.loaded:
                blob = load_blob(self.paths[name])
                P = blob['P_emb'].detach().float().numpy()
                U = blob['U_emb'].detach().float().numpy()
                if P.shape[0] != len(self.catalog.pid_to_asin) or U.shape[0] != self.catalog.num_users:
                    raise ApiError(400, f'{name} has {P.shape[0]} products and {U.shape[0]} users, the '
                                        f'data {len(self.catalog.pid_to_asin)} and {self.catalog.num_users}: '
                                        f'trained on other data')
                cfg = blob.get('config') or {}
                metrics = blob.get('metrics') or {}
                self.loaded[name] = {
                    'P': P, 'U': U, 'P_norm': None,
                    'info': {'name': name, 'model': blob.get('model', '?'), 'arch': cfg.get('arch') or '-',
                             'kind': set_kind(name, blob.get('paths') or []),
                             'paths': list(blob.get('paths') or []), 'seed': cfg.get('seed'),
                             'test_hit10': metrics.get('Hit@10'), 'test_ndcg10': metrics.get('NDCG@10'),
                             'val_bpr': blob.get('best_val_loss')}}
            return self.loaded[name]

    def normed(self, name):
        """Unit-length product embeddings for cosine similarity (None for popularity's single
        number), computed on the first "similar products" query."""
        entry = self.get(name)
        with self.lock:
            if entry['P_norm'] is None and entry['P'].shape[1] > 1:
                norms = np.linalg.norm(entry['P'], axis=1, keepdims=True)
                entry['P_norm'] = entry['P'] / np.clip(norms, 1e-12, None)
        return entry['P_norm']


class Api:
    """Routes a GET path to its JSON answer; no HTTP here, so it can be tested directly."""

    def __init__(self, catalog, models):
        self.catalog = catalog
        self.models = models

    def route(self, raw_path):
        url = urlsplit(raw_path)
        seg = [unquote(s) for s in url.path.split('/') if s]
        q = {k: v[-1] for k, v in parse_qs(url.query).items()}
        if seg[:1] != ['api']:
            raise ApiError(404, 'not found: every endpoint starts with /api/')
        seg = seg[1:]
        if seg == ['health']:
            return {'status': 'ok', 'models': len(self.models.paths), 'products': len(self.catalog.pid_to_asin),
                    'users': self.catalog.num_users, 'categories': len(self.catalog.paths)}
        if seg == ['models']:
            return {'models': [self.models.get(name)['info'] for name in self.models.paths]}
        if seg == ['users']:
            return self.sample_users(q)
        if len(seg) == 2 and seg[0] == 'users':
            uid = self.catalog.user_index(seg[1])
            return {'user': self.catalog.user(uid),
                    'products': [self.catalog.product(p) for p in sorted(self.catalog.seen.get(uid, ()))]}
        if len(seg) == 3 and seg[0] == 'users' and seg[2] == 'recommendations':
            return self.for_user(seg[1], q)
        if seg == ['products']:
            return self.search(q)
        if len(seg) == 2 and seg[0] == 'products':
            pid = self.product_index(seg[1])
            return {'product': {**self.catalog.product(pid), 'categories': self.catalog.categories(pid)},
                    'reviewers': self.catalog.reviewers.get(pid, 0)}
        if len(seg) == 3 and seg[0] == 'products' and seg[2] == 'similar':
            return self.similar(seg[1], q)
        raise ApiError(404, f'no endpoint {url.path}')

    # ---- query parameters
    @staticmethod
    def number(q, key, default, lo, hi):
        try:
            value = int(q.get(key, default))
        except ValueError:
            raise ApiError(400, f'{key} must be a whole number')
        if not lo <= value <= hi:
            raise ApiError(400, f'{key} must be between {lo} and {hi}')
        return value

    @staticmethod
    def flag(q, key, default):
        value = q.get(key, str(default)).lower()
        if value not in ('true', 'false', '1', '0', 'yes', 'no'):
            raise ApiError(400, f'{key} must be true or false')
        return value in ('true', '1', 'yes')

    def model_names(self, q):
        names = [n for n in q.get('models', '').split(',') if n] or list(self.models.paths)[:1]
        if not names:
            raise ApiError(404, 'no model is served')
        return names

    def product_index(self, asin):
        if asin not in self.catalog.asin_to_pid:
            raise ApiError(404, f'product {asin!r} not found')
        return self.catalog.asin_to_pid[asin]

    def ranked(self, scores, k, exclude):
        top = [p for p in topk_products(scores, k, exclude) if p not in exclude and np.isfinite(scores[p])]
        return [{'rank': r, **self.catalog.product(int(p)), 'score': round(float(scores[p]), 6)}
                for r, p in enumerate(top, 1)]

    # ---- endpoints
    def for_user(self, key, q):
        uid = self.catalog.user_index(key)
        k = self.number(q, 'k', 10, 1, MAX_K)
        exclude_seen = self.flag(q, 'exclude_seen', True)
        exclude = self.catalog.seen.get(uid, set()) if exclude_seen else set()
        results = []
        for name in self.model_names(q):
            m = self.models.get(name)
            results.append({**m['info'], 'items': self.ranked(m['P'] @ m['U'][uid], k, exclude)})
        return {'user': self.catalog.user(uid), 'k': k, 'exclude_seen': exclude_seen, 'results': results}

    def similar(self, asin, q):
        pid = self.product_index(asin)
        k = self.number(q, 'k', 10, 1, MAX_K)
        results = []
        for name in self.model_names(q):
            P_norm = self.models.normed(name)
            if P_norm is None:
                raise ApiError(400, f'{name}: its product embedding is a single number (popularity), so '
                                    f'"similar products" means nothing for it')
            results.append({**self.models.get(name)['info'],
                            'items': self.ranked(P_norm @ P_norm[pid], k, {pid})})
        return {'product': self.catalog.product(pid), 'k': k, 'results': results}

    def sample_users(self, q):
        lo = self.number(q, 'min_reviewed', 6, 0, 10 ** 6)
        hi = self.number(q, 'max_reviewed', 20, 0, 10 ** 6)
        limit = self.number(q, 'limit', 20, 1, 1000)
        seed = self.number(q, 'seed', 0, 0, 2 ** 31 - 1)
        pool = sorted(uid for uid, items in self.catalog.seen.items() if lo <= len(items) <= hi)
        picked = random.Random(seed).sample(pool, min(limit, len(pool)))
        return {'matching': len(pool), 'users': [self.catalog.user(uid) for uid in sorted(picked)]}

    def search(self, q):
        words = q.get('q', '').lower().split()
        if not words:
            raise ApiError(400, 'give q=<words>')
        limit = self.number(q, 'limit', 20, 1, 1000)
        hits = []
        for pid, asin in enumerate(self.catalog.pid_to_asin):
            text = (asin + ' ' + self.catalog.products.get(asin, {}).get('title', '')).lower()
            if all(w in text for w in words):
                hits.append(pid)
                if len(hits) == limit:
                    break
        return {'products': [self.catalog.product(p) for p in hits]}


class Handler(BaseHTTPRequestHandler):
    api = None
    cors = '*'

    def do_OPTIONS(self):  # a browser's preflight before a cross-origin request
        self.reply(204, None)

    def do_GET(self):
        try:
            status, body = 200, self.api.route(self.path)
        except ApiError as e:
            status, body = e.status, {'error': e.message}
        except Exception as e:  # keep the server up and say what went wrong
            status, body = 500, {'error': f'{type(e).__name__}: {e}'}
        self.reply(status, body)

    def reply(self, status, body):
        data = b'' if body is None else json.dumps(body, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Access-Control-Allow-Origin', self.cors)
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        if body is not None:
            self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def make_server(api, host, port, cors='*'):
    handler = type('ApiHandler', (Handler,), {'api': api, 'cors': cors})
    return ThreadingHTTPServer((host, port), handler)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--emb', nargs='*', default=None,
                    help='results to serve (.pt files, globs ok); default: the representative results')
    ap.add_argument('--root', type=str, default='data')
    ap.add_argument('--dataset', type=str, default='AMAZON')
    ap.add_argument('--products-csv', type=str, default='',
                    help='filtered_products.csv, for titles and product groups')
    ap.add_argument('--users-csv', type=str, default='', help='filtered_users.csv, for customer ids')
    ap.add_argument('--categories-csv', type=str, default='',
                    help=f'{CATEGORIES_FILE}, the category names (default: next to --products-csv)')
    ap.add_argument('--category-paths-csv', type=str, default='',
                    help=f'{CATEGORY_PATHS_FILE}, the products\' category paths (default: next to --products-csv)')
    ap.add_argument('--host', type=str, default='127.0.0.1',
                    help='127.0.0.1 = this machine only; 0.0.0.0 = every network the machine is on')
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--cors', type=str, default='*',
                    help='the origin a browser front end may call from (* = any)')
    args = ap.parse_args()

    paths = []
    for pattern in (args.emb if args.emb else DEFAULT_EMB):
        found = sorted(glob.glob(pattern))
        if not found:
            print(f'skip {pattern}: not found')
        paths.extend(found)
    if not paths:
        raise SystemExit('no result to serve: pass --emb')
    categories_csv, category_paths_csv, off = category_files(args.products_csv, args.categories_csv,
                                                             args.category_paths_csv)
    catalog = Catalog(args.root, args.dataset, args.products_csv, args.users_csv, categories_csv, category_paths_csv)
    categories = (f'on ({len(catalog.paths)} products with a path, read in {catalog.category_seconds:.1f} s)'
                  if categories_csv else f'off ({off})')
    print(f'{len(catalog.pid_to_asin)} products, {catalog.num_users} users; '
          f'titles {"on" if catalog.products else "off (no --products-csv)"}, '
          f'customer ids {"on" if catalog.cid_to_uid else "off (no --users-csv)"}, categories {categories}')
    api = Api(catalog, Models(paths, catalog))
    for name in api.models.paths:   # load each once now: a wrong file fails here, not on a request
        info = api.models.get(name)['info']
        print(f'  {name}: {info["model"]}, {info["kind"]}, test Hit@10 {info["test_hit10"]}')
    server = make_server(api, args.host, args.port, args.cors)
    print(f'Serving on http://{args.host}:{args.port}/api/  (Ctrl+C stops it)')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    server.server_close()


if __name__ == '__main__':
    main()
