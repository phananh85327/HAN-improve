"""recommend_api.py on a synthetic dataset: every endpoint answers as documented, the lists match
the scores recommend.py would compute, the category paths are read as the extraction writes them
(and leave every list as it was), and the HTTP layer sends JSON with the CORS headers.

    python tests/test_recommend_api.py      (or: python -m pytest tests)
"""
import csv
import json
import os
import sys
import tempfile
import threading
import urllib.error
import urllib.request

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recommend_api import Api, ApiError, Catalog, Models, category_files, make_server  # noqa: E402
from test_data_handover import N_P, N_U, make_dataset  # noqa: E402

GROUPS = ['Book', 'DVD', 'Music', 'Video']
CLASSICS = 'Fiction, "Classics"'        # a name the CSV must quote


def write_categories(tmp):
    """filtered_categories.csv and filtered_category_paths.csv as the extraction writes them, next to
    products.csv. Every product has one path (its group, then a shelf), except: n3 has two paths
    whose rows are out of order; n4 has a level without a name, a duplicate path and a path of
    unnamed levels only; n5 has no rows. One row names a product the data does not have."""
    names = {'10': 'Books', '11': 'Programming', '12': CLASSICS, '13': ''}
    names.update({f'2{g}': group for g, group in enumerate(GROUPS)})
    names.update({f'3{s}': f'Shelf {s}' for s in range(3)})
    rows = []
    for i in range(N_P):
        if i not in (3, 4, 5):
            rows += [(f'n{i}', 0, 0, f'2{i % 4}'), (f'n{i}', 0, 1, f'3{i % 3}')]
    rows += [('n3', 1, 1, '12'), ('n3', 0, 1, '11'), ('n3', 1, 0, '10'), ('n3', 0, 0, '10')]
    rows += [('n4', 0, 0, '13'), ('n4', 0, 1, '10'), ('n4', 0, 2, '11'),
             ('n4', 1, 0, '10'), ('n4', 1, 1, '11'), ('n4', 2, 0, '13'), ('gone', 0, 0, '10')]
    with open(os.path.join(tmp, 'filtered_categories.csv'), 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerows([('ID', 'Name'), *names.items()])
    with open(os.path.join(tmp, 'filtered_category_paths.csv'), 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerows([('ASIN', 'No', 'Seq', 'CategoryID'), *rows])


def first_path(asin):
    """The category every answer should carry for a product of write_categories."""
    i = int(asin[1:])
    return ['Books', 'Programming'] if i in (3, 4) else None if i == 5 else [GROUPS[i % 4], f'Shelf {i % 3}']


def build(tmp, categories=True):
    """Synthetic data, its CSVs (with the two category files unless categories=False), one HAN-like
    result and the popularity result."""
    make_dataset(tmp)
    with open(os.path.join(tmp, 'products.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['ID', 'ASIN', 'Discontinued', 'Title', 'Group', 'SalesRank', 'AvgRating'])
        for i in range(N_P):   # every 7th product has no sales rank (0, as in the data)
            w.writerow([i, f'n{i}', 0, f'Product number {i}', GROUPS[i % 4], (i + 1) * 10 if i % 7 else 0,
                        4.5 if i % 2 else 3])
    with open(os.path.join(tmp, 'users.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['ID', 'Name'])
        for i in range(N_U):
            w.writerow([f'C{i:04d}', f'C{i:04d}'])
    g = torch.Generator().manual_seed(0)
    han = os.path.join(tmp, 'han_amazon_lp_manual_5_s1.pt')
    torch.save({'model': 'HAN', 'paths': ['PP', 'PUP', 'PCP', 'PPUP'], 'P_emb': torch.randn(N_P, 8, generator=g),
                'U_emb': torch.randn(N_U, 8, generator=g), 'metrics': {'Hit@10': 0.5, 'NDCG@10': 0.3},
                'best_val_loss': 0.4, 'config': {'arch': 'amazon_lp_manual_5', 'seed': 1}}, han)
    pop = os.path.join(tmp, 'popularity_s1.pt')
    torch.save({'model': 'popularity', 'paths': [], 'P_emb': torch.arange(N_P, dtype=torch.float32)[:, None],
                'U_emb': torch.ones(N_U, 1), 'metrics': {'Hit@10': 0.4}, 'config': {'seed': 1}}, pop)
    products = os.path.join(tmp, 'products.csv')
    if categories:
        write_categories(tmp)
    categories_csv, category_paths_csv, _ = category_files(products)
    catalog = Catalog(tmp, 'AMAZON', products, os.path.join(tmp, 'users.csv'), categories_csv, category_paths_csv)
    return Api(catalog, Models([han, pop], catalog)), han


def status(api, path):
    try:
        api.route(path)
    except ApiError as e:
        return e.status
    return 200


def test_endpoints():
    with tempfile.TemporaryDirectory() as tmp:
        api, han = build(tmp)
        assert api.route('/api/health')['models'] == 2
        info = {m['name']: m for m in api.route('/api/models')['models']}
        assert info['han_amazon_lp_manual_5_s1']['kind'] == 'manual' and info['popularity_s1']['kind'] == 'none'

        # a user's list: the same ranking as recommend.py's dot product, reviewed products left out
        r = api.route('/api/users/C0003/recommendations?models=han_amazon_lp_manual_5_s1&k=7')
        items = r['results'][0]['items']
        blob = torch.load(han)
        scores = (blob['P_emb'] @ blob['U_emb'][3]).numpy()
        seen = api.catalog.seen[3]
        expect = [p for p in np.argsort(-scores) if p not in seen][:7]
        assert [it['asin'] for it in items] == [f'n{p}' for p in expect]
        assert r['user'] == {'id': 'C0003', 'index': 3, 'reviewed': len(seen)} and items[0]['group']
        kept = api.route('/api/users/3/recommendations?k=100&exclude_seen=false')['results'][0]['items']
        assert [it['asin'] for it in kept] == [f'n{p}' for p in np.argsort(-scores)[:100]]

        # several models at once; popularity gives the most-reviewed unseen products
        both = api.route('/api/users/C0003/recommendations?models=han_amazon_lp_manual_5_s1,popularity_s1&k=3')
        assert [x['name'] for x in both['results']] == ['han_amazon_lp_manual_5_s1', 'popularity_s1']
        assert both['results'][1]['items'][0]['asin'] == f'n{max(set(range(N_P)) - seen)}'

        # similar products: cosine, never the product itself; popularity refuses
        sim = api.route('/api/products/n5/similar?k=5')['results'][0]['items']
        assert len(sim) == 5 and 'n5' not in {it['asin'] for it in sim}
        assert status(api, '/api/products/n5/similar?models=popularity_s1') == 400

        assert api.route('/api/products?q=number 12')['products'][0]['asin'] == 'n12'
        assert api.route('/api/products/n7')['reviewers'] == sum(7 in s for s in api.catalog.seen.values())

        # every product in an answer carries its details; a 0 in the CSV means unknown
        n7 = sum(7 in s for s in api.catalog.seen.values())
        assert api.route('/api/products/n7')['product'] == {
            'asin': 'n7', 'title': 'Product number 7', 'group': 'Video', 'sales_rank': None,
            'avg_rating': 4.5, 'reviewers': n7, 'category': ['Video', 'Shelf 1'],
            'categories': [['Video', 'Shelf 1']]}
        assert api.route('/api/products/n12')['product']['sales_rank'] == 130
        first = items[0]
        assert set(first) == {'rank', 'asin', 'title', 'group', 'sales_rank', 'avg_rating', 'reviewers',
                              'category', 'score'}
        assert first['reviewers'] == api.catalog.reviewers[int(first['asin'][1:])]
        assert api.route('/api/users/C0001')['products'][0]['avg_rating'] in (3.0, 4.5)

        # categories: the first path on every product of every answer ...
        answers = [items, kept, both['results'][1]['items'], sim,
                   api.route('/api/products?q=number&limit=200')['products'],
                   api.route('/api/users/C0001')['products']]
        assert all(p['category'] == first_path(p['asin']) for answer in answers for p in answer)
        # ... and every path, in No and Seq order, on the product's own page
        n3 = api.route('/api/products/n3')['product']
        assert n3['categories'] == [['Books', 'Programming'], ['Books', CLASSICS]]
        assert n3['category'] == ['Books', 'Programming']
        # a level without a name is dropped, a repeated path kept once, an all-unnamed path dropped
        assert api.route('/api/products/n4')['product']['categories'] == [['Books', 'Programming']]
        n5 = api.route('/api/products/n5')['product']
        assert n5['category'] is None and n5['categories'] == []
        assert api.route('/api/health')['categories'] == N_P - 1
        assert len(api.route('/api/users/C0001')['products']) == len(api.catalog.seen[1])
        sample = api.route('/api/users?min_reviewed=5&max_reviewed=6&limit=4')
        assert len(sample['users']) <= 4 and all(5 <= u['reviewed'] <= 6 for u in sample['users'])

        for path, code in (('/api/users/nobody/recommendations', 404), ('/api/products/x1/similar', 404),
                           ('/api/users/C0001/recommendations?models=nope', 404),
                           ('/api/users/C0001/recommendations?k=0', 400), ('/api/nothing', 404),
                           ('/api/users/C0001/recommendations?exclude_seen=maybe', 400)):
            assert status(api, path) == code, path


def test_without_category_files():
    """Without the category files the API answers as before: the same items, scores and order, and
    every category null. The files are found next to products.csv by default."""
    with tempfile.TemporaryDirectory() as tmp:
        api, _ = build(tmp)
        products = os.path.join(tmp, 'products.csv')
        catalog = Catalog(tmp, 'AMAZON', products, os.path.join(tmp, 'users.csv'))
        bare = Api(catalog, Models(list(api.models.paths.values()), catalog))

        def lists(answer):
            return [[{k: v for k, v in item.items() if k != 'category'} for item in result['items']]
                    for result in answer['results']]
        for path in ('/api/users/C0003/recommendations?models=han_amazon_lp_manual_5_s1,popularity_s1&k=20',
                     '/api/users/C0007/recommendations?k=20&exclude_seen=false',
                     '/api/products/n3/similar?k=20'):
            with_categories, without = api.route(path), bare.route(path)
            assert lists(with_categories) == lists(without), path
            assert all(item['category'] is None for result in without['results'] for item in result['items'])
        assert bare.route('/api/products/n3')['product']['categories'] == []
        assert bare.route('/api/products/n3')['product']['category'] is None
        assert bare.route('/api/health')['categories'] == 0

        # default files, and what switches categories off
        assert category_files(products) == (os.path.join(tmp, 'filtered_categories.csv'),
                                            os.path.join(tmp, 'filtered_category_paths.csv'), None)
        os.remove(os.path.join(tmp, 'filtered_category_paths.csv'))
        names, paths, off = category_files(products)
        assert names is None and paths is None and 'filtered_category_paths.csv' in off
        assert category_files('') == (None, None, 'no --categories-csv or --products-csv')


def test_http():
    with tempfile.TemporaryDirectory() as tmp:
        api, _ = build(tmp)
        server = make_server(api, '127.0.0.1', 0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = f'http://127.0.0.1:{server.server_address[1]}'
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never via a proxy
        try:
            with opener.open(base + '/api/users/C0002/recommendations?k=3') as resp:
                assert resp.status == 200 and resp.headers['Access-Control-Allow-Origin'] == '*'
                assert resp.headers['Content-Type'].startswith('application/json')
                assert len(json.loads(resp.read())['results'][0]['items']) == 3
            try:
                opener.open(base + '/api/products/x1')
                assert False, 'expected 404'
            except urllib.error.HTTPError as e:
                assert e.code == 404 and 'error' in json.loads(e.read())
            req = urllib.request.Request(base + '/api/models', method='OPTIONS')
            with opener.open(req) as resp:
                assert resp.status == 204 and 'GET' in resp.headers['Access-Control-Allow-Methods']
        finally:
            server.shutdown()
            server.server_close()


if __name__ == '__main__':
    test_endpoints()
    test_without_category_files()
    test_http()
    print('ok')
