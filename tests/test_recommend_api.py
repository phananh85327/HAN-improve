"""recommend_api.py on a synthetic dataset: every endpoint answers as documented, the lists match
the scores recommend.py would compute, and the HTTP layer sends JSON with the CORS headers.

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
from recommend_api import Api, ApiError, Catalog, Models, make_server  # noqa: E402
from test_data_handover import N_P, N_U, make_dataset  # noqa: E402

GROUPS = ['Book', 'DVD', 'Music', 'Video']


def build(tmp):
    """Synthetic data, its two CSVs, one HAN-like result and the popularity result."""
    make_dataset(tmp)
    with open(os.path.join(tmp, 'products.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['ID', 'ASIN', 'Discontinued', 'Title', 'Group', 'SalesRank', 'AvgRating'])
        for i in range(N_P):
            w.writerow([i, f'n{i}', 0, f'Product number {i}', GROUPS[i % 4], 0, 0])
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
    catalog = Catalog(tmp, 'AMAZON', os.path.join(tmp, 'products.csv'), os.path.join(tmp, 'users.csv'))
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
        assert len(api.route('/api/users/C0001')['products']) == len(api.catalog.seen[1])
        sample = api.route('/api/users?min_reviewed=5&max_reviewed=6&limit=4')
        assert len(sample['users']) <= 4 and all(5 <= u['reviewed'] <= 6 for u in sample['users'])

        for path, code in (('/api/users/nobody/recommendations', 404), ('/api/products/x1/similar', 404),
                           ('/api/users/C0001/recommendations?models=nope', 404),
                           ('/api/users/C0001/recommendations?k=0', 400), ('/api/nothing', 404),
                           ('/api/users/C0001/recommendations?exclude_seen=maybe', 400)):
            assert status(api, path) == code, path


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
    test_http()
    print('ok')
