"""Data hand-over guards (HAN review part H1): runs record which data and split they used,
and graphs or results from other data are refused instead of being reused.

    python tests/test_data_handover.py      (or: python -m pytest tests)

Writes small synthetic HGB datasets (node.dat, link.dat, label files, optional meta.json)
in the layout dat_generator.py produces for AMAZON: types P 0 / C 1 / U 2, relations 0-7.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from data_loader import data_loader  # noqa: E402
from link_data import build_link_split, check_data, data_provenance, edges_fingerprint  # noqa: E402
from train_han import load_metapath_graphs  # noqa: E402

N_P, N_C, N_U = 150, 12, 80     # P > 99 negatives + a user's positives


def make_dataset(root, seed=0, meta=True):
    """root/AMAZON with random also-bought pairs, a category chain, 1-2 categories per
    product and 4-7 reviews per user. Same node counts for every seed."""
    rng = np.random.default_rng(seed)
    d = os.path.join(root, 'AMAZON')
    os.makedirs(d, exist_ok=True)
    c0, u0 = N_P, N_P + N_C
    with open(os.path.join(d, 'node.dat'), 'w', encoding='utf-8') as f:
        for i, (t, dim) in enumerate([(0, 6)] * N_P + [(1, 3)] * N_C + [(2, 3)] * N_U):
            f.write(f'{i}\tn{i}\t{t}\t' + ','.join(f'{x:.3f}' for x in rng.random(dim)) + '\n')
    rel = {r: [] for r in range(8)}
    pairs = {tuple(sorted(rng.choice(N_P, 2, replace=False))) for _ in range(40)}
    for a, b in sorted(pairs):
        rel[0].append((a, b)); rel[1].append((b, a))
    for c in range(1, N_C):
        rel[2].append((c0 + c - 1, c0 + c)); rel[3].append((c0 + c, c0 + c - 1))
    for p in range(N_P):
        for c in rng.choice(N_C, rng.integers(1, 3), replace=False):
            rel[4].append((p, c0 + c)); rel[5].append((c0 + c, p))
    for u in range(N_U):
        for p in rng.choice(N_P, rng.integers(4, 8), replace=False):
            rel[6].append((u0 + u, p)); rel[7].append((p, u0 + u))
    with open(os.path.join(d, 'link.dat'), 'w', encoding='utf-8') as f:
        for r in range(8):
            for h, t in rel[r]:
                f.write(f'{h}\t{t}\t{r}\t1.0\n')
    for name, ids in (('label.dat', range(0, N_P, 2)), ('label.dat.test', range(1, N_P, 2))):
        with open(os.path.join(d, name), 'w', encoding='utf-8') as f:
            for i in ids:
                f.write(f'{i}\tn{i}\t0\t{i % 3}\n')
    if meta:
        with open(os.path.join(d, 'meta.json'), 'w', encoding='utf-8') as f:
            json.dump({'title_features': 'lsa', 'title_dim': 6, 'category_links': 'leaf'}, f)
    return d


def load(d, split_seed=1):
    dl = data_loader(d)
    s = build_link_split(dl, 2, 0, seed=split_seed)
    return dl, s, data_provenance(d, dl, s)


def run(script, *args):
    """Run one of the repo's scripts; returns (exit code, stdout + stderr)."""
    p = subprocess.run([sys.executable, os.path.join(REPO, script), *args], cwd=REPO,
                       capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def build(root, graph_dir, *extra):
    code, out = run('build_metapath_graphs.py', '--root', root, '--paths', 'PUP,PCP',
                    '--split-seed', '1', '--graph-dir', graph_dir, *extra)
    assert code == 0, out
    return out


def test_fingerprint_follows_data_and_split():
    with tempfile.TemporaryDirectory() as tmp:
        dl, s, prov = load(make_dataset(tmp))
        assert prov['train_edges_hash'] == edges_fingerprint(s['train_edges'][::-1])  # order-free
        assert prov['node_counts'] == {0: N_P, 1: N_C, 2: N_U}
        assert prov['meta']['category_links'] == 'leaf'
        assert load(os.path.join(tmp, 'AMAZON'), split_seed=2)[2]['train_edges_hash'] != prov['train_edges_hash']
        other = make_dataset(os.path.join(tmp, 'b'), seed=1)       # same node counts, other reviews
        assert load(other)[2]['train_edges_hash'] != prov['train_edges_hash']


def test_data_without_meta_json_stops_the_run():
    with tempfile.TemporaryDirectory() as tmp:
        prov = load(make_dataset(tmp, meta=False))[2]
        try:
            check_data(prov, 1)
            raise AssertionError('legacy data was accepted')
        except SystemExit as e:
            assert 'meta.json' in str(e)
        check_data(prov, 1, allow_legacy=True)                     # on purpose: warning only
        check_data(load(make_dataset(os.path.join(tmp, 'new')))[2], 1)


def test_graph_from_other_data_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        a = make_dataset(os.path.join(tmp, 'a'))
        b = make_dataset(os.path.join(tmp, 'b'), seed=1)
        graphs = os.path.join(tmp, 'graphs')
        build(os.path.join(tmp, 'a'), graphs)
        _, sa, pa = load(a)
        _, sb, pb = load(b)
        ok = load_metapath_graphs(graphs, ['PUP', 'PCP'], 20, 1, N_P, 'cpu', pa['train_edges_hash'])
        assert len(ok) == 2
        for args in ((pb['train_edges_hash'],), (pb['train_edges_hash'], True)):   # never, even with the flag
            try:
                load_metapath_graphs(graphs, ['PUP'], 20, 1, N_P, 'cpu', *args)
                raise AssertionError('a graph built from other data was accepted')
            except SystemExit as e:
                assert 'other training edges' in str(e)
        try:
            load_metapath_graphs(graphs, ['PUP'], 20, 1, N_P + 5, 'cpu', pa['train_edges_hash'])
            raise AssertionError('a graph with another node count was accepted')
        except SystemExit as e:
            assert 'nodes' in str(e)
        # a graph from before the fingerprint existed: refused unless allowed
        side = os.path.join(graphs, 'PUP_top20_split1.meta.json')
        meta = json.load(open(side))
        del meta['train_edges_hash']
        json.dump(meta, open(side, 'w'))
        try:
            load_metapath_graphs(graphs, ['PUP'], 20, 1, N_P, 'cpu', pa['train_edges_hash'])
            raise AssertionError('a graph without a fingerprint was accepted')
        except SystemExit as e:
            assert 'does not record' in str(e)
        assert len(load_metapath_graphs(graphs, ['PUP'], 20, 1, N_P, 'cpu', pa['train_edges_hash'], True)) == 1


def test_skip_existing_rebuilds_a_stale_graph():
    with tempfile.TemporaryDirectory() as tmp:
        make_dataset(os.path.join(tmp, 'a'))
        make_dataset(os.path.join(tmp, 'b'), seed=1)
        graphs = os.path.join(tmp, 'graphs')
        build(os.path.join(tmp, 'a'), graphs)
        out = build(os.path.join(tmp, 'b'), graphs, '--skip-existing')
        assert out.count('Rebuilding') == 2 and 'Skipping' not in out, out
        pb = load(os.path.join(tmp, 'b', 'AMAZON'))[2]
        assert json.load(open(os.path.join(graphs, 'PCP_top20_split1.meta.json')))['train_edges_hash'] == pb['train_edges_hash']
        out = build(os.path.join(tmp, 'b'), graphs, '--skip-existing')
        assert out.count('Skipping') == 2, out


def test_builder_stops_on_legacy_data():
    with tempfile.TemporaryDirectory() as tmp:
        make_dataset(tmp, meta=False)
        code, out = run('build_metapath_graphs.py', '--root', tmp, '--paths', 'PUP',
                        '--graph-dir', os.path.join(tmp, 'g'))
        assert code != 0 and 'meta.json' in out, out


def _blob(path, data_hash, n_p=N_P, n_u=N_U, seed=1, hit=0.5, **config):
    torch.save({'model': 'HAN', 'dataset': 'AMAZON', 'target_relation': 'U,P', 'paths': ['PUP'],
                'split_seed': 1, 'P_emb': torch.randn(n_p, 4), 'U_emb': torch.randn(n_u, 4),
                'metrics': {'Hit@10': hit}, 'best_val_loss': 0.3,
                'data': None if data_hash is None else {'train_edges_hash': data_hash},
                'config': {'arch': 'amazon_lp_x', 'seed': seed, 'beta_warmup': 15, **config}}, path)


def test_summary_keeps_data_versions_apart():
    with tempfile.TemporaryDirectory() as tmp:
        _blob(os.path.join(tmp, 'old.pt'), None, hit=0.85)
        _blob(os.path.join(tmp, 'new.pt'), 'abcd', hit=0.40)
        code, out = run('summarize_results.py', os.path.join(tmp, '*.pt'), '--group')
        assert code == 0, out
        assert 'WARNING' in out and '2 different data versions' in out, out
        grouped = [l for l in out.split('Grouped over run seeds')[1].splitlines() if l.startswith('| HAN')]
        assert len(grouped) == 2 and not any('+/-' in l for l in grouped), grouped   # not averaged
        os.remove(os.path.join(tmp, 'old.pt'))
        code, out = run('summarize_results.py', os.path.join(tmp, '*.pt'))
        assert 'WARNING' not in out, out


def test_summary_keeps_settings_apart():
    with tempfile.TemporaryDirectory() as tmp:
        for seed in (1, 2):
            _blob(os.path.join(tmp, f'w15_s{seed}.pt'), 'abcd', seed=seed, hit=0.80 + seed / 100)
        _blob(os.path.join(tmp, 'w0_s1.pt'), 'abcd', seed=1, hit=0.70, beta_warmup=0)
        _blob(os.path.join(tmp, 'w15_s3_gpu1.pt'), 'abcd', seed=3, hit=0.83, gpu=1)   # not a setting
        code, out = run('summarize_results.py', os.path.join(tmp, '*.pt'), '--group')
        assert code == 0, out
        rows = [l for l in out.split('Grouped over run seeds')[1].splitlines() if l.startswith('| HAN')]
        assert len(rows) == 2, rows
        assert any('beta_warmup=0' in l and '| 1 | 1 |' in l for l in rows), rows
        assert any('beta_warmup=15' in l and '| 3 | 1,2,3 |' in l for l in rows), rows


def test_eval_emb_skips_results_from_other_data():
    with tempfile.TemporaryDirectory() as tmp:
        d = make_dataset(tmp)
        h = load(d)[2]['train_edges_hash']
        _blob(os.path.join(tmp, 'a_other.pt'), 'abcd')
        _blob(os.path.join(tmp, 'b_bigger.pt'), None, n_p=N_P + 7)
        _blob(os.path.join(tmp, 'c_same.pt'), h)
        code, out = run('eval_emb.py', '--root', tmp, '--cpu', '--emb', os.path.join(tmp, '*.pt'))
        assert code == 0, out
        parts = out.split('=== ')
        assert 'SKIPPED: trained on other data or another split' in parts[1], parts[1]
        assert 'SKIPPED: the blob has' in parts[2], parts[2]
        assert 'SKIPPED' not in parts[3] and 'Hit@10' in parts[3], parts[3]


def test_check_handover():
    with tempfile.TemporaryDirectory() as tmp:
        d = make_dataset(tmp)
        prov = load(d)[2]
        rec = os.path.join(tmp, 'search.json')

        def check(record):
            json.dump(record, open(rec, 'w'))
            return run('check_handover.py', rec, '--root', tmp)
        base = {'dataset': 'AMAZON', 'target_relation': 'U,P', 'split_seed': 1,
                'data': {k: prov[k] for k in ('node_counts', 'num_target_edges', 'train_edges_hash')}}
        code, out = check(base)
        assert code == 0 and 'SAME' in out, out
        code, out = check({**base, 'split_seed': 2})               # searched on another split
        assert code == 1 and 'fingerprint' in out, out
        code, out = check({k: v for k, v in base.items() if k != 'data'})
        assert code == 2, out
        shutil.rmtree(os.path.join(tmp, 'AMAZON'))
        make_dataset(tmp, seed=3)                                  # same counts, other data
        code, out = check(base)
        assert code == 1 and 'fingerprint' in out, out


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
