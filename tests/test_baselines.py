"""Baselines (HAN review part H4): popularity, the GAT's target-edge removal, metapath2vec walks.

    python tests/test_baselines.py      (or: python -m pytest tests)
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

TESTS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TESTS)
sys.path.insert(0, REPO)
sys.path.insert(0, TESTS)
from build_metapath_graphs import build_base_matrices  # noqa: E402
from data_loader import data_loader  # noqa: E402
from link_data import build_link_split  # noqa: E402
from test_data_handover import make_dataset  # noqa: E402
from train_gat import make_target_dropper  # noqa: E402
from train_metapath2vec import generate_walks  # noqa: E402


def run(script, *args):
    p = subprocess.run([sys.executable, os.path.join(REPO, script), *args], cwd=REPO,
                       capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def test_popularity_scores_are_training_counts():
    with tempfile.TemporaryDirectory() as tmp:
        d = make_dataset(tmp)
        out = os.path.join(tmp, 'pop.pt')
        code, log = run('popularity_baseline.py', '--root', tmp, '--save-emb', out)
        assert code == 0, log
        blob = torch.load(out)
        s = build_link_split(data_loader(d), 2, 0, seed=1)
        counts = np.asarray(s['train_adj_src2dst'].sum(0)).ravel()
        score = (blob['U_emb'][0] @ blob['P_emb'].T).numpy()
        assert np.array_equal(np.floor(score), counts)               # the tie-break stays below 1
        assert np.all(np.diff(score[np.argsort(counts, kind='stable')]) >= -0.5)
        code, log = run('eval_emb.py', '--root', tmp, '--cpu', '--emb', out)
        assert code == 0 and '[matches stored]' in log and 'STORED' not in log, log


def test_gat_drops_exactly_the_batch_edges():
    src = torch.tensor([0, 1, 2, 3, 3, 4, 0, 4])
    dst = torch.tensor([3, 4, 3, 0, 2, 1, 0, 4])
    drop = make_target_dropper((src, dst), 5)
    s, d = drop(torch.tensor([0, 2]), torch.tensor([3, 3]))          # (0,3) and (2,3), both directions
    assert sorted(zip(s.tolist(), d.tolist())) == [(0, 0), (1, 4), (4, 1), (4, 4)]
    s, d = drop(torch.tensor([1]), torch.tensor([2]))                # not an edge: nothing removed
    assert len(s) == 8


def test_gat_trains_with_and_without_target_edges():
    with tempfile.TemporaryDirectory() as tmp:
        make_dataset(tmp)
        for extra in ([], ['--keep-target-edges']):
            code, log = run('train_gat.py', '--root', tmp, '--cpu', '--hidden', '16', '--in-dim', '8',
                            '--num-heads', '2', '--stage', '2', '--batch-size', '256', '--save-emb', '',
                            *extra)
            assert code == 0 and 'Hit@10' in log, log


def test_metapath2vec_walks_follow_templates_on_training_edges():
    with tempfile.TemporaryDirectory() as tmp:
        dl = data_loader(make_dataset(tmp))
        s = build_link_split(dl, 2, 0, seed=1)
        base = build_base_matrices(dl, 'AMAZON', s)
        shifts = {'P': dl.nodes['shift'][0], 'C': dl.nodes['shift'][1], 'U': dl.nodes['shift'][2]}
        letter = np.empty(dl.nodes['total'], dtype='<U1')
        for t, i in (('P', 0), ('C', 1), ('U', 2)):
            letter[shifts[t]:shifts[t] + dl.nodes['count'][i]] = t
        held_out = set(map(tuple, np.concatenate([s['val_edges'], s['test_edges']]).tolist()))
        for tpl in ('PUP', 'PCCP', 'PUPCP'):
            steps = [tpl[i:i + 2] for i in range(len(tpl) - 1)]
            want = tpl[0] + ''.join(steps[i % len(steps)][1] for i in range(9))
            walks = generate_walks(base, [tpl], dl.nodes['count'][0], shifts, 2, 9, np.random.default_rng(0))
            for w in walks:
                w = w[w >= 0]
                assert ''.join(letter[w]) == want[:len(w)]
                for a, b in zip(w[:-1], w[1:]):
                    la, lb = a - shifts[letter[a]], b - shifts[letter[b]]
                    assert base[letter[a] + letter[b]][la, lb] > 0
                    if {letter[a], letter[b]} == {'U', 'P'}:
                        assert ((la, lb) if letter[a] == 'U' else (lb, la)) not in held_out


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
