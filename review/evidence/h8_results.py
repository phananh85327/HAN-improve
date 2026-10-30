"""HAN review H8: the numbers in review/h8_results.md, from the run logs and results.csv.

    python review/evidence/h8_results.py output/AMAZON/v3

1. every run: exit code, data, stop reason (early stop or cap), validation gain left at the stop
2. results.csv against the logs
3. the main comparison (3 seeds): Welch t-test per metric
4. the 24 sets at seed 1: order within each size, validation against test
5. HAN's path weights (beta): paths it drops, weight per path relative to an equal share
6. Hit@10 by the user's number of training purchases, headline sets
"""
import csv
import glob
import os
import re
import statistics as st
import sys

from scipy import stats

out = sys.argv[1]
logdir = os.path.join(out, 'logs')
EMPTY = {'PCCP', 'PCCPP', 'PCCPPP', 'PCCPCP', 'PCCPUP', 'PPCCPP'}   # no endpoint for 76-81% of products


def parse(path):
    txt = open(path, encoding='utf-8', errors='replace').read()
    lines = re.split(r'[\r\n]+', txt)   # tqdm's \r progress can hide log lines
    model = next((m for m, s in (('HAN', 'train_han'), ('GAT', 'train_gat'), ('metapath2vec', 'metapath2vec'))
                  if s in lines[0]), None)
    if model is None:
        return None
    ns = next((l for l in lines if l.startswith('Namespace(')), '')
    get = lambda k: (re.search(rf"\b{k}=([^,)]+)", ns) or [None, None])[1]
    r = {'file': os.path.basename(path)[:-4], 'model': model, 'arch': (get('arch') or "'-'").strip("'"),
         'seed': get('seed'), 'cap': int(get('stage') or get('epochs')), 'warmup': get('beta_warmup')}
    epochs = [(float(a), float(b)) for a, b in re.findall(r'^Epoch ([\d.]+), \S+ \S+ [\d.]+, val bpr ([\d.]+)', txt, re.M)]
    r['last'] = epochs[-1][0]
    r['early'] = bool(re.search(r'Early stop at epoch', txt))
    m = re.search(r'Best epoch ([\d.]+), val bpr ([\d.]+)', txt)
    r['best_epoch'], r['val'] = float(m.group(1)), float(m.group(2))
    m = re.search(r'Test ranking metrics:(.*?)By user', txt, re.S)
    r['hit10'] = float(re.search(r'Hit@10: ([\d.]+)', m.group(1)).group(1))
    r['exit'] = int((re.search(r'exit code (\d+)', txt) or [None, -1])[1])
    r['fp'] = (re.search(r'training-edge fingerprint (\w+)', txt) or [None, '?'])[1]
    # validation gain over the last 10 epochs (HAN, GAT): best-so-far, warm-up epochs excluded
    if model != 'metapath2vec':
        skip = int(r['warmup'] or 0)
        best, run = [], float('inf')
        for _, v in epochs[skip:]:
            run = min(run, v)
            best.append(run)
        r['gain10'] = best[-11] - best[-1] if len(best) > 10 else None
    m = re.search(r'Semantic attention over meta-paths:\n((?:\s+\S+: [\d.]+\n)+)', txt)
    r['beta'] = {a: float(b) for a, b in re.findall(r'(\S+): ([\d.]+)', m.group(1))} if m else {}
    blk = txt.split('By user training degree:')[1].split('By duplicate')[0]
    r['deg'] = {k.strip(): float(h) for k, h in re.findall(r'(deg[^(]*)\(\s*\d+ edges\)\s+NDCG@10 [\d.]+\s+Hit@10 ([\d.]+)', blk)}
    return r


runs = [r for p in sorted(glob.glob(os.path.join(logdir, '*_s*.txt'))) for r in [parse(p)] if r]
print(f'1. {len(runs)} training logs in {logdir}')
bad = [r['file'] for r in runs if r['exit'] != 0 or r['fp'] != 'c5f7bccbf61c1b0f']
capped = [r['file'] for r in runs if not r['early']]
print(f'   exit code or data wrong: {bad or "none"}; ended by their cap, not early stopping: {capped or "none"}')
for model in ('HAN', 'GAT', 'metapath2vec'):
    rs = [r for r in runs if r['model'] == model]
    print(f'   {model}: {len(rs)} runs, stopped at epoch {min(r["last"] for r in rs):g}-{max(r["last"] for r in rs):g} '
          f'(cap {sorted({r["cap"] for r in rs})})', end='')
    if model != 'metapath2vec':
        g = sorted((r['gain10'], r['file']) for r in rs)
        print(f'; val BPR gained in the last 10 epochs: median {st.median(x for x, _ in g):.4f}, '
              f'max {g[-1][0]:.4f} ({g[-1][1]})', end='')
    print()
lm = [r['gain10'] for r in runs if r['model'] == 'HAN' and '_v3_' in r['arch']]
print(f'   LMSPS runs: at most {max(lm):.4f} left; runs with more than 0.001 left: '
      + ', '.join(r['file'] for r in runs if (r.get('gain10') or 0) > 0.001))

rows = list(csv.DictReader(open(os.path.join(out, 'results.csv'), encoding='utf-8')))
bylog = {r['file']: r for r in runs}
miss = [x['file'] for x in rows if x['model'] != 'popularity' and (x['file'][:-3] not in bylog or
        abs(float(x['val_bpr']) - bylog[x['file'][:-3]]['val']) > 6e-5 or
        abs(float(x['hit10']) - bylog[x['file'][:-3]]['hit10']) > 6e-5)]
print(f'2. results.csv: {len(rows)} rows; rows that do not match their log: {miss or "none"}')

print('3. main comparison, 3 seeds each (Welch t-test; the split is fixed, so this is training noise only)')
val = lambda m, a, c: [float(x[c]) for x in rows if x['model'] == m and x['arch'] == a]
for model in ('HAN', 'metapath2vec'):
    for a, b in (('amazon_lp_v3_f4_a1', 'amazon_lp_manual_5'), ('amazon_lp_v3_f6_a1', 'amazon_lp_manual_6')):
        for c in ('hit10', 'ndcg10', 'hit10_full', 'recall20_full'):
            x, y = val(model, a, c), val(model, b, c)
            t = stats.ttest_ind(x, y, equal_var=False)
            print(f'   {model:12s} {a[10:]} {st.mean(x):.4f}+/-{st.stdev(x):.4f} vs {b[10:]} {st.mean(y):.4f}+/-{st.stdev(y):.4f}'
                  f'  {c:13s} diff {st.mean(x) - st.mean(y):+.4f}  p={t.pvalue:.4f}')
g = val('GAT', '-', 'hit10')
print(f'   GAT Hit@10 {st.mean(g):.4f}+/-{st.stdev(g):.4f} over {len(g)} runs')

print('4. the 24 sets, seed 1')
s1 = {(x['model'], x['arch']): x for x in rows if x['seed'] == '1'}
han = {a: x for (m, a), x in s1.items() if m == 'HAN'}
kind = lambda a: 'LMSPS' if '_v3_' in a else 'random' if 'rand' in a else 'manual'
for n in (4, 6):
    cand = [a for a in han if kind(a) != 'manual' and f'rand{n}' in a or f'_f{n}_' in a]
    for c, rev in (('hit10', True), ('val_bpr', False), ('hit10_full', True)):
        order = sorted(cand, key=lambda a: float(han[a][c]), reverse=rev)
        print(f'   {n} paths, HAN {c:10s}: ' + ' > '.join(a[10:] for a in order))
    m2v = {a: s1[('metapath2vec', a)] for a in cand}
    print(f'   {n} paths, m2v {"hit10":10s}: ' + ' > '.join(a[10:] for a in sorted(cand, key=lambda a: -float(m2v[a]['hit10']))))
for k in ('rand4', 'rand6'):
    v = [float(han[a]['hit10']) for a in han if k in a]
    print(f'   HAN {k} draws: mean {st.mean(v):.4f}, range {min(v):.4f}-{max(v):.4f}')
a = list(han)
rho = stats.spearmanr([float(han[x]['val_bpr']) for x in a], [float(han[x]['hit10']) for x in a])
rho_f = stats.spearmanr([float(han[x]['val_bpr']) for x in a], [float(han[x]['hit10_full']) for x in a])
print(f'   validation BPR against test, over the 24 sets: Spearman {rho.correlation:+.2f} (Hit@10), '
      f'{rho_f.correlation:+.2f} (full-ranking Hit@10); lowest validation BPR: '
      f'{min(a, key=lambda x: float(han[x]["val_bpr"]))[10:]}')
flip = [x for x in a if x != 'amazon_lp_manual_4' and float(s1[('metapath2vec', x)]['hit10_full']) > float(han[x]['hit10_full'])]
print(f'   HAN beats metapath2vec on sampled Hit@10 in '
      f'{sum(float(han[x]["hit10"]) > float(s1[("metapath2vec", x)]["hit10"]) for x in a)}/24 sets; '
      f'metapath2vec beats HAN on full-ranking Hit@10 in {len(flip)}/24')

print('5. HAN path weights (beta), seed 1: paths dropped (beta < 0.01), weight x number of paths per path')
h1 = [r for r in runs if r['model'] == 'HAN' and r['seed'] == '1']
use = {}
for r in h1:
    dropped = [p for p, b in r['beta'].items() if b < 0.01]
    print(f'   {r["arch"][10:]:10s} dropped: {" ".join(dropped) or "-"}')
    for p, b in r['beta'].items():
        use.setdefault(p, []).append(b * len(r['beta']))
for label, test in (('near-empty category paths', lambda p: p in EMPTY),
                    ('other category paths, no U', lambda p: 'C' in p and 'U' not in p and p not in EMPTY),
                    ('paths through U', lambda p: 'U' in p and p not in EMPTY),
                    ('P-P paths', lambda p: set(p) == {'P'})):
    v = [x for p, xs in use.items() if test(p) for x in xs]
    d = sum(1 for r in h1 for p, b in r['beta'].items() if test(p) and b < 0.01)
    print(f'   {label:27s}: {len(v):2d} uses, dropped in {d}, median relative weight {st.median(v):.2f}')

print('6. Hit@10 by training purchases of the user, mean over seeds')
for model in ('HAN', 'metapath2vec'):
    for k in ('amazon_lp_v3_f4_a1', 'amazon_lp_manual_5', 'amazon_lp_v3_f6_a1', 'amazon_lp_manual_6'):
        rs = [r for r in runs if r['model'] == model and r['arch'] == k]
        keys = [d for d in rs[0]['deg'] if d != 'deg=0']
        print(f'   {model:12s} {k[10:]:9s} ' + '  '.join(f'{d}: {st.mean(r["deg"][d] for r in rs):.3f}' for d in keys))
