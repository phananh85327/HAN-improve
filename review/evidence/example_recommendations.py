"""The numbers behind the picks in review/example_recommendations.md: which results
recommend_6path.bat, recommend_4path.bat and recommend_api.py run, and why.

    python review/evidence/example_recommendations.py output/AMAZON/v3

Reads results.csv in that folder, review/evidence/by_group.txt (the output of h8_by_group.py),
arch.py and the two .bat files. No result file and no GPU needed.

1. the random draws of each size by validation BPR, for HAN and for metapath2vec: each model's
   median draw is its pick, and the .bat files must name exactly those medians
2. 4 against 6 paths per family of sets, for both models (why 6 paths is the main pick)
3. how popular each representative result's full-ranking top 10 is, and full-ranking Hit@10 of
   HAN and metapath2vec on the sets they share (why metapath2vec is in, popularity out)
4. which representative sets hold PP, the also-bought link itself (reading "similar products")
"""
import csv
import os
import re
import statistics as st
import sys

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, REPO)
from arch import archs  # noqa: E402

out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, 'output', 'AMAZON', 'v3')
with open(os.path.join(out, 'results.csv'), encoding='utf-8') as fh:
    rows = {r['file'][:-len('.pt')]: r for r in csv.DictReader(fh)}
MODELS = {'han': 'HAN', 'm2v': 'metapath2vec'}


def num(name, col):
    return float(rows[name][col])


def key(name):
    """han_amazon_lp_rand6_d3_s1 -> rand6_d3"""
    return name.split('amazon_lp_', 1)[1].rsplit('_s', 1)[0] if 'amazon_lp_' in name else name


def bat_files(size):
    with open(os.path.join(REPO, f'recommend_{size}path.bat'), encoding='utf-8') as fh:
        return re.search(r'^set FILES=(.*)$', fh.read(), flags=re.M).group(1).split()


print('1. The five random draws by validation BPR (seed 1), best first; the median is the pick')
for size in (6, 4):
    files = bat_files(size)
    for m, label in MODELS.items():
        draws = sorted((f'{m}_amazon_lp_rand{size}_d{d}_s1' for d in range(1, 6)),
                       key=lambda n: num(n, 'val_bpr'))
        print(f'  {label}, {size} paths (mean test Hit@10 of the five: '
              f'{st.mean(num(n, "hit10") for n in draws):.3f})')
        for i, n in enumerate(draws, 1):
            print(f'    {i}. {key(n):9s} val BPR {num(n, "val_bpr"):.4f}   Hit@10 {num(n, "hit10"):.3f}'
                  f'   full-ranking Hit@10 {num(n, "hit10_full"):.3f}' + ('   <- the pick' if i == 3 else ''))
        picked = [f for f in files if f.startswith(f'{m}_amazon_lp_rand')]
        assert picked == [draws[2]], f'recommend_{size}path.bat names {picked}, the median is {draws[2]}'

print('\n2. Mean test Hit@10 per family at seed 1, 4 -> 6 paths')
FAMILIES = {'searched': lambda s: [f'v3_f{s}_a1', f'v3_f{s}_a2', f'v3_f{s}_c'],
            'manual': lambda s: ['manual_5' if s == 4 else 'manual_6'],
            'random': lambda s: [f'rand{s}_d{d}' for d in range(1, 6)]}
for m, label in MODELS.items():
    cells = []
    for fam, keys in FAMILIES.items():
        a, b = (st.mean(num(f'{m}_amazon_lp_{k}_s1', 'hit10') for k in keys(s)) for s in (4, 6))
        cells.append(f'{fam} {a:.3f} -> {b:.3f}')
    print(f'  {label:13s}' + '   '.join(cells))

print('\n3. The full-ranking top 10 of a sample of test users (by_group.txt)')
with open(os.path.join(REPO, 'review', 'evidence', 'by_group.txt'), encoding='utf-8') as fh:
    txt = fh.read().replace('\r', '')
top = {b[0]: b[1:] for b in re.findall(
    r'^(\S+)\.pt: Hit@10 [\d.]+\n(?:  .*\n)*?  full-ranking top 10 of (\d+) test users: mean training '
    r'degree ([\d.]+) \(all products ([\d.]+)\), from the top 1% most-bought ([\d.]+)%, distinct '
    r'products (\d+)', txt, flags=re.M)}
shown = bat_files(6) + [f for f in bat_files(4) if f not in bat_files(6)] + ['popularity_s1']
users, every = top[shown[0]][0], top[shown[0]][2]
print(f'  {users} test users; a product has {every} training buyers on average')
print(f'  {"result":28s} {"Hit@10":>7s} {"full":>6s}  {"mean buyers":>11s}  {"top 1%":>6s}  {"distinct":>8s}')
for n in shown:
    _, deg, _, top1, distinct = top[n]
    print(f'  {n:28s} {num(n, "hit10"):7.3f} {num(n, "hit10_full"):6.3f}  {deg:>11s}  {top1:>5s}%  {distinct:>8s}')
shared = [key(f) for f in bat_files(6) + bat_files(4) if f.startswith('han_') and 'rand' not in f]
print('  full-ranking Hit@10 on the sets both models share, HAN against metapath2vec:')
for k in shared:
    print(f'    {k:9s} {num(f"han_amazon_lp_{k}_s1", "hit10_full"):.3f} against '
          f'{num(f"m2v_amazon_lp_{k}_s1", "hit10_full"):.3f}')

print('\n4. Representative sets that hold PP, the also-bought link itself (GAT has it as edges)')
for k in dict.fromkeys(key(f) for f in bat_files(6) + bat_files(4) if f != 'gat_union_s1'):
    paths = archs[f'amazon_lp_{k}'][0]
    print(f'  {k:9s} {"PP   " if "PP" in paths else "no PP"}  {" ".join(paths)}')
