"""Static checks of run_all.bat (HAN review part H6). The batch file cannot run here, so this
checks what most often breaks one: labels, scripts, flags and arch.py keys.

    python tests/test_run_all.py      (or: python -m pytest tests)
"""
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from arch import archs  # noqa: E402

BAT = open(os.path.join(REPO, 'run_all.bat'), encoding='utf-8').read()
LINES = [l.strip() for l in BAT.splitlines() if l.strip() and not l.strip().upper().startswith('REM')]


def variables():
    """`set NAME=value` lines, with %NAME% references expanded (the batch file's own order)."""
    out = {'PY': 'python', 'TEMP': 'TEMP', '~nx0': 'run_all.bat'}
    for line in LINES:
        m = re.match(r'(?:if not defined \w+ )?set ([A-Z_]+)=(.*)$', line)
        if m and not line.startswith('set /p'):
            out.setdefault(m.group(1), re.sub(r'%(\w+)%', lambda v: out.get(v.group(1), v.group(0)), m.group(2)))
    return out


def expand(text, env):
    return re.sub(r'%(\w+)%', lambda v: env.get(v.group(1), v.group(0)), text)


def test_every_label_used_is_defined():
    defined = set(re.findall(r'^:(\w+)', BAT, flags=re.M))
    used = set(re.findall(r'call :(\w+)', BAT)) | set(re.findall(r'goto (\w+)', BAT))
    used -= {'eof'}
    stages = re.search(r'for %%S in \(([^)]*)\) do \(\s*if /i "%STAGE%"=="%%S"', BAT).group(1).split()
    assert not (used | set(stages)) - defined, (used | set(stages)) - defined


def test_scripts_and_flags_exist():
    env = variables()
    seen = 0
    for line in LINES:
        for script, args in re.findall(r'%PY% (\w+\.py)([^&|]*)', line):
            path = os.path.join(REPO, script)
            assert os.path.isfile(path), script
            src = open(path, encoding='utf-8').read()
            for flag in re.findall(r'(?<!\S)(--[a-z][a-z0-9-]*)', expand(args, env)):
                assert f"'{flag}'" in src, f'{script} has no {flag}'
                seen += 1
    assert seen > 30, seen


def test_sets_exist_in_arch():
    env = variables()
    for name in ('LMSPS', 'MANUAL', 'RANDOM_SETS', 'HEADLINE', 'SMOKE_KEY'):
        for key in env[name].split():
            assert key in archs, key
    # every Fleet v3 set pasted into arch.py is trained, and nothing else under that name
    assert sorted(env['LMSPS'].split()) == sorted(k for k in archs if k.startswith('amazon_lp_v3_'))


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
