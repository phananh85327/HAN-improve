#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Print the arch.py keys that run_all.bat should train, one line, space-separated.

Each argument is a key stem; a key matches when it is the stem followed by digits only:
    python arch_keys.py amazon_lp_v3_f4_a amazon_lp_v3_f6_a
prints e.g. "amazon_lp_v3_f4_a1 amazon_lp_v3_f4_a2 amazon_lp_v3_f6_a1", in arch.py order.
So `amazon_lp_v3_f4_a` picks the averaged one-SE sets that reselect_fleet.py prints, and
not `amazon_lp_v3_f4_d005_a1` or `amazon_lp_v3_f4_c`.

Exit code 1 (and a message on stderr) when a stem matches nothing, e.g. before the Fleet v3
entries have been pasted into arch.py, so the batch file can stop instead of training nothing.
"""

import re
import sys

from arch import archs


def keys_for(stems):
    """{stem: [matching keys in arch.py order]}"""
    return {stem: [k for k in archs if re.fullmatch(re.escape(stem) + r'\d+', k)] for stem in stems}


def main(stems):
    found = keys_for(stems)
    missing = [s for s, ks in found.items() if not ks]
    if missing:
        sys.stderr.write('no arch.py key matches ' + ', '.join(f'{s}<number>' for s in missing)
                         + ' -- paste the entries into arch.py first\n')
        return 1
    print(' '.join(k for s in stems for k in found[s]))
    return 0


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1:]))
