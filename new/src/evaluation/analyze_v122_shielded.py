"""Quick paired Wilcoxon analysis of v122_proposed_shielded results."""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
old = pd.read_csv(ROOT / 'results' / 'detailed_stats_master_v12_v12_policy.csv')
new = pd.read_csv(ROOT / 'results' / 'v122_proposed_shielded' / 'online_episodes.csv')

pens = old[old.Method == 'Pensieve']
mpc = old[old.Method == 'RobustMPC']
prev_shield = old[old.Method == 'Proposed_Shielded']

KEY = ['Video', 'Episode']


def paired(name, base, metric='QoE'):
    A = new[new.Method == name].set_index(KEY)[metric]
    B = base.set_index(KEY)[metric]
    j = A.index.intersection(B.index)
    if len(j) < 10:
        return None
    d = (A.loc[j] - B.loc[j]).values
    if (d == 0).all():
        return float(np.median(d)), float(np.mean(d)), 1.0, int(len(d))
    _, p = stats.wilcoxon(d)
    return float(np.median(d)), float(np.mean(d)), float(p), int(len(d))


CFGS = [
    'shield_legacy',
    'thresh_cat3.0',
    'thresh_cat3.0_vmafFB',
    'thresh_cat4.0',
    'thresh_cat4.0_vmafFB',
    'thresh_cat5.0',
    'shield_off',
]


def show(label, base):
    print(f'\n=== vs {label} (paired, n=80) ===')
    print(f'{"Config":28s} {"med dQ":>7s} {"mean dQ":>8s} {"pQ":>8s} {"med dR%":>8s} {"pR":>8s}')
    print('-' * 78)
    for c in CFGS:
        qr = paired(c, base, 'QoE')
        rr = paired(c, base, 'Rebuffer')
        if qr and rr:
            print(
                f'{c:28s} {qr[0]:+7.2f} {qr[1]:+8.2f} {qr[2]:8.1e} '
                f'{rr[0]:+8.2f} {rr[2]:8.1e}'
            )


show('Pensieve', pens)
show('RobustMPC', mpc)
show('prior Proposed_Shielded (paper baseline)', prev_shield)
