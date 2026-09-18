"""Reproduce the diagnostic-only risk/coverage figure from scored frozen actions."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = Path(__file__).resolve().parent / 'scoring'
result = json.loads((out / 'selective_reliability.json').read_text())
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
for ax, condition in zip(axes, ['clean', 'target15_b']):
    stats = result[condition]
    curve = stats['diagnostics_ANALYSIS_ONLY']['risk_coverage_curve'][1:]
    ax.plot([r['coverage'] for r in curve], [r['risk_cmsa_failure'] for r in curve],
            '.-', label='Diagnostic g thresholds')
    ax.scatter([stats['coverage']], [1 - stats['accepted_summary']['cmsa']],
               s=65, color='darkorange', zorder=5, label='Fixed runtime rule')
    ax.axhline(1 - stats['base_summary']['cmsa'], color='gray', linestyle='--', label='All-group risk')
    ax.set(title=condition, xlabel='Group coverage', xlim=(0, 1.02), ylim=(0, 1))
    ax.grid(alpha=.25)
axes[0].set_ylabel('Risk: CMSA failure rate')
axes[1].legend(fontsize=8)
fig.suptitle('Analysis only: tied scores grouped; no threshold selected')
fig.tight_layout()
fig.savefig(out / 'risk_coverage.png', dpi=160)
fig.savefig(out / 'risk_coverage.svg')
