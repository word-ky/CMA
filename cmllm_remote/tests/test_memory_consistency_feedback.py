import sys
from pathlib import Path
import numpy as np
import subprocess

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from memory_consistency_feedback import initial_action, select_candidate


def mask(x1, y1, x2, y2):
    result = np.zeros((100, 100), dtype=np.uint8)
    result[y1:y2, x1:x2] = 1
    return result


def test_strict_and_loose_head_stop_without_direct():
    miner = mask(10, 10, 40, 90)
    for helmet in [mask(15, 12, 25, 22), mask(15, 49, 25, 57)]:
        output, trace = select_candidate(helmet, miner)
        assert output is helmet and trace['action'] == 'REF_ACCEPT'


def test_body_prediction_triggers_and_better_head_replaces():
    miner, ref, direct = mask(10, 10, 40, 90), mask(15, 65, 25, 75), mask(15, 12, 25, 22)
    output, trace = select_candidate(ref, miner, direct)
    assert initial_action(ref, miner)['head_score_ref'] == 0.3
    assert output is direct and trace['selected'] == 'DIRECT'


def test_tie_and_worse_keep_ref():
    miner, ref = mask(10, 10, 40, 90), mask(15, 65, 25, 75)
    for direct in [ref.copy(), mask(70, 12, 80, 22)]:
        output, trace = select_candidate(ref, miner, direct)
        assert output is ref and trace['selected'] == 'REF'


def test_same_prediction_different_memory_changes_action():
    helmet = mask(15, 12, 25, 22)
    assert initial_action(helmet, mask(10, 10, 40, 90))['action'] == 'REF_ACCEPT'
    assert initial_action(helmet, mask(60, 10, 90, 90))['action'] == 'DIRECT_FALLBACK'


def test_empty_prediction_can_recover_without_target_input():
    ref, miner, direct = mask(0, 0, 0, 0), mask(10, 10, 40, 90), mask(15, 12, 25, 22)
    output, trace = select_candidate(ref, miner, direct)
    assert output is direct and trace['head_score_ref'] == 0


def test_read_audit_rejects_target_and_metric_files(tmp_path):
    script = """
from run_memory_consistency_feedback import install_audit
from pathlib import Path
import sys
root = Path(sys.argv[1])
target = root / 'helmet_target.npy'
metric = root / 'memory_metrics.json'
target.write_bytes(b'not read'); metric.write_text('{}')
install_audit([])
for path in [target, metric]:
    try:
        path.read_bytes()
    except AssertionError:
        continue
    raise AssertionError('oracle file was allowed')
"""
    import os
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / 'scripts'))
    subprocess.run([sys.executable, '-c', script, str(tmp_path)], env=env, check=True)
