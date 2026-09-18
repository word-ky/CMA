import sys
from pathlib import Path
import numpy as np
import hashlib
import json
import subprocess
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from memory_contrast_reliability import decide, head_support


def box(x1, y1, x2, y2):
    m = np.zeros((20, 20), np.uint8)
    m[y1:y2, x1:x2] = 1
    return m


def test_own_memory_accepts_and_swapped_memory_abstains():
    memories = [box(0, 0, 8, 20), box(12, 0, 20, 20)]
    predictions = [box(1, 1, 3, 3), box(13, 1, 15, 3)]
    assert decide(predictions, memories)['support_matrix'] == [[1, 0], [0, 1]]
    assert decide(predictions, memories)['action'] == 'ACCEPT'
    assert decide(predictions, memories[::-1])['action'] == 'ABSTAIN_ESCALATE'


def test_empty_and_ties_abstain():
    m = box(0, 0, 8, 20)
    p = box(1, 1, 3, 3)
    assert decide([p, p], [m, m])['action'] == 'ABSTAIN_ESCALATE'
    assert decide([p, np.zeros_like(p)], [m, box(12, 0, 20, 20)])['action'] == 'ABSTAIN_ESCALATE'


def test_support_fraction_and_exact_upper_sixty_percent():
    h, bbox = head_support(box(0, 0, 8, 20))
    assert bbox == [0, 0, 8, 20] and h.sum() == 96
    p = box(0, 10, 8, 14)
    result = decide([p, box(13, 1, 15, 3)], [box(0, 0, 8, 20), box(12, 0, 20, 20)])
    assert result['support_matrix'][0] == [.5, 0]


def test_mutating_helmet_target_does_not_change_scores_or_actions(tmp_path):
    target = tmp_path / 'helmet_target.npy'
    memories = [box(0, 0, 8, 20), box(12, 0, 20, 20)]
    predictions = [box(1, 1, 3, 3), box(13, 1, 15, 3)]
    np.save(target, np.zeros((20, 20)))
    before = decide(predictions, memories)
    np.save(target, np.ones((20, 20)))
    assert decide(predictions, memories) == before


def test_diagnostic_auc_handles_ties_and_order():
    from score_memory_contrast_reliability import diagnostic_auc
    assert diagnostic_auc([1, 0], [1, 1]) == (.5, .5, .75)
    assert diagnostic_auc([1, 0], [1, 0]) == (1., 1., 1.)
    assert diagnostic_auc([1, 0], [0, 1]) == (0., .5, .25)


def test_offline_actions_ignore_changed_target_file(tmp_path):
    trials = []
    for i, x in enumerate([0, 12]):
        pred, miner = tmp_path / f'p{i}.npy', tmp_path / f'm{i}.png'
        np.save(pred, box(x + 1, 1, x + 3, 3))
        Image.fromarray(box(x, 0, x + 8, 20)).save(miner)
        trials.append({'entity_id': str(i), 'prediction': str(pred), 'miner_mask': str(miner),
                       'prediction_sha256': hashlib.sha256(pred.read_bytes()).hexdigest(),
                       'miner_sha256': hashlib.sha256(miner.read_bytes()).hexdigest()})
    spec = tmp_path / 'inputs.json'
    spec.write_text(json.dumps({'rule': 'strict fixed contrast', 'groups': [
        {'group_id': 'synthetic', 'condition': 'clean', 'trials': trials}]}))
    target = tmp_path / 'helmet_target.npy'
    outputs = []
    script = Path(__file__).resolve().parents[1] / 'scripts/run_memory_contrast_reliability.py'
    for value in [0, 1]:
        np.save(target, np.full((20, 20), value))
        out = tmp_path / f'actions_{value}.json'
        subprocess.run([sys.executable, str(script), '--inputs', str(spec), '--output', str(out)], check=True)
        outputs.append(json.loads(out.read_text()))
    assert outputs[0]['groups'] == outputs[1]['groups']
    assert str(target) not in outputs[0]['raster_reads'] + outputs[1]['raster_reads']
