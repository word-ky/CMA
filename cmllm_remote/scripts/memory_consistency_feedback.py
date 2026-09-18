"""Fixed Cycle019 policy: miner geometry only, no targets or quality oracle."""
from helmet_geometry import helmet_geometry

THRESHOLD = 0.7
DIRECT_QUERY = 'Please segment the mining helmet in the image.'


def initial_action(pred_ref, miner_memory):
    score = helmet_geometry(pred_ref, miner_memory)['helmet_head_score']
    return {'head_score_ref': score,
            'action': 'REF_ACCEPT' if score >= THRESHOLD else 'DIRECT_FALLBACK'}


def select_candidate(pred_ref, miner_memory, pred_direct=None):
    trace = initial_action(pred_ref, miner_memory)
    if trace['action'] == 'REF_ACCEPT':
        return pred_ref, {**trace, 'selected': 'REF', 'terminal': 'STOP_ACCEPT'}
    assert pred_direct is not None
    direct_score = helmet_geometry(pred_direct, miner_memory)['helmet_head_score']
    use_direct = direct_score > trace['head_score_ref']
    return (pred_direct if use_direct else pred_ref), {
        **trace, 'head_score_direct': direct_score,
        'selected': 'DIRECT' if use_direct else 'REF', 'terminal': 'STOP_ACCEPT',
    }
