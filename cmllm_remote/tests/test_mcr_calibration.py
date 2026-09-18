import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from calibrate_mcr_threshold import accepted, choose_threshold


def rows(condition, score, success, count):
    return [dict(condition=condition, group_contrast=score, cmsa_success=success, action='ACCEPT')
            for _ in range(count)]


def test_smallest_feasible_shared_threshold_and_ties():
    data = (rows('clean', .8, True, 10) + rows('target15_b', .8, True, 5)
            + rows('target15_b', .9, True, 3) + rows('target15_b', .1, False, 2))
    result = choose_threshold(data)
    assert result['tau'] == .8
    assert result['feasible_count'] == 1
    assert result == choose_threshold(data[::-1])
    assert accepted(data[0], .8)
    assert not accepted({**data[0], 'action': 'ABSTAIN_ESCALATE'}, .8)


def test_infeasible_clean_coverage_cannot_be_relaxed():
    data = rows('clean', .1, True, 10) + rows('target15_b', .8, True, 8) + rows('target15_b', .1, False, 2)
    result = choose_threshold(data)
    assert result['tau'] is None
    assert result['candidate_count'] == 2


def test_nonpositive_scores_never_become_candidates():
    data = rows('clean', 0, True, 10) + rows('target15_b', -.1, False, 10)
    assert choose_threshold(data)['candidate_table'] == []
