"""Fixed, model-free counterfactual memory support gate (Cycle020)."""
import numpy as np
from helmet_geometry import bbox_from_mask


def head_support(miner_memory):
    bbox = bbox_from_mask(miner_memory)
    h, w = miner_memory.shape
    if bbox is None:
        return np.zeros((h, w), dtype=bool), None
    x1, y1, x2, y2 = bbox
    # Pixel centers rasterize the exact upper60% rectangle; no dilation/rounding sweep.
    y, x = np.ogrid[:h, :w]
    support = ((x + .5 >= max(0, x1)) & (x + .5 < min(w, x2))
               & (y + .5 >= max(0, y1)) & (y + .5 < min(h, y1 + .6 * (y2 - y1))))
    return support, bbox


def decide(predictions, miner_memories):
    assert len(predictions) == len(miner_memories) == 2
    heads = [head_support(m) for m in miner_memories]
    areas = [int(np.count_nonzero(p)) for p in predictions]
    scores = [[float(np.count_nonzero((p > 0) & head[0])) / max(area, 1)
               for head in heads] for p, area in zip(predictions, areas)]
    margins = [scores[0][0] - scores[0][1], scores[1][1] - scores[1][0]]
    accept = all(areas) and scores[0][0] > 0 and scores[1][1] > 0 and all(m > 0 for m in margins)
    return {'support_matrix': scores, 'margins': margins, 'group_contrast': min(margins),
            'assignment_gap': sum(margins), 'prediction_areas': areas,
            'miner_bboxes_xyxy': [head[1] for head in heads],
            'head_support_areas': [int(head[0].sum()) for head in heads],
            'action': 'ACCEPT' if accept else 'ABSTAIN_ESCALATE'}
