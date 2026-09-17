"""Standalone single-entity version history; not wired into the Stage-3 policy.

Updates record unverified candidates, not automatic acceptance decisions. No GT
quality is consumed. Reliability stays None unless supplied by a future verifier.
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np


def bbox_from_mask(mask):
    """Pixel xyxy with exclusive right/bottom, matching the existing executor."""
    ys, xs = np.where(np.asarray(mask) > 0)
    if len(xs) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


@dataclass
class EntityMemory:
    entity_id: str
    mask: np.ndarray
    bbox: tuple | None
    image: Any
    provenance: dict
    version: int = 1
    appearance_feature: Any = None
    semantic_feature: Any = None
    reliability: float | None = None


class EntityMemoryStore:
    """Persistent across calls in one process, with copy-isolated snapshots.

    No disk persistence, feature encoder, verifier or multi-entity retrieval is
    implemented. Entity IDs are caller-assigned bookkeeping, not proof of identity.
    """

    def __init__(self, initial):
        snapshot = deepcopy(initial)
        snapshot.version = 1
        self._versions = [snapshot]

    @classmethod
    def write(cls, entity_id, mask, image, provenance, *, appearance_feature=None,
              semantic_feature=None, reliability=None):
        return cls(EntityMemory(
            entity_id=entity_id, mask=np.asarray(mask), bbox=bbox_from_mask(mask),
            image=image, provenance=provenance, appearance_feature=appearance_feature,
            semantic_feature=semantic_feature, reliability=reliability,
        ))

    def read(self, version=None):
        if version is None:
            return deepcopy(self._versions[-1])
        if not 1 <= version <= len(self._versions):
            raise ValueError("Unknown memory version")
        return deepcopy(self._versions[version - 1])

    def update_candidate(self, mask, image, provenance, *, appearance_feature=None,
                         semantic_feature=None, reliability=None):
        # New visual evidence invalidates old features/reliability unless recomputed.
        candidate = EntityMemory(
            entity_id=self._versions[-1].entity_id,
            mask=np.asarray(mask), bbox=bbox_from_mask(mask), image=image,
            provenance=provenance, version=len(self._versions) + 1,
            appearance_feature=appearance_feature, semantic_feature=semantic_feature,
            reliability=reliability,
        )
        self._versions.append(deepcopy(candidate))
        return self.read()

    def rollback(self, *, step, version=None):
        """Restore a prior snapshot as a NEW version, keeping the audit history."""
        if version is None:
            version = len(self._versions) - 1
        if not 1 <= version < len(self._versions):
            raise ValueError("Rollback requires a prior memory version")
        restored = self.read(version)
        restored.provenance = {
            **restored.provenance, "source_action": "ROLLBACK_MEMORY", "step": step,
            "restored_version": version, "rollback_from_version": len(self._versions),
        }
        restored.version = len(self._versions) + 1
        self._versions.append(restored)
        return self.read()


def from_anchor_state(state, *, entity_id, source_action, step):
    """Map the legacy best-anchor state without interpreting oracle IoU as reliability.

    `anchor` was selected using GT in the old executor. Mark that provenance;
    this adapter does not make its input oracle-free. `anchor_image` is preserved
    with the mask rather than substituted by the possibly newer current_image.
    """
    if state.get("anchor") is None:
        raise ValueError("No anchor to write into memory")
    return EntityMemoryStore.write(
        entity_id, state["anchor"], state["anchor_image"],
        {"source_action": source_action, "step": step,
         "image_state": state["anchor_state"], "source": "legacy_executor_anchor",
         "selection_uses_gt": True},
    )
