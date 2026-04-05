"""Bidirectional Hebbian storage — association + disassociation.

Called by expert.store(). Pure functions on vector arrays.
"""

import numpy as np

from config import HEBBIAN_THRESHOLD, HEBBIAN_DECAY, HEBBIAN_MIN_WEIGHT


def apply_decay(new_vec: np.ndarray, existing_vecs: list[np.ndarray],
                weights: list[float]) -> None:
    """Mutate weights in place: decay scenes dissimilar to new_vec.

    Scenes above HEBBIAN_THRESHOLD left untouched.
    Scenes below get weights *= HEBBIAN_DECAY, floored at HEBBIAN_MIN_WEIGHT.
    """
    n = np.linalg.norm(new_vec)
    if n < 1e-8:
        return
    vn = new_vec / n
    for i, existing in enumerate(existing_vecs):
        en = np.linalg.norm(existing)
        if en < 1e-8:
            continue
        sim = float(np.dot(vn, existing / en))
        if sim < HEBBIAN_THRESHOLD:
            weights[i] = max(HEBBIAN_MIN_WEIGHT, weights[i] * HEBBIAN_DECAY)
