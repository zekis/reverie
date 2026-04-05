"""Context vector state for one query session.

Starts as query embedding. After each cycle, blends partial answer.
Tracks shift from original.
"""

import numpy as np

from config import QUERY_WEIGHT, ANSWER_WEIGHT


class Feedback:
    def __init__(self, query_vec: np.ndarray):
        self.original = query_vec.copy()
        self.context = query_vec.copy()
        self.history: list[np.ndarray] = [query_vec.copy()]

    def blend(self, answer_vec: np.ndarray):
        self.context = QUERY_WEIGHT * self.context + ANSWER_WEIGHT * answer_vec
        n = np.linalg.norm(self.context)
        if n > 1e-8:
            self.context /= n
        self.history.append(self.context.copy())

    @property
    def shifted(self) -> bool:
        on = np.linalg.norm(self.original) + 1e-8
        cn = np.linalg.norm(self.context) + 1e-8
        sim = float(np.dot(self.original, self.context) / (on * cn))
        return (1.0 - sim) > 0.05
