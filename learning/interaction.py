"""Interaction outcomes update the brain. Three paths:

1. Correct, high confidence (margin > HIGH_SALIENCE)
   → strengthen edges, update hits, NO new scene.
   The scene already exists — that's why the answer was right.
   Replay buffer (in oscillator) handles offline consolidation.

2. Wrong or abstained, with ground truth provided
   → store a corrective semantic fact as a new scene:
       "The answer to {question} is {expected}."
   → update misses on sources that fired.
   High prediction error — worth consolidating.

3. Low-confidence answer (margin < LOW_SALIENCE) with no ground truth
   → ignore entirely. Routine exchange, no prediction error,
     nothing new to learn. No scene, no hit, no miss.

Interaction *scenes* are no longer stored. Episodic (what was said)
and semantic (what was learned) memory are separated. Only the latter
writes to the scene pool.
"""

import numpy as np

from config import (HIGH_SALIENCE_MARGIN, LOW_SALIENCE_MARGIN,
                    MATCH_SIMILARITY)

EDGE_LR = 0.1


def record_interaction(question: str, result: dict,
                       loader, parser, embedder,
                       expected: str | None = None):
    """Route the query outcome to one of three learning paths."""
    answer = result.get("answer")
    margin = result.get("margin", 0.0)
    reinforced = result.get("reinforced", 0)
    sources = result.get("sources", [])

    # Path A: explicit ground truth overrides implicit signal.
    if expected is not None:
        if _matches(answer, expected, embedder):
            _reward(loader, sources, reinforced)
        else:
            _penalise(loader, sources)
            _store_correction(question, expected, loader, parser, embedder)
        return

    # Path B: implicit signal from margin alone.
    # Abstention → miss.
    if not answer:
        _penalise(loader, sources)
        return

    # High-salience confident answer → hit + edge strengthen. No scene.
    if margin > HIGH_SALIENCE_MARGIN:
        _reward(loader, sources, reinforced)
        return

    # Below low-salience threshold → routine exchange, ignore.
    if margin < LOW_SALIENCE_MARGIN:
        return

    # Middle band → ambiguous, no update.


# ---- Paths ----

def _reward(loader, sources, reinforced):
    for lem in sources:
        if lem in loader.warm:
            loader.warm[lem].learn(True)
    if reinforced >= 2:
        _strengthen_edges(loader, sources)


def _penalise(loader, sources):
    for lem in sources:
        if lem in loader.warm:
            loader.warm[lem].learn(False)


def _store_correction(question, expected, loader, parser, embedder):
    """Write a clean semantic scene containing the correct answer."""
    q_clean = question.rstrip("?. ").strip()
    text = f"The answer to {q_clean} is {expected}."
    scene = parser.parse_scene(text, embedder)
    scene["source"] = "correction"
    for key in scene["keys"][:3]:
        loader.store_fact(key, scene)


def _strengthen_edges(loader, sources):
    """Fire-together-wire-together. Bidirectional, capped at 1.0."""
    for i, lem_a in enumerate(sources):
        if lem_a not in loader.warm:
            continue
        exp_a = loader.warm[lem_a]
        for lem_b in sources[i + 1:]:
            if lem_b not in loader.warm:
                continue
            exp_b = loader.warm[lem_b]
            w_ab = exp_a.edge_weights.get(lem_b, 0.1)
            exp_a.edge_weights[lem_b] = w_ab + EDGE_LR * (1.0 - w_ab)
            w_ba = exp_b.edge_weights.get(lem_a, 0.1)
            exp_b.edge_weights[lem_a] = w_ba + EDGE_LR * (1.0 - w_ba)


# ---- Similarity helpers ----

def _matches(answer, expected, embedder) -> bool:
    if not answer or not expected:
        return False
    a = answer.strip().lower()
    e = expected.strip().lower()
    if e in a or a in e:
        return True
    av = embedder.embed(answer)
    ev = embedder.embed(expected)
    an = av / (np.linalg.norm(av) + 1e-8)
    en = ev / (np.linalg.norm(ev) + 1e-8)
    return float(np.dot(an, en)) > MATCH_SIMILARITY
