"""Synthesise expert responses. Group similar answers, pick highest avg margin.

No specificity hardcoding. Weights come from expert.local_alpha
(confidence-scaled), which is learned from hit/miss history.

Information gain: answers too similar to the question itself are
circular (e.g. interaction scenes that echo the question). They get
filtered below INFO_GAIN_THRESHOLD and down-weighted above it.
Biological analogue: CA1 novelty detection in the hippocampus.
"""

import numpy as np

from config import INFO_GAIN_THRESHOLD


def reconcile(responses: list[dict], embedder,
              alphas: dict[str, float] | None = None,
              query_vec: np.ndarray | None = None) -> dict:
    """Group semantically similar answers, weight by alpha × margin × gain."""
    alphas = alphas or {}
    valid = [r for r in responses if r.get("answer")]
    if not valid:
        return {"answer": None, "confidence": 0.0,
                "reinforced": 0, "margin": 0.0, "sources": []}

    # Information gain filter: suppress answers that echo the question
    if query_vec is not None:
        qn = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        filtered = []
        for r in valid:
            a_vec = embedder.embed(r["answer"])
            an = a_vec / (np.linalg.norm(a_vec) + 1e-8)
            gain = 1.0 - float(np.dot(qn, an))
            if gain < INFO_GAIN_THRESHOLD:
                continue  # circular — the answer just re-states the question
            r = dict(r)
            r["gain"] = gain
            filtered.append(r)
        valid = filtered
        if not valid:
            return {"answer": None, "confidence": 0.0,
                    "reinforced": 0, "margin": 0.0, "sources": []}

    groups = []
    for r in valid:
        alpha = alphas.get(r["lemma"], 1.0)
        gain = r.get("gain", 1.0)
        # Quadratic gain — circular answers score near zero, not
        # just halved. A modest-gain wh-word answer (gain≈0.5) gets
        # 0.25x its score; a genuinely novel answer (gain≈0.9)
        # keeps 0.81x — preserves ranking while crushing circulars.
        ws = r["score"] * alpha * (gain * gain)
        rv = embedder.embed(r["answer"])
        rvn = rv / (np.linalg.norm(rv) + 1e-8)
        merged = False
        for g in groups:
            gn = g["vec"] / (np.linalg.norm(g["vec"]) + 1e-8)
            if float(np.dot(rvn, gn)) > 0.8:
                g["scores"].append(ws)
                g["count"] += 1
                g["sources"].append(r["lemma"])
                if ws > g["best"]:
                    g["text"] = r["answer"]
                    g["best"] = ws
                merged = True
                break
        if not merged:
            groups.append({
                "text": r["answer"], "vec": rv,
                "scores": [ws], "best": ws, "count": 1,
                "sources": [r["lemma"]],
            })
    for g in groups:
        g["avg"] = sum(g["scores"]) / len(g["scores"])
    groups.sort(key=lambda g: -g["avg"])
    best = groups[0]
    margin = (best["avg"] - groups[1]["avg"]) if len(groups) > 1 else best["avg"]
    margin = max(0.0, min(1.0, margin))
    conf = margin * (1.0 + 0.1 * min(best["count"] - 1, 5))
    if margin < 0.002:
        return {"answer": None, "confidence": conf,
                "reinforced": 0, "margin": margin, "sources": []}
    return {
        "answer": best["text"], "confidence": conf,
        "reinforced": best["count"], "margin": margin,
        "sources": best["sources"],
    }
