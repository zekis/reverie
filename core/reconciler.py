"""Geometric reconciler — type-agnostic answer synthesis.

Each expert response carries answer_vec (the slot's stored vector).
The reconciler groups responses by cosine similarity on answer_vec,
weights by alpha x margin x info-gain, and returns the strongest group.

No embedder. No language assumption. Works for any modality.

Information gain: answers too similar to the query itself are circular
(e.g. scenes that echo the query). They get filtered below
INFO_GAIN_THRESHOLD and down-weighted above it via gain^2.
"""

import numpy as np

from config import INFO_GAIN_THRESHOLD


def reconcile(responses: list[dict],
              alphas: dict[str, float] | None = None,
              query_vec: np.ndarray | None = None) -> dict:
    """Group responses by answer_vec similarity, weight by alpha*score*gain^2."""
    alphas = alphas or {}
    valid = [r for r in responses
             if r.get("answer") is not None and r.get("answer_vec") is not None]
    if not valid:
        return {"answer": None, "answer_vec": None, "confidence": 0.0,
                "reinforced": 0, "margin": 0.0, "sources": []}

    # Information gain filter: suppress answers that echo the query.
    # Only meaningful when query and answer live in the same vector
    # space (same dim). Cross-modal or mixed-shape scenes skip the
    # filter — gain defaults to 1.0 (treated as maximally novel).
    if query_vec is not None:
        qn = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        filtered = []
        for r in valid:
            av = np.frombuffer(r["answer_vec"], dtype=np.float32)
            an_norm = np.linalg.norm(av)
            if an_norm < 1e-8:
                continue
            if len(av) != len(query_vec):
                # Dim mismatch — skip gain filter, keep response.
                r = dict(r)
                r["gain"] = 1.0
                filtered.append(r)
                continue
            an = av / an_norm
            gain = 1.0 - float(np.dot(qn, an))
            if gain < INFO_GAIN_THRESHOLD:
                continue  # circular
            r = dict(r)
            r["gain"] = gain
            filtered.append(r)
        valid = filtered
        if not valid:
            return {"answer": None, "answer_vec": None, "confidence": 0.0,
                    "reinforced": 0, "margin": 0.0, "sources": []}

    groups = []
    for r in valid:
        alpha = alphas.get(r["lemma"], 1.0)
        gain = r.get("gain", 1.0)
        # Quadratic gain — circular answers score near zero, not
        # just halved. A modest-gain answer (gain~0.5) gets 0.25x;
        # a genuinely novel answer (gain~0.9) keeps 0.81x.
        ws = r["score"] * alpha * (gain * gain)
        rv = np.frombuffer(r["answer_vec"], dtype=np.float32)
        rn = np.linalg.norm(rv)
        if rn < 1e-8:
            continue
        rvn = rv / rn
        merged = False
        for g in groups:
            gn = g["vec"] / (np.linalg.norm(g["vec"]) + 1e-8)
            if float(np.dot(rvn, gn)) > 0.8:
                g["scores"].append(ws)
                g["count"] += 1
                g["sources"].append(r["lemma"])
                if ws > g["best"]:
                    g["text"] = r["answer"]
                    g["answer_vec"] = r["answer_vec"]
                    g["best"] = ws
                merged = True
                break
        if not merged:
            groups.append({
                "text": r["answer"], "vec": rv,
                "answer_vec": r["answer_vec"],
                "scores": [ws], "best": ws, "count": 1,
                "sources": [r["lemma"]],
            })
    if not groups:
        return {"answer": None, "answer_vec": None, "confidence": 0.0,
                "reinforced": 0, "margin": 0.0, "sources": []}
    for g in groups:
        g["avg"] = sum(g["scores"]) / len(g["scores"])
    groups.sort(key=lambda g: -g["avg"])
    best = groups[0]
    margin = (best["avg"] - groups[1]["avg"]) if len(groups) > 1 else best["avg"]
    margin = max(0.0, min(1.0, margin))
    conf = margin * (1.0 + 0.1 * min(best["count"] - 1, 5))
    if margin < 0.002:
        return {"answer": None, "answer_vec": None, "confidence": conf,
                "reinforced": 0, "margin": margin, "sources": []}
    return {
        "answer": best["text"],
        "answer_vec": best["answer_vec"],
        "confidence": conf,
        "reinforced": best["count"],
        "margin": margin,
        "sources": best["sources"],
    }
