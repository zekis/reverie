"""HTTP API — decouples the brain from its clients.

The brain is a domain-agnostic substrate. Text is the first client,
but any structured data (scenes with slots + vectors) can be stored
and queried. Clients connect from any language via HTTP.

Endpoints:
  POST /learn          {"text": "..."}                  → add fact
  POST /learn/bulk     {"texts": ["...", ...]}          → bulk add
  POST /learn/scene    {"slots":[], "vec":"<b64>", ...} → domain-agnostic
  POST /query          {"question": "..."}              → oscillating query
  POST /forget/{lemma}                                  → decay (not delete)
  GET  /state                                           → daydream + top warm
  GET  /snapshot                                        → full warm snapshot
  GET  /stats                                           → loader stats
  GET  /edges/{lemma}                                   → expert's edges
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import numpy as np

from config import HEBBIAN_MIN_WEIGHT, EMBEDDING_DIM
from language import parser as _parser


class LearnRequest(BaseModel):
    text: str


class BulkLearnRequest(BaseModel):
    texts: list[str]


class QueryRequest(BaseModel):
    question: str
    expected: str | None = None


class SceneSlot(BaseModel):
    role: str
    text: str = ""
    vec: str  # base64-encoded float32 bytes
    is_gap: bool = False


class SceneRequest(BaseModel):
    slots: list[SceneSlot]
    vec: str               # scene-level vector, base64
    keys: list[str]        # which experts to store in
    source: str = "external"
    timestamp: float | None = None


def _decode_vec(b64: str) -> bytes:
    """Decode base64 string to raw float32 bytes. Validates length."""
    raw = base64.b64decode(b64)
    if len(raw) != EMBEDDING_DIM * 4:
        raise HTTPException(
            400, f"vec length {len(raw)} bytes, "
                 f"expected {EMBEDDING_DIM * 4} (=EMBEDDING_DIM*4)")
    return raw


def build_app(brain, log_events: bool = True) -> FastAPI:
    """Wrap a live Brain instance in an HTTP interface."""
    app = FastAPI(title="reverie", version="0.1")

    def _log(msg: str):
        if log_events:
            print(f"[api] {msg}", flush=True)

    @app.post("/learn")
    def learn(req: LearnRequest):
        stored = brain.learn(req.text)
        _log(f"learn  -> {stored} experts  :: {req.text!r}")
        return {"stored_in": stored, "text": req.text}

    @app.post("/learn/bulk")
    def learn_bulk(req: BulkLearnRequest):
        brain.learn_bulk(req.texts)
        return {"count": len(req.texts), "stats": brain.stats()}

    @app.post("/learn/scene")
    def learn_scene(req: SceneRequest):
        """Domain-agnostic storage: any system that can produce slot
        vectors (vision, speech, sensors) writes scenes here directly.
        No text pipeline, no MiniLM — vectors arrive pre-computed.
        """
        scene_data = {
            "slots": [
                {"role": s.role, "text": s.text,
                 "vec": _decode_vec(s.vec), "is_gap": s.is_gap}
                for s in req.slots
            ],
            "source_text": req.source,
            "vec": _decode_vec(req.vec),
            "keys": req.keys,
        }
        for key in req.keys[:3]:
            brain.loader.store_fact(key, scene_data)
        _log(f"scene  -> {req.keys[:3]} :: source={req.source}")
        return {"stored_in": req.keys[:3], "source": req.source}

    @app.post("/query")
    def query(req: QueryRequest):
        result = brain.query(req.question, expected=req.expected)
        tag = f" [exp={req.expected!r}]" if req.expected else ""
        _log(f"query  -> {result.get('answer')!r} "
             f"(conf={result.get('confidence', 0):.2f} "
             f"cycles={result.get('cycles')} "
             f"sources={result.get('sources')}){tag} "
             f":: {req.question!r}")
        return result

    @app.post("/forget/{lemma}")
    def forget(lemma: str):
        """Decay scenes, outgoing edges, incoming edges, and activation.

        Scene weights halved (floored at HEBBIAN_MIN_WEIGHT).
        Outgoing edges from this expert halved (floored at EDGE_MIN).
        Incoming edges from every warm expert halved.
        Activation knocked down to 10% so daydream can't resurrect it
        through residual warmth before the next cool-tick.

        This is the biological equivalent of active inhibition — not
        deletion, but making the memory very quiet.
        """
        if lemma not in brain.loader.warm:
            brain.loader.warm_expert(lemma)
        if lemma not in brain.loader.warm:
            raise HTTPException(404, f"no expert named '{lemma}'")
        exp = brain.loader.warm[lemma]
        EDGE_MIN = 0.05

        # 1. Scene weights
        if exp.scene_weights:
            s_before = sum(exp.scene_weights) / len(exp.scene_weights)
            for i in range(len(exp.scene_weights)):
                exp.scene_weights[i] = max(
                    HEBBIAN_MIN_WEIGHT, exp.scene_weights[i] * 0.5)
            exp._rebuild_matrix()
            s_after = sum(exp.scene_weights) / len(exp.scene_weights)
        else:
            s_before = s_after = 0.0

        # 2. Outgoing edges from this expert
        out_count = len(exp.edge_weights)
        for k in exp.edge_weights:
            exp.edge_weights[k] = max(EDGE_MIN, exp.edge_weights[k] * 0.5)

        # 3. Incoming edges from all other warm experts
        in_count = 0
        for other_lemma, other in brain.loader.warm.items():
            if other_lemma == lemma:
                continue
            if lemma in other.edge_weights:
                other.edge_weights[lemma] = max(
                    EDGE_MIN, other.edge_weights[lemma] * 0.5)
                in_count += 1

        # 4. Knock activation down so daydream can't sustain it
        act_before = exp.activation
        exp.activation *= 0.1

        _log(f"forget -> {lemma} scenes={len(exp.scene_weights)} "
             f"w:{s_before:.2f}→{s_after:.2f} "
             f"out_edges={out_count} in_edges={in_count} "
             f"act:{act_before:.2f}→{exp.activation:.2f}")
        return {
            "lemma": lemma, "scenes": len(exp.scene_weights),
            "avg_weight_before": round(s_before, 3),
            "avg_weight_after": round(s_after, 3),
            "outgoing_edges_decayed": out_count,
            "incoming_edges_decayed": in_count,
            "activation_before": round(act_before, 3),
            "activation_after": round(exp.activation, 3),
            "action": "decayed",
        }

    @app.get("/state")
    def state():
        snap = brain.snapshot()
        top = sorted(snap.items(),
                     key=lambda x: -x[1]["activation"])[:5]
        return {
            "thinking_about": brain.daydream_state(),
            "top_warm": [
                {"lemma": lem, "activation": d["activation"],
                 "confidence": d["confidence"], "scenes": d["scenes"]}
                for lem, d in top
            ],
            "stats": brain.stats(),
        }

    @app.get("/snapshot")
    def snapshot():
        return brain.snapshot()

    @app.get("/stats")
    def stats():
        return brain.stats()

    @app.get("/replay")
    def replay_state():
        """Current replay buffer — recent interactions being consolidated."""
        buf = brain.oscillator._replay_buffer
        return {
            "size": len(buf),
            "entries": [
                {"sources": e["sources"], "strength": round(e["strength"], 3),
                 "replays": e["replays"]}
                for e in buf
            ],
        }

    class DebugText(BaseModel):
        text: str

    @app.post("/debug/keys")
    def debug_keys(req: DebugText):
        """Show what the parser extracts from a given text."""
        keys = _parser.extract_keys(req.text)
        gap = _parser.parse_gap(req.text, brain.embedder)
        return {
            "text": req.text,
            "keys": keys,
            "gap_role": gap["role"],
            "gap_root": gap["root"],
            "gap_subject": gap.get("subject"),
            "is_boolean": gap.get("is_boolean", False),
            "given": gap.get("given", []),
        }

    class DebugExpertMatch(BaseModel):
        text: str
        lemma: str
        k: int = 5

    @app.post("/debug/match")
    def debug_match(req: DebugExpertMatch):
        """Show the top-K best matching scenes in one expert for a query.
        Reveals weight × similarity trade-off and which scene wins."""
        import numpy as np
        if req.lemma not in brain.loader.warm:
            brain.loader.warm_expert(req.lemma)
        if req.lemma not in brain.loader.warm:
            raise HTTPException(404, f"'{req.lemma}' not warm")
        exp = brain.loader.warm[req.lemma]
        if exp.scene_matrix is None:
            return {"lemma": req.lemma, "error": "no scenes"}
        gap = _parser.parse_gap(req.text, brain.embedder)
        qvec = gap["query_vec"]
        qn = qvec / (np.linalg.norm(qvec) + 1e-8)
        # cos similarity = raw dot product of normalized vecs
        mat = np.vstack(exp.scene_vecs)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        mat_normed = mat / np.maximum(norms, 1e-8)
        raw_cos = mat_normed @ qn
        weighted = exp.scene_matrix @ qn  # this is what query() uses
        idx = np.argsort(-weighted)[:req.k]
        return {
            "lemma": req.lemma,
            "query": req.text,
            "gap_role": gap["role"],
            "top": [
                {
                    "rank": i + 1,
                    "scene_idx": int(j),
                    "weight": round(float(exp.scene_weights[j]), 3),
                    "cos_sim": round(float(raw_cos[j]), 3),
                    "weighted_score": round(float(weighted[j]), 3),
                    "text": exp.scene_texts[j][:80],
                    "slots": [
                        {"role": s["role"], "text": s["text"][:40]}
                        for s in exp.scene_slots[j]
                    ],
                }
                for i, j in enumerate(idx)
            ],
        }

    @app.post("/debug/query")
    def debug_query(req: DebugText):
        """Run one cycle-0 query and return per-expert responses
        before reconciliation. Shows what each fired expert produced."""
        gap = _parser.parse_gap(req.text, brain.embedder)
        keys = _parser.extract_keys(req.text)
        responses = []
        for lem in keys:
            expert = brain.loader.warm_expert(lem) or brain.loader.get_or_create(lem)
            if expert.scene_matrix is None:
                responses.append({"lemma": lem, "status": "no scenes"})
                continue
            r = expert.query(gap["query_vec"].tobytes(), gap["role"],
                             gap["subject_vec"].tobytes())
            responses.append({
                "lemma": lem,
                "answer": r.get("answer"),
                "score": round(r.get("score", 0.0), 3),
                "margin": round(r.get("margin", 0.0), 3),
                "n_scenes": r.get("n_scenes", 0),
                "local_alpha": round(expert.local_alpha, 3),
            })
        return {
            "text": req.text,
            "keys": keys,
            "gap_role": gap["role"],
            "responses": responses,
        }

    @app.post("/debug/scene")
    def debug_scene(req: DebugText):
        """Show how a scene is parsed — slots, roles, gaps."""
        scene = _parser.parse_scene(req.text, brain.embedder)
        return {
            "text": req.text,
            "keys": scene["keys"],
            "slots": [
                {"role": s["role"], "text": s["text"], "is_gap": s["is_gap"]}
                for s in scene["slots"]
            ],
        }

    @app.post("/save")
    def save():
        n = brain.save()
        _log(f"save   -> {n} experts persisted")
        return {"saved": n, "stats": brain.stats()}

    @app.get("/edges/{lemma}")
    def edges(lemma: str):
        if lemma not in brain.loader.warm:
            raise HTTPException(404, f"'{lemma}' not in warm pool")
        exp = brain.loader.warm[lemma]
        sorted_edges = sorted(exp.edge_weights.items(),
                              key=lambda x: -x[1])
        return {
            "lemma": lemma,
            "activation": exp.activation,
            "confidence": exp.confidence,
            "scenes": len(exp.scene_vecs),
            "hits": exp.hit_count,
            "misses": exp.miss_count,
            "edges": [{"to": k, "weight": round(v, 3)}
                      for k, v in sorted_edges],
        }

    return app
