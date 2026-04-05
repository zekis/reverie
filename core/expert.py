"""Light expert — one lemma, no model, pure geometry."""

import os
import time
import json
import numpy as np

from config import (COOLING_RATE, EMBEDDING_DIM,
                    LOCAL_ALPHA_MIN_SAMPLES)
from learning.hebbian import apply_decay

# Wh-words are question markers, not answers. If they survive scene
# storage (e.g. from interaction scenes like "When asked What...")
# they must never be returned by gap extraction. This is a lexical
# filter, distinct from WH_TAGS which drives parsing upstream.
_WH_ANSWER_BLOCK = frozenset((
    "what", "who", "whom", "whose", "which",
    "where", "when", "why", "how",
))


class LightExpert:
    def __init__(self, lemma: str):
        self.lemma = lemma
        self.scene_vecs: list[np.ndarray] = []
        self.scene_texts: list[str] = []
        self.scene_slots: list[list[dict]] = []
        self.scene_weights: list[float] = []
        self.scene_matrix: np.ndarray | None = None

        self.activation: float = 0.0
        self.hit_count: int = 0
        self.miss_count: int = 0
        self.last_query_time: float = 0.0

        self.edge_weights: dict[str, float] = {}

    # ---- Learning stats ----

    @property
    def confidence(self) -> float:
        total = self.hit_count + self.miss_count
        if total < LOCAL_ALPHA_MIN_SAMPLES:
            return 0.5  # neutral
        return self.hit_count / total

    @property
    def local_alpha(self) -> float:
        """Confidence-scaled weight for reconciler. 0.5 = neutral."""
        total = self.hit_count + self.miss_count
        if total < LOCAL_ALPHA_MIN_SAMPLES:
            return 1.0
        c = self.hit_count / total
        return 0.5 + c  # range [0.5, 1.5]

    # ---- Core ops ----

    def _rebuild_matrix(self):
        if not self.scene_vecs:
            self.scene_matrix = None
            return
        mat = np.vstack(self.scene_vecs)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        mat_normed = mat / np.maximum(norms, 1e-8)
        weights = np.array(self.scene_weights)[:, None]
        self.scene_matrix = mat_normed * weights

    def store(self, scene_data: dict, apply_hebbian: bool = True,
              initial_weight: float = 1.0):
        """Add scene. apply_hebbian=True (default) decays existing scenes
        that contradict this one. initial_weight lets callers store new
        scenes below baseline (<1.0) so they must EARN retrieval weight
        via reinforcement — useful when mixing freshly-stored exploration
        scenes with previously-reinforced long-term scenes."""
        self.scene_texts.append(scene_data["source_text"])
        self.scene_slots.append(scene_data["slots"])
        vec = np.frombuffer(scene_data["vec"], dtype=np.float32).copy()
        if self.scene_vecs and apply_hebbian:
            apply_decay(vec, self.scene_vecs, self.scene_weights)
        self.scene_vecs.append(vec)
        self.scene_weights.append(initial_weight)
        self._rebuild_matrix()
        for key in scene_data.get("keys", []):
            if key != self.lemma and key not in self.edge_weights:
                self.edge_weights[key] = 0.1

    def query(self, query_vec_bytes: bytes, gap_role: str,
              subject_vec_bytes: bytes) -> dict:
        """Matrix multiply, extract gap slot, return answer + margin."""
        t0 = time.perf_counter()
        self.last_query_time = time.time()
        if self.scene_matrix is None:
            return {"answer": None, "score": 0.0, "margin": 0.0,
                    "lemma": self.lemma, "n_scenes": 0, "latency_us": 0}

        qvec = np.frombuffer(query_vec_bytes, dtype=np.float32)
        qn = qvec / (np.linalg.norm(qvec) + 1e-8)
        scores = self.scene_matrix @ qn
        sorted_idx = np.argsort(-scores)
        top_score = float(scores[sorted_idx[0]])
        runner_up = float(scores[sorted_idx[1]]) if len(sorted_idx) > 1 else 0.0
        margin = top_score - runner_up
        subject_vec = np.frombuffer(subject_vec_bytes, dtype=np.float32)

        # Try the top-K scenes in order, preferring STRONG extractions
        # (target-role slot matches) over weak fallbacks. A lower-scored
        # scene that has the slot the query asks for beats a higher-
        # scored scene that only has fallback slots.
        answer_text = None
        answer_vec_bytes = None
        best_slots = self.scene_slots[sorted_idx[0]]
        K = min(5, len(sorted_idx))
        for j in range(K):
            idx = int(sorted_idx[j])
            slot = self._extract_gap(
                self.scene_slots[idx], gap_role, subject_vec, strong_only=True)
            if slot is not None:
                answer_text = slot.get("text", "")
                answer_vec_bytes = slot.get("vec")
                best_slots = self.scene_slots[idx]
                break
        # If no strong extraction found, fall back on the top scene's weak
        if answer_text is None:
            slot = self._extract_gap(best_slots, gap_role, subject_vec)
            if slot is not None:
                answer_text = slot.get("text", "")
                answer_vec_bytes = slot.get("vec")
        if answer_text is None and top_score > 0.7:
            top_idx = int(sorted_idx[0])
            if top_idx < len(self.scene_texts):
                answer_text = self.scene_texts[top_idx]
                # Scene-level fallback: use the scene vector itself
                answer_vec_bytes = self.scene_vecs[top_idx].astype(
                    np.float32).tobytes()

        self.activation *= (1.0 - COOLING_RATE)
        latency = (time.perf_counter() - t0) * 1e6
        return {
            "answer": answer_text,
            "answer_vec": answer_vec_bytes,
            "score": float(top_score),
            "margin": float(margin),
            "lemma": self.lemma,
            "n_scenes": len(self.scene_vecs),
            "latency_us": int(latency),
        }

    def _extract_gap(self, slots, gap_role, subject_vec, strong_only=False):
        target_map = {
            "nsubj": ["nsubj"],
            "dobj": ["dobj", "ROOT"],
            "attr": ["attr", "acomp", "dobj"],
            "acomp": ["acomp", "attr"],
            "advmod": ["prep_from", "prep_with", "prep_by", "prep_through",
                       "prep_via", "prep_using", "dobj", "ROOT"],
            "nummod": ["nummod"],
            "det": ["attr", "acomp", "dobj"],
            "pobj": ["prep_on", "prep_in", "prep_at", "prep_for", "prep_with"],
        }
        # Text roles have a bespoke fallback chain. For custom / non-text
        # roles (e.g. "action", "outcome") fall through to the literal
        # role — the slot labelled with that role IS the target.
        targets = target_map.get(gap_role, [gap_role])
        skip = {"ROOT"}
        if gap_role in ("attr", "acomp", "det"):
            skip.add("nsubj")
        # nummod/compound should only answer matching gaps, never leak
        # into attr/dobj/nsubj fallbacks (interaction scenes with numeric
        # metadata were being returned as answers otherwise).
        if gap_role not in ("nummod",):
            skip.add("nummod")
        if gap_role not in ("compound",):
            skip.add("compound")

        def echo(slot):
            if "vec" not in slot or subject_vec is None:
                return False
            v = np.frombuffer(slot["vec"], dtype=np.float32)
            # Dim mismatch or zero norm → no echo check (disabled
            # automatically for non-text adapters that don't use a
            # subject vector).
            if len(v) != len(subject_vec) or len(v) == 0:
                return False
            vn = np.linalg.norm(v)
            svn = np.linalg.norm(subject_vec)
            if vn < 1e-8 or svn < 1e-8:
                return False
            return float(np.dot(v, subject_vec) / (vn * svn)) > 0.92

        def is_wh_answer(slot):
            return slot.get("text", "").strip().lower() in _WH_ANSWER_BLOCK

        # 1) Target pass — preferred role, skip WH, skip echo
        for target in targets:
            for slot in slots:
                if slot.get("is_gap") or is_wh_answer(slot):
                    continue
                role = slot.get("role", "")
                if target.startswith("prep_"):
                    if not role.startswith("prep_"):
                        continue
                elif role != target:
                    continue
                if echo(slot):
                    continue
                return slot
        if strong_only:
            return None  # caller wants target-match only
        # 2) Strict fallback — skip WH, skip echo, skip skip-roles
        for slot in slots:
            if (slot.get("is_gap") or is_wh_answer(slot)
                    or slot.get("role", "") in skip):
                continue
            if echo(slot):
                continue
            return slot
        # 3) Loose last-resort — drop echo and skip-roles. Only WH and
        # gap are excluded. A WH-only scene returns None; anything else
        # returns its first non-WH slot.
        for slot in slots:
            if slot.get("is_gap") or is_wh_answer(slot):
                continue
            return slot
        return None

    # ---- Temporal state ----

    def fire(self, strength: float = 0.5):
        self.activation = min(1.0, self.activation + strength)

    def cool(self):
        self.activation *= (1.0 - COOLING_RATE)

    def learn(self, correct: bool):
        if correct:
            self.hit_count += 1
        else:
            self.miss_count += 1

    # ---- Cold storage ----

    def save_cold(self, path: str):
        # Flatten all slot vectors into one array + per-scene counts so
        # the echo filter still works after reload.
        slot_counts = np.array([len(s) for s in self.scene_slots],
                               dtype=np.int32)
        if self.scene_slots:
            flat_slot_vecs = np.vstack([
                np.frombuffer(s["vec"], dtype=np.float32)
                for scene in self.scene_slots for s in scene
            ]) if slot_counts.sum() > 0 else np.zeros(
                (0, EMBEDDING_DIM), dtype=np.float32)
        else:
            flat_slot_vecs = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

        np.savez_compressed(
            path,
            scene_matrix=(self.scene_matrix if self.scene_matrix is not None
                          else np.zeros((0, EMBEDDING_DIM), dtype=np.float32)),
            scene_vecs=(np.array(self.scene_vecs) if self.scene_vecs
                        else np.zeros((0, EMBEDDING_DIM), dtype=np.float32)),
            scene_weights=(np.array(self.scene_weights) if self.scene_weights
                           else np.zeros(0, dtype=np.float32)),
            slot_vecs=flat_slot_vecs,
            slot_counts=slot_counts,
        )
        clean_slots = [
            [{"role": s.get("role", ""), "text": s.get("text", ""),
              "is_gap": s.get("is_gap", False)} for s in scene]
            for scene in self.scene_slots
        ]
        meta = {
            "lemma": self.lemma,
            "scene_texts": self.scene_texts,
            "scene_slots": clean_slots,
            "activation": self.activation,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "edge_weights": self.edge_weights,
        }
        with open(path + ".meta", "w") as f:
            json.dump(meta, f)

    def load_cold(self, path: str):
        data = np.load(path + ".npz")
        vecs = data["scene_vecs"]
        if len(vecs) > 0:
            self.scene_vecs = [vecs[i].astype(np.float32)
                               for i in range(len(vecs))]
            self.scene_matrix = data["scene_matrix"].astype(np.float32)
            if "scene_weights" in data and len(data["scene_weights"]) == len(vecs):
                self.scene_weights = data["scene_weights"].tolist()
            else:
                self.scene_weights = [1.0] * len(vecs)
        meta_path = path + ".meta"
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                self.scene_texts = meta.get("scene_texts", [])
                raw = meta.get("scene_slots", [])
                # Re-attach slot vectors from the flat array
                slot_vecs_flat = data.get("slot_vecs")
                slot_counts = data.get("slot_counts")
                if (slot_vecs_flat is not None and slot_counts is not None
                        and len(slot_counts) == len(raw)):
                    self.scene_slots = []
                    offset = 0
                    for scene_raw, count in zip(raw, slot_counts):
                        count = int(count)
                        scene = []
                        for j, s in enumerate(scene_raw):
                            if j < count:
                                vec = slot_vecs_flat[offset + j].astype(
                                    np.float32).tobytes()
                            else:
                                vec = np.zeros(EMBEDDING_DIM,
                                               dtype=np.float32).tobytes()
                            scene.append({**s, "vec": vec})
                        self.scene_slots.append(scene)
                        offset += count
                else:
                    # Back-compat: zero vecs (echo filter degraded)
                    self.scene_slots = [
                        [{**s, "vec": np.zeros(EMBEDDING_DIM,
                                               dtype=np.float32).tobytes()}
                         for s in scene]
                        for scene in raw
                    ]
                self.activation = meta.get("activation", 0.0)
                self.hit_count = meta.get("hit_count", 0)
                self.miss_count = meta.get("miss_count", 0)
                self.edge_weights = meta.get("edge_weights", {})
            except (json.JSONDecodeError, IOError):
                pass
