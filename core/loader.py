"""Warm pool + cold storage + LRU eviction."""

import os
import threading
import numpy as np

from config import MAX_WARM_EXPERTS, COLD_STORE_DIR
from core.expert import LightExpert


class ExpertLoader:
    def __init__(self):
        self.warm: dict[str, LightExpert] = {}
        self._lock = threading.Lock()
        self.bulk_mode: bool = False  # disable eviction during bulk load
        os.makedirs(COLD_STORE_DIR, exist_ok=True)

    # ---- Bulk mode ----

    def start_bulk(self):
        self.bulk_mode = True

    def end_bulk(self):
        """After bulk storage, trim to MAX_WARM_EXPERTS by scene density."""
        self.bulk_mode = False
        if len(self.warm) <= MAX_WARM_EXPERTS:
            return
        ranked = sorted(
            self.warm.items(),
            key=lambda kv: -len(kv[1].scene_vecs)
        )
        keep = dict(ranked[:MAX_WARM_EXPERTS])
        evicting = [k for k in self.warm if k not in keep]
        for lemma in evicting:
            expert = self.warm[lemma]
            expert.save_cold(os.path.join(COLD_STORE_DIR, lemma))
            del self.warm[lemma]

    # ---- Core ----

    def get_or_create(self, lemma: str) -> LightExpert:
        with self._lock:
            if lemma in self.warm:
                return self.warm[lemma]
            if not self.bulk_mode and len(self.warm) >= MAX_WARM_EXPERTS:
                self._evict_lru()
            expert = LightExpert(lemma)
            cold_path = os.path.join(COLD_STORE_DIR, lemma)
            if os.path.exists(cold_path + ".npz"):
                expert.load_cold(cold_path)
            self.warm[lemma] = expert
            return expert

    def _evict_lru(self):
        if not self.warm:
            return
        lru = min(self.warm, key=lambda l: self.warm[l].last_query_time)
        expert = self.warm[lru]
        expert.save_cold(os.path.join(COLD_STORE_DIR, lru))
        del self.warm[lru]

    def store_fact(self, lemma: str, scene_data: dict,
                   apply_hebbian: bool = True):
        expert = self.get_or_create(lemma)
        expert.store(scene_data, apply_hebbian=apply_hebbian)

    def query_expert(self, lemma: str, query_vec: bytes,
                     gap_role: str, subject_vec: bytes) -> dict:
        expert = self.get_or_create(lemma)
        return expert.query(query_vec, gap_role, subject_vec)

    def warm_expert(self, lemma: str) -> LightExpert | None:
        cold_path = os.path.join(COLD_STORE_DIR, lemma)
        if not os.path.exists(cold_path + ".npz"):
            return self.warm.get(lemma)
        return self.get_or_create(lemma)

    def is_cold(self, lemma: str) -> bool:
        return (lemma not in self.warm and os.path.exists(
            os.path.join(COLD_STORE_DIR, lemma + ".npz")))

    def _cold_lemmas(self) -> list[str]:
        if not os.path.exists(COLD_STORE_DIR):
            return []
        return [f[:-4] for f in os.listdir(COLD_STORE_DIR)
                if f.endswith(".npz")]

    def get_nearest(self, vec_bytes: bytes, k: int = 5,
                    include_cold: bool = False) -> list[dict]:
        qvec = np.frombuffer(vec_bytes, dtype=np.float32)
        qn = qvec / (np.linalg.norm(qvec) + 1e-8)
        scored = []
        for lem, exp in self.warm.items():
            if exp.scene_vecs:
                evec = np.mean(exp.scene_vecs, axis=0)
                en = evec / (np.linalg.norm(evec) + 1e-8)
                scored.append({
                    "lemma": lem, "sim": float(np.dot(qn, en)),
                    "n_scenes": len(exp.scene_vecs), "is_cold": False,
                })
        if include_cold:
            for lem in self._cold_lemmas():
                if lem in self.warm:
                    continue
                try:
                    data = np.load(os.path.join(COLD_STORE_DIR, lem + ".npz"))
                    vecs = data["scene_vecs"]
                    if len(vecs) > 0:
                        evec = np.mean(vecs, axis=0)
                        en = evec / (np.linalg.norm(evec) + 1e-8)
                        scored.append({
                            "lemma": lem, "sim": float(np.dot(qn, en)),
                            "n_scenes": len(vecs), "is_cold": True,
                        })
                except Exception:
                    pass
        scored.sort(key=lambda x: -x["sim"])
        return scored[:k]

    def save_all(self) -> int:
        """Persist every warm expert to cold storage. Returns count saved."""
        saved = 0
        for lemma, expert in self.warm.items():
            expert.save_cold(os.path.join(COLD_STORE_DIR, lemma))
            saved += 1
        return saved

    def warm_all_cold(self, limit: int = None) -> int:
        """Pre-load cold experts into warm pool. Bounded by MAX_WARM_EXPERTS
        or the given limit, whichever is smaller."""
        cap = min(limit or MAX_WARM_EXPERTS, MAX_WARM_EXPERTS)
        loaded = 0
        for lemma in self._cold_lemmas():
            if len(self.warm) >= cap:
                break
            if lemma in self.warm:
                continue
            self.get_or_create(lemma)  # loads from cold inside
            loaded += 1
        return loaded

    def stats(self) -> dict:
        return {
            "warm": len(self.warm),
            "cold": len(self._cold_lemmas()),
            "total_scenes": sum(len(e.scene_vecs) for e in self.warm.values()),
        }
