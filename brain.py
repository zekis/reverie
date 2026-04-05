"""Brain — single entry point. Oscillator starts on init, never stops."""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.loader import ExpertLoader
from core.registry import Registry
from core.oscillator import Oscillator
from learning.interaction import (record_interaction,
                                  record_interaction_geometric)


class Brain:
    def __init__(self, warm_cold: bool = True, use_language: bool = True,
                 run_oscillator: bool = True):
        """A Brain is one modality's memory.

        use_language=True wires in the MiniLM embedder + spaCy parser
        so the text adapter (learn/query) works. Set False for pure
        sensorimotor / non-language brains — only learn_scene and
        query_scene are available in that mode, and the heavy language
        deps are never imported.
        """
        self.use_language = use_language
        if use_language:
            from language.embedder import Embedder
            from language import parser as _parser
            self.embedder = Embedder()
            self._parser = _parser
        else:
            self.embedder = None
            self._parser = None
        self.loader = ExpertLoader()
        self.registry = Registry(self.loader)
        self.oscillator = Oscillator(self.loader)
        if warm_cold:
            n = self.loader.warm_all_cold()
            if n > 0:
                print(f"[brain] pre-warmed {n} experts from cold storage")
        if run_oscillator:
            self.oscillator.start()

    def save(self) -> int:
        """Persist all warm experts to cold storage. Returns count."""
        return self.loader.save_all()

    # ---- Public API ----

    # ---- Primitives: type-agnostic ----

    def learn_scene(self, scene: dict, keys: list[str] | None = None,
                    top_k: int = 3, apply_hebbian: bool = True) -> int:
        """Store a pre-built scene. scene must contain {vec, slots, keys,
        source_text}. Stores into top_k keys by specificity (fewest
        existing scenes). Set top_k=None to use all keys. Set
        apply_hebbian=False to accumulate experience without decay
        (motor-babbling / exploration phase)."""
        keys = keys if keys is not None else scene.get("keys", [])
        if not keys:
            return 0
        scored = []
        for k in keys:
            n = len(self.loader.warm[k].scene_vecs) if k in self.loader.warm else 0
            scored.append((k, n))
        scored.sort(key=lambda x: x[1])
        if top_k is not None:
            scored = scored[:top_k]
        for key, _ in scored:
            self.loader.store_fact(key, scene, apply_hebbian=apply_hebbian)
        return len(scored)

    def query_scene(self, gap: dict, keys: list[str]) -> dict:
        """Run an oscillating query from a pre-built gap dict. Records
        the interaction via the geometric (implicit) path."""
        result = self.oscillator.query(gap, keys)
        record_interaction_geometric(result, self.loader)
        return result

    # ---- Text adapter (requires use_language=True) ----

    def learn(self, text: str):
        """Store a fact. Top 3 keys by specificity (fewest existing scenes)."""
        if not self.use_language:
            raise RuntimeError("learn(text) requires use_language=True")
        scene = self._parser.parse_scene(text, self.embedder)
        return self.learn_scene(scene)

    def learn_bulk(self, texts: list[str]):
        """Bulk storage mode — no LRU eviction until done."""
        if not self.use_language:
            raise RuntimeError("learn_bulk(texts) requires use_language=True")
        self.loader.start_bulk()
        try:
            for t in texts:
                self.learn(t)
        finally:
            self.loader.end_bulk()

    def query(self, question: str, expected: str | None = None) -> dict:
        """Run an oscillating text query. If expected is provided, the
        interaction uses ground-truth salience; otherwise falls back
        to implicit margin-based signal."""
        if not self.use_language:
            raise RuntimeError("query(text) requires use_language=True")
        gap = self._parser.parse_gap(question, self.embedder)
        keys = self._parser.extract_keys(question)
        result = self.oscillator.query(gap, keys)
        record_interaction(question, result, self.loader,
                           self._parser, self.embedder, expected=expected)
        return result

    # ---- Introspection ----

    def snapshot(self) -> dict:
        out = {}
        for lemma, exp in self.loader.warm.items():
            if exp.activation <= 0.05:
                continue
            top_scene = None
            if exp.scene_vecs:
                idx = int(np.argmax(exp.scene_weights))
                top_scene = exp.scene_texts[idx]
            out[lemma] = {
                "activation": round(exp.activation, 3),
                "confidence": round(exp.confidence, 3),
                "scenes": len(exp.scene_vecs),
                "top_scene": top_scene,
                "warm_edges": sorted(exp.edge_weights.items(),
                                     key=lambda x: -x[1])[:3],
            }
        return out

    def daydream_state(self, threshold: float = 0.05) -> list[str]:
        """Lemmas currently above warmth threshold, hottest first."""
        active = sorted(
            [(l, e.activation) for l, e in self.loader.warm.items()
             if e.activation > threshold],
            key=lambda x: -x[1],
        )
        return [lemma for lemma, _ in active[:10]]

    def stats(self) -> dict:
        return self.loader.stats()

    def stop(self):
        """Clean shutdown — stops oscillator then persists all warm experts."""
        self.oscillator.stop()
        n = self.save()
        print(f"[brain] saved {n} experts to cold storage")
