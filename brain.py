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
                    top_k: int = 3, apply_hebbian: bool = True,
                    initial_weight: float = 1.0) -> list:
        """Store a pre-built scene. scene must contain {vec, slots, keys,
        source_text}. Stores into top_k keys by specificity (fewest
        existing scenes). Set top_k=None to use all keys. Set
        apply_hebbian=False to accumulate experience without decay
        (motor-babbling / exploration phase).

        Returns list of (expert, scene_idx) handles — one per key the
        scene was stored into. Callers who track reinforcement use these
        handles with brain.reinforce() and brain.session_end().
        """
        keys = keys if keys is not None else scene.get("keys", [])
        if not keys:
            return []
        scored = []
        for k in keys:
            n = len(self.loader.warm[k].scene_vecs) if k in self.loader.warm else 0
            scored.append((k, n))
        scored.sort(key=lambda x: x[1])
        if top_k is not None:
            scored = scored[:top_k]
        handles = []
        for key, _ in scored:
            expert = self.loader.get_or_create(key)
            expert.store(scene, apply_hebbian=apply_hebbian,
                         initial_weight=initial_weight)
            handles.append((expert, len(expert.scene_vecs) - 1))
        return handles

    def reinforce(self, handles: list, multiplier: float,
                  min_w: float = 0.1, max_w: float = 2.5,
                  rebuild: bool = True):
        """Multiply scene weights by multiplier for each (expert, idx)
        handle. Clips to [min_w, max_w]. Rebuilds affected expert
        matrices so the weight change is visible to subsequent queries.

        Set rebuild=False in tight loops and call brain.rebuild() when
        you're ready to refresh retrieval."""
        touched = set()
        for exp, idx in handles:
            if 0 <= idx < len(exp.scene_weights):
                exp.scene_weights[idx] = float(np.clip(
                    exp.scene_weights[idx] * multiplier, min_w, max_w))
                touched.add(id(exp))
        if rebuild:
            for exp in self.loader.warm.values():
                if id(exp) in touched and exp.scene_vecs:
                    exp._rebuild_matrix()

    def rebuild(self):
        """Rebuild scene matrices for every warm expert. Call this after
        deferred reinforce/decay updates."""
        for exp in self.loader.warm.values():
            if exp.scene_vecs:
                exp._rebuild_matrix()

    def session_begin(self) -> dict:
        """Snapshot current per-expert scene counts. Use with
        session_end() to consolidate: any scenes stored between
        begin/end that weren't reinforced get their weights decayed."""
        return {id(exp): len(exp.scene_vecs)
                for exp in self.loader.warm.values()}

    def session_end(self, snapshot: dict, reinforced_handles: list,
                    decay: float = 0.85, min_w: float = 0.1):
        """Consolidation: decay the weight of any scene stored since
        snapshot that wasn't in reinforced_handles. Reinforced scenes
        keep their boosted weight; baseline new scenes fade. Old
        scenes from prior sessions are untouched.

        Biological analogue: sleep consolidation — unreinforced
        experience weakens, reinforced experience persists."""
        reinforced_set = {(id(exp), idx) for exp, idx in reinforced_handles}
        for exp in self.loader.warm.values():
            start = snapshot.get(id(exp), 0)
            for i in range(start, len(exp.scene_weights)):
                if (id(exp), i) not in reinforced_set:
                    exp.scene_weights[i] = max(
                        min_w, exp.scene_weights[i] * decay)
        for exp in self.loader.warm.values():
            if exp.scene_vecs:
                exp._rebuild_matrix()

    def query_scene(self, gap: dict, keys: list[str]) -> dict:
        """Run an oscillating query from a pre-built gap dict. Records
        the interaction via the geometric (implicit) path."""
        result = self.oscillator.query(gap, keys)
        record_interaction_geometric(result, self.loader)
        return result

    # ---- Text adapter (requires use_language=True) ----

    def learn(self, text: str):
        """Store a fact. Top 3 keys by specificity (fewest existing scenes).
        Returns count of keys stored into (text-API preserves old contract)."""
        if not self.use_language:
            raise RuntimeError("learn(text) requires use_language=True")
        scene = self._parser.parse_scene(text, self.embedder)
        return len(self.learn_scene(scene))

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
