"""Brain — single entry point. Oscillator starts on init, never stops."""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from language.embedder import Embedder
from language import parser
from language.reconciler import reconcile
from core.loader import ExpertLoader
from core.registry import Registry
from core.oscillator import Oscillator
from learning.interaction import record_interaction


class Brain:
    def __init__(self, warm_cold: bool = True):
        self.embedder = Embedder()
        self.loader = ExpertLoader()
        self.registry = Registry(self.loader)
        self.oscillator = Oscillator(
            self.loader, self.embedder, parser, reconcile
        )
        if warm_cold:
            n = self.loader.warm_all_cold()
            if n > 0:
                print(f"[brain] pre-warmed {n} experts from cold storage")
        self.oscillator.start()

    def save(self) -> int:
        """Persist all warm experts to cold storage. Returns count."""
        return self.loader.save_all()

    # ---- Public API ----

    def learn(self, text: str):
        """Store a fact. Top 3 keys by specificity (fewest existing scenes)."""
        scene = parser.parse_scene(text, self.embedder)
        keys = scene["keys"]
        if not keys:
            return 0
        # Specificity: keys whose experts have fewest scenes are most specific
        scored = []
        for k in keys:
            n = len(self.loader.warm[k].scene_vecs) if k in self.loader.warm else 0
            scored.append((k, n))
        scored.sort(key=lambda x: x[1])
        stored = 0
        for key, _ in scored[:3]:
            self.loader.store_fact(key, scene)
            stored += 1
        return stored

    def learn_bulk(self, texts: list[str]):
        """Bulk storage mode — no LRU eviction until done."""
        self.loader.start_bulk()
        try:
            for t in texts:
                self.learn(t)
        finally:
            self.loader.end_bulk()

    def query(self, question: str, expected: str | None = None) -> dict:
        """Run an oscillating query. If expected is provided, the
        interaction uses ground-truth salience; otherwise falls back
        to implicit margin-based signal."""
        result = self.oscillator.query(question)
        record_interaction(question, result, self.loader,
                           parser, self.embedder, expected=expected)
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
