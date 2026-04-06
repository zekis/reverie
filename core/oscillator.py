"""The heartbeat — always running.

Two modes, one rhythm:
  Query:    keys → fire warm experts → reconcile → blend context → cycle
  Daydream: traverse warm edges → faintly activate → cool all → snapshot

Expert contributions are modulated by a continuous sine wave. Each
expert has a natural phase (derived from its lemma hash). At any point
in the wave, some experts are near peak (fully contributing) and others
near trough (mostly suppressed). This is lateral inhibition — it stops
the loudest signal from dominating every cycle and gives quieter
experts their turn to be heard.
"""

import hashlib
import math
import threading
import time
import numpy as np

from config import (CYCLE_MS, MAX_CYCLES, CONFIDENCE_THRESHOLD,
                    DAYDREAM_TRAVERSAL_STRENGTH, DAYDREAM_MIN_ACTIVATION,
                    DAYDREAM_EDGE_FANOUT, PHASE_INCREMENT,
                    REPLAY_BUFFER_SIZE, REPLAY_DECAY,
                    REPLAY_DROP_BELOW, REPLAY_EDGE_LR)
from learning.feedback import Feedback
from core.reconciler import reconcile as default_reconcile


def _natural_phase(lemma: str) -> float:
    """Deterministic phase offset for an expert, derived from its lemma.
    Distributed across [0, 2π) so that at any wave position roughly
    half the experts are above average and half below."""
    h = int(hashlib.md5(lemma.encode()).hexdigest()[:8], 16)
    return (h / 0xFFFFFFFF) * 2 * math.pi


def _wave_weight(phase: float, lemma: str) -> float:
    """Continuous wave modulation: returns 0..1. At the expert's natural
    peak → 1.0, at its trough → 0.0, smoothly varying in between."""
    return 0.5 + 0.5 * math.sin(phase + _natural_phase(lemma))


class Oscillator:
    def __init__(self, loader, reconcile_fn=None):
        self.loader = loader
        self.reconcile = reconcile_fn or default_reconcile
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._query_lock = threading.Lock()
        self._cycle_count = 0
        self._phase = 0.0  # continuous wave phase (radians)
        # Replay buffer: recent interactions (sources + residual strength)
        # that get replayed offline during quiet daydream ticks.
        self._replay_buffer: list[dict] = []

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._daydream_loop,
                                        daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0):
        """Signal shutdown and wait for current cycle to finish."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    # ---- Daydream: runs between queries ----

    def _daydream_loop(self):
        period = CYCLE_MS / 1000.0
        while not self._stop.is_set():
            t0 = time.perf_counter()
            # Only daydream if nothing else is holding the lock
            if self._query_lock.acquire(blocking=False):
                try:
                    self._daydream_tick()
                finally:
                    self._query_lock.release()
            self._cycle_count += 1
            elapsed = time.perf_counter() - t0
            remaining = period - elapsed
            if remaining > 0:
                time.sleep(remaining)

    def _daydream_tick(self):
        """Cool all, then traverse edges from survivors to re-warm
        neighbours. Wave-modulated: each expert's spread strength is
        scaled by its position on the continuous sine wave, creating
        a rolling activation pattern instead of uniform spread."""
        warm = self.loader.warm
        if not warm:
            return
        self._phase += PHASE_INCREMENT
        # 1. Cool everything
        for exp in warm.values():
            exp.cool()
        # 2. Find survivors — nodes warm enough to still spread activation
        survivors = [(lemma, exp) for lemma, exp in warm.items()
                     if exp.activation > DAYDREAM_MIN_ACTIVATION]
        # 3. Traverse top edges from each survivor, wave-modulated
        for lemma, exp in survivors:
            wave_w = _wave_weight(self._phase, lemma)
            edges = sorted(exp.edge_weights.items(),
                           key=lambda x: -x[1])[:DAYDREAM_EDGE_FANOUT]
            for neighbour_lemma, edge_w in edges:
                strength = (exp.activation * edge_w
                            * DAYDREAM_TRAVERSAL_STRENGTH * wave_w)
                if strength < 1e-4:
                    continue
                neighbour = warm.get(neighbour_lemma)
                if neighbour is not None:
                    neighbour.fire(strength)

        # 4. Replay consolidation — if the graph has gone quiet, replay
        # the most recent interaction to keep recent experience alive.
        if not survivors and self._replay_buffer:
            self._replay_once()

    def _replay_once(self):
        """Re-fire sources of the most recent interaction and re-strengthen
        their pairwise edges. Decays the interaction's strength each time;
        drops it when below threshold."""
        interaction = self._replay_buffer[-1]
        sources = interaction["sources"]
        strength = interaction["strength"]
        warm = self.loader.warm
        # Re-fire the sources
        for lem in sources:
            exp = warm.get(lem)
            if exp is not None:
                exp.fire(strength)
        # Re-strengthen the same pair edges, scaled by current strength
        lr = REPLAY_EDGE_LR * strength
        for i, lem_a in enumerate(sources):
            if lem_a not in warm:
                continue
            a = warm[lem_a]
            for lem_b in sources[i + 1:]:
                if lem_b not in warm:
                    continue
                b = warm[lem_b]
                w_ab = a.edge_weights.get(lem_b, 0.1)
                a.edge_weights[lem_b] = w_ab + lr * (1.0 - w_ab)
                w_ba = b.edge_weights.get(lem_a, 0.1)
                b.edge_weights[lem_a] = w_ba + lr * (1.0 - w_ba)
        # Decay and drop if exhausted
        interaction["strength"] *= REPLAY_DECAY
        interaction["replays"] += 1
        if interaction["strength"] < REPLAY_DROP_BELOW:
            self._replay_buffer.pop()

    def record_for_replay(self, sources: list[str]):
        """Push a completed query onto the replay buffer at full strength."""
        if not sources or len(sources) < 2:
            return  # single-source interactions have no pair to strengthen
        self._replay_buffer.append({
            "sources": list(sources),
            "strength": 1.0,
            "replays": 0,
            "timestamp": time.time(),
        })
        # Cap buffer at REPLAY_BUFFER_SIZE (oldest drops)
        while len(self._replay_buffer) > REPLAY_BUFFER_SIZE:
            self._replay_buffer.pop(0)

    # ---- Query mode ----

    def query(self, gap: dict, keys: list[str]) -> dict:
        """Run an oscillating query to completion. Blocks daydream.

        gap: {query_vec: np.ndarray, subject_vec: np.ndarray,
              role: str, ...}  (any other fields ignored)
        keys: list of expert lemmas to seed cycle 0
        """
        with self._query_lock:
            return self._query_cycles(gap, keys)

    def _query_cycles(self, gap: dict, keys: list[str]) -> dict:
        fb = Feedback(gap["query_vec"])
        best_answer, best_conf = None, 0.0
        best_answer_vec: bytes | None = None
        best_sources: list[str] = []
        best_margin = 0.0
        best_reinforced = 0
        prev_answer = None
        # Reset wave phase per query — each query sweeps the same arc,
        # ensuring reproducible expert weighting across cycles.
        query_phase = 0.0

        for cycle in range(MAX_CYCLES):
            query_phase += PHASE_INCREMENT

            # Cycle 0: use extracted keys. Later: nearest-neighbour from context.
            if cycle == 0:
                lemmas = keys
            else:
                nearest = self.loader.get_nearest(
                    fb.context.tobytes(), k=8, include_cold=True)
                lemmas = [n["lemma"] for n in nearest if n["sim"] > 0.3]

            responses = []
            alphas: dict[str, float] = {}
            for lem in lemmas:
                expert = self.loader.warm_expert(lem) or self.loader.get_or_create(lem)
                if expert.scene_matrix is None:
                    continue
                # Wave-modulated firing: expert contribution is scaled
                # by its position on the continuous sine wave. Near peak
                # → full contribution; near trough → mostly suppressed.
                wave_w = _wave_weight(query_phase, lem)
                expert.fire(strength=0.3 * wave_w)
                r = expert.query(fb.context.tobytes(), gap["role"],
                                 gap["subject_vec"].tobytes())
                if r.get("answer") is not None:
                    # Modulate the raw similarity score by wave weight
                    # so the reconciler naturally favours currently-
                    # peaking experts over currently-suppressed ones.
                    r["score"] *= wave_w
                    responses.append(r)
                    alphas[lem] = expert.local_alpha * wave_w
                    # Inhibition: weak-scoring experts get dampened
                    if r.get("score", 0) < 0.3:
                        expert.activation *= 0.5

            res = self.reconcile(responses, alphas,
                                 query_vec=gap["query_vec"])
            answer = res.get("answer")
            margin = res.get("margin", 0.0)
            conf = res.get("confidence", 0.0)
            reinf = res.get("reinforced", 0)

            if answer is not None and res.get("answer_vec") is not None:
                av = np.frombuffer(res["answer_vec"], dtype=np.float32)
                # Only blend when dims align — cross-dim adapters
                # (e.g. scene_vec=concat vs slot_vec=component) can't
                # feed back into the same context.
                if len(av) == len(fb.context):
                    fb.blend(av)
            if conf > best_conf:
                best_answer = answer
                best_answer_vec = res.get("answer_vec")
                best_conf = conf
                best_sources = res.get("sources", [])
                best_margin = margin
                best_reinforced = reinf
            # Exit conditions
            if margin > CONFIDENCE_THRESHOLD and reinf >= 2:
                self.record_for_replay(best_sources)
                return {"answer": best_answer, "answer_vec": best_answer_vec,
                        "confidence": best_conf,
                        "cycles": cycle + 1, "margin": best_margin,
                        "sources": best_sources,
                        "reinforced": best_reinforced}
            if answer == prev_answer and answer and margin > CONFIDENCE_THRESHOLD * 0.5:
                self.record_for_replay(best_sources)
                return {"answer": best_answer, "answer_vec": best_answer_vec,
                        "confidence": best_conf,
                        "cycles": cycle + 1, "margin": best_margin,
                        "sources": best_sources,
                        "reinforced": best_reinforced}
            prev_answer = answer

        if best_margin > CONFIDENCE_THRESHOLD and best_reinforced >= 2:
            self.record_for_replay(best_sources)
        return {"answer": best_answer, "answer_vec": best_answer_vec,
                "confidence": best_conf,
                "cycles": MAX_CYCLES, "margin": best_margin,
                "sources": best_sources, "reinforced": best_reinforced}
