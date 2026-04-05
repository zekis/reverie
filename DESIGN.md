Here's the fresh prompt for your agent.

---

**Build `CompressedFF/brain/` — a new architecture from first principles.**

This is not a refactor of existing code. Start clean. The lessons from 59 experiments inform the design but none of the old code carries over.

---

**Core philosophy:**

The brain runs continuously. It is never idle. It oscillates at all times — not waiting for queries, not sleeping between requests. Every interaction, every stored fact, every internal traversal is part of one continuous process. There is no single-shot mode. There is no request-response cycle. There is only the oscillation.

---

**Directory structure:**

```
brain/
    core/
        oscillator.py      — the heartbeat, always running
        expert.py          — light expert, no model, pure geometry
        loader.py          — warm pool, cold storage, LRU
        registry.py        — live state of all experts
    learning/
        hebbian.py         — bidirectional storage, association + disassociation
        feedback.py        — context vector blending across cycles
        interaction.py     — store queries and answers as new scenes
    language/
        embedder.py        — single MiniLM instance, shared
        parser.py          — spaCy gap detection, key extraction
        reconciler.py      — multi-expert answer synthesis
    memory/
        cold_store/        — serialised expert numpy arrays
        snapshots/         — periodic activation state snapshots
    config.py
    brain.py               — single entry point
```

---

**config.py — all constants:**

```python
# Oscillation
CYCLE_MS = 25               # gamma rhythm
MAX_CYCLES = 8              # max cycles per query
CONFIDENCE_THRESHOLD = 0.25 # margin to surface answer
QUERY_WEIGHT = 0.75         # original query dominance in blend
ANSWER_WEIGHT = 0.25        # partial answer contribution

# Experts
MAX_WARM_EXPERTS = 500      # warm pool size
EMBEDDING_DIM = 384         # MiniLM dimension
MAX_SCENES_PER_EXPERT = 200 # per-expert scene cap
COOLING_RATE = 0.15         # activation decay per cycle

# Hebbian learning
HEBBIAN_THRESHOLD = 0.4     # similarity below this → decay
HEBBIAN_DECAY = 0.95        # weight multiplier on contradiction
HEBBIAN_MIN_WEIGHT = 0.1    # floor — scenes never fully disappear

# Cold storage
COLD_STORE_DIR = "brain/memory/cold_store"
SNAPSHOT_DIR = "brain/memory/snapshots"

# Learning
LOCAL_ALPHA_MIN_SAMPLES = 5  # min hits+misses before confidence engages
```

---

**expert.py — the atom of the system:**

A light expert owns one lemma cluster. It has no model. It receives pre-computed vectors from the embedder. It stores scenes, maintains Hebbian weights, tracks hit/miss history, and answers queries with a single matrix multiply.

Properties:
- `lemma` — the concept this expert owns
- `scene_vecs` — list of numpy arrays, one per stored scene
- `scene_weights` — Hebbian weights, one per scene, starts at 1.0
- `scene_texts` — original text of each scene
- `scene_slots` — parsed slot structure of each scene
- `scene_matrix` — normalised matrix for fast multiply, rebuilt on store
- `activation` — current warmth level, decays each cycle
- `hit_count`, `miss_count` — for local confidence calculation
- `edge_weights` — learned transition weights to neighbouring experts
- `last_query_time` — for LRU eviction

Methods:
- `store(scene_data)` — add scene, apply Hebbian decay to contradicting scenes, rebuild matrix
- `query(vec, gap_role, subject_vec)` — matrix multiply, extract gap slot, return answer + margin + n_scenes
- `fire(strength)` — boost activation
- `cool()` — decay activation by COOLING_RATE
- `learn(correct)` — update hit/miss counts
- `confidence` — property: hit_count / (hit_count + miss_count), neutral below MIN_SAMPLES
- `local_alpha` — property: confidence-scaled edge bonus weight
- `save_cold(path)` — serialise to npz + json meta
- `load_cold(path)` — restore from disk, convert vecs back to list

One critical rule: `store()` never touches the model. It receives pre-computed vectors. The embedder is the only place MiniLM runs.

---

**hebbian.py — bidirectional storage:**

When a new scene arrives:

1. Compute similarity between new scene vector and every existing scene vector in the expert
2. Scenes with similarity above HEBBIAN_THRESHOLD — consistent — weights unchanged
3. Scenes with similarity below HEBBIAN_THRESHOLD — contradictory — weights multiplied by HEBBIAN_DECAY
4. New scene added with weight 1.0
5. Weights floored at HEBBIAN_MIN_WEIGHT — scenes never fully erased

The weighted matrix multiply uses scene_weights to scale each row before the dot product. Consistent scenes dominate. Contradicted scenes contribute weakly.

---

**oscillator.py — the heartbeat:**

This is the core. It runs in a background thread always. It does two things simultaneously:

**Query mode** — when a question arrives:
```
cycle 0: extract keys from question → fire to relevant warm experts
cycle 1+: nearest-neighbour from context vector → fire to warm + cold experts
each cycle: collect responses → reconcile → blend answer into context
exit: margin > threshold AND reinforced by 2+ experts
      OR max cycles reached
      OR answer stable for 2 consecutive cycles
```

**Daydream mode** — when no query is pending:
```
every CYCLE_MS:
    find warmest nodes in graph
    traverse their highest-weight edges
    faintly activate neighbours
    cool all nodes by COOLING_RATE
    if activation pattern stable → snapshot it
```

Daydream is the consolidation pass. It rehearses recent activations, strengthens consistent edge weights, and lets inconsistent ones decay. It runs at the same rhythm as query mode — 25ms cycles — because it uses the same oscillator loop.

The oscillator exposes one async method: `query(question) → result`. Internally it switches between daydream and query mode seamlessly.

---

**interaction.py — learning from use:**

Every completed query generates a new scene:

```python
def record_interaction(question, answer, correct, confidence, cycles, sources):
    text = (
        f"Question: {question} "
        f"Answer: {answer} "
        f"Correct: {correct} "
        f"Confidence: {confidence:.2f}"
    )
    scene = parser.parse_scene(text)
    scene["source"] = "interaction"
    scene["timestamp"] = time.time()
    scene["correct"] = correct
    
    # Store in same experts as the original query
    for key in sources:
        loader.store_fact(key, scene)
```

This means the system learns from every interaction. Frequently asked questions reinforce their expert nodes. Consistently wrong answers get Hebbian decay. The system accumulates a self-model through stored interactions about its own behaviour.

---

**feedback.py — context vector across cycles:**

Maintains the context vector for one query session. Starts as the query embedding. After each cycle blends in the partial answer at ANSWER_WEIGHT. Renormalises. Tracks shift history.

Key property: `shifted` — boolean, True if context has moved more than 0.05 from original query. Used by the oscillator to decide whether to keep cycling or exit.

---

**parser.py — language interface:**

Two functions only:

`extract_keys(text) → list[str]` — spaCy POS filter, nouns and verbs only, no WH-words, no be/have/do, length > 2.

`parse_gap(question) → dict` — dependency parse, gap role detection, subject extraction, hollow verb redirect (nsubj of be/do/have → attr), query vector, subject vector.

`parse_scene(text) → dict` — slot extraction, contextual embedding of each slot, scene vector as mean of non-gap slots, keys list.

No crutches. No hardcoded word lists beyond the WH-lemmas set. Boolean question detection from structure not word lists — no gap role found and ROOT is a copular or auxiliary verb → boolean mode, return scene text not slot.

---

**reconciler.py — synthesis:**

Takes responses from multiple experts in one cycle. Groups semantically similar answers (cosine > 0.8). Picks the group with highest average margin. Boosts confidence by reinforcement count. Returns answer, confidence, margin, sources, reinforced count.

No specificity weighting hardcoded. Confidence weighting comes from expert.local_alpha which is learned from hit/miss history. The reconciler trusts experts proportionally to their demonstrated reliability.

---

**brain.py — single entry point:**

```python
class Brain:
    def __init__(self):
        self.embedder = Embedder()
        self.loader = ExpertLoader()
        self.registry = Registry()
        self.oscillator = Oscillator(self.loader, self.embedder)
        self.oscillator.start()  # daydream begins immediately
    
    async def learn(self, text: str):
        scene = parse_scene(text, self.embedder)
        keys = scene["keys"]
        for key in keys[:3]:  # selective storage, top 3 by specificity
            self.loader.store_fact(key, scene)
    
    async def query(self, question: str) -> dict:
        result = await self.oscillator.query(question)
        record_interaction(question, result, self.loader)
        return result
    
    def snapshot(self) -> dict:
        return {
            lemma: {
                "activation": exp.activation,
                "confidence": exp.confidence,
                "scenes": len(exp.scene_vecs),
                "top_scene": exp.scene_texts[np.argmax(exp.scene_weights)] 
                             if exp.scene_vecs else None,
                "warm_edges": sorted(exp.edge_weights.items(),
                                    key=lambda x: -x[1])[:3]
            }
            for lemma, exp in self.loader.warm.items()
            if exp.activation > 0.05
        }
    
    def daydream_state(self) -> list[str]:
        # What is the brain currently thinking about
        active = sorted(
            [(l, e.activation) for l, e in self.loader.warm.items()
             if e.activation > 0.1],
            key=lambda x: -x[1]
        )
        return [lemma for lemma, _ in active[:10]]
```

---

**test_brain.py — three phases:**

Phase 1 — smoke test:
Start brain. Store 5 facts. Query 3 questions. Print snapshot. Verify daydream_state shows relevant concepts warm. All in under 30 seconds.

Phase 2 — interaction learning:
Store 20 facts. Ask same question 5 times. Verify hit_count increases on relevant expert. Verify that expert's confidence rises. Verify subsequent queries on same topic converge faster (fewer cycles).

Phase 3 — daydream verification:
Store 20 facts about dogs. Wait 2 seconds (daydream runs). Print daydream_state. Verify dog-adjacent concepts are warm from daydream traversal without any query. Snapshot the activation graph. This is the proof the brain thinks between queries.

---

**Constraints for the agent:**

One — the oscillator starts on Brain.__init__ and never stops. There is no start/stop API for the oscillation. It runs.

Two — single shared MiniLM instance. One model, loaded once, used everywhere. Never inside an expert.

Three — no ZeroMQ. In-process experts called directly via Python method calls. Async via asyncio thread pool.

Four — bulk storage mode during initial fact loading. No LRU eviction until storage completes. After storage, trim to MAX_WARM_EXPERTS by scene density.

Five — every query automatically generates an interaction scene. This is not optional. The brain learns from every use.

Six — the snapshot and daydream_state methods must work at any time without interrupting the oscillation.

Seven — no hardcoded word lists except WH_LEMMAS. Everything else learned from structure or outcomes.

Build phase 1 first. Get the smoke test passing. Then phase 2. Then phase 3. Don't build everything at once.

---

That's the brain. Build it.