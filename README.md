# Reverie

A continuously-oscillating brain. Stores knowledge as geometric scenes.
Learns Hebbian edges from co-activation. Consolidates through replay.
No LLM. No external API. Runs on your own machine.

## Core idea

Reverie is an **oscillating associative memory** built from lightweight experts — one per concept (lemma) — that store scenes as slot-vector arrays. Queries fan out to relevant experts in parallel, each answers with a matrix-multiply, and a reconciler picks the winning answer. Between queries the brain **daydreams**: it traverses the edge graph from warm nodes to their neighbours, keeping densely-connected clusters alive without external input. Recent interactions are re-played offline, consolidating edges in the background.

The architecture has one philosophical commitment: **the brain never stops**. There is no idle. There is no request-response cycle. There is only the oscillation, at a 25ms gamma rhythm, running in a background thread from `Brain.__init__` onward.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                     Brain                            │
│  ┌────────────────┐  ┌──────────────────────────┐    │
│  │ Oscillator     │  │ ExpertLoader             │    │
│  │  - 25ms rhythm │◄─┤  - warm pool (≤500)      │    │
│  │  - query cycles│  │  - cold store (unlimited)│    │
│  │  - daydream    │  │  - LRU eviction          │    │
│  │  - replay buf  │  └──────────────────────────┘    │
│  └────────┬───────┘           │                      │
│           │                   │                      │
│  ┌────────▼─────────┐ ┌───────▼─────────────────┐   │
│  │ Embedder         │ │ LightExpert (per lemma) │   │
│  │  (shared MiniLM) │ │  - scene matrix         │   │
│  └──────────────────┘ │  - Hebbian weights      │   │
│  ┌──────────────────┐ │  - edge weights         │   │
│  │ Parser (spaCy)   │ │  - hit/miss stats       │   │
│  └──────────────────┘ └─────────────────────────┘   │
└──────────────────────────────────────────────────────┘
           ▲
           │ HTTP (FastAPI)
     ┌─────┴─────┬──────────┬──────────┐
     │  chat.py  │ trainer  │ telegram │ ...
     └───────────┴──────────┴──────────┘
```

### Experts are the atoms

Each expert owns a lemma (`dog`, `reverie`, `learn`) and holds every scene that mentions it as a top-3 key. A scene is a set of typed slots (`nsubj`, `dobj`, `prep_from`, `ROOT`, ...) each carrying a contextual embedding. Querying an expert is one BLAS call — matrix-multiply the query vector against the weighted scene matrix — followed by gap-role extraction from the top-scoring scene.

### The oscillator runs two modes on one rhythm

- **Query mode**: extract keys → fire experts → reconcile answers → blend partial answer into query context → repeat up to MAX_CYCLES
- **Daydream mode**: cool all warm experts, then traverse highest-weight edges from survivors to re-warm their neighbours — a self-sustaining wave through the graph's densely connected regions

Switches seamlessly between the two. Same 25ms tick.

### Hebbian everything

- **Scene weights** start at 1.0. When a contradictory scene is stored, existing scenes with cosine similarity < 0.4 decay by 0.95 (floored at 0.1 — memory becomes quiet, not erased).
- **Edges** start at 0.1. When two experts co-activate on a confident answer, their bidirectional edge grows: `w += 0.1 × (1-w)`. Crosses the self-sustain threshold (≈0.5) after ~6 co-activations.
- **Expert confidence** tracks `hits / (hits+misses)`, scales `local_alpha` in `[0.5, 1.5]` — reliable experts contribute more to the reconciler.

### Consolidation happens between queries

Successful queries push their source set onto a **replay buffer**. When the daydream finds the graph quiet, it replays the most recent interaction — re-firing sources and strengthening their edges at decaying strength (0.7× per replay). A single confident query plus replay chain produces ~2× the edge growth of one query alone.

### Three learning paths, no interaction-scene contamination

Every query routes to exactly one path:

| Path | Trigger | Effect |
|---|---|---|
| **Rewarded** | `expected=` matches answer, or margin > 0.40 | `learn(True)` on sources; edges strengthen. No new scene. |
| **Corrected** | `expected=` provided AND answer mismatches | `learn(False)` on sources; corrective scene written: `"The answer to X is Y."` |
| **Penalised** | Answer is `None` | `learn(False)` on sources. No scene. |
| **Ignored** | Low-confidence, no `expected` | Nothing. Routine exchange, no prediction error. |

Interaction transcripts are never stored. Episodic (what was said) and semantic (what was learned) memory are separated.

## Quick start

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Terminal 1 — start the brain
python run_brain.py
# Brain online, API listening on http://127.0.0.1:7700

# Terminal 2 — talk to it
python chat.py
```

You can also:

- `python trainer.py lessons/*.yaml` — batch teach + quiz from lesson files
- `python telegram_bot.py` — expose the brain as a Telegram bot (requires `.env`)
- direct HTTP calls from any language — see API below

## API

All endpoints speak JSON over HTTP on port 7700 by default.

| Endpoint | Purpose |
|---|---|
| `POST /learn` | `{"text": "..."}` — teach one fact |
| `POST /learn/bulk` | `{"texts": [...]}` — teach many |
| `POST /learn/scene` | `{"slots":[], "vec":"<b64>", "keys":[...]}` — domain-agnostic, pre-computed vectors |
| `POST /query` | `{"question": "...", "expected": "..."}` — ask, with optional ground truth |
| `POST /forget/{lemma}` | halve scene weights, outgoing+incoming edge weights, and activation |
| `GET /state` | current daydream state + top-5 warm experts |
| `GET /snapshot` | all warm experts with activation/confidence/scene counts |
| `GET /edges/{lemma}` | one expert's edges + hit/miss stats |
| `GET /stats` | loader totals |
| `GET /replay` | replay buffer contents |
| `POST /save` | persist all warm experts to cold storage |
| `POST /debug/keys` | show parser key extraction |
| `POST /debug/scene` | show parsed slots for a text |
| `POST /debug/query` | per-expert responses before reconciliation |
| `POST /debug/match` | top-K scenes in one expert for a query, with scores |

## Lesson format

```yaml
# lessons/my_topic.yaml
topic: my_topic
reps: 3
facts:
  - "A fact."
  - "Another fact."
quiz:
  - q: "A question?"
    expected: "expected answer"
```

Run with `python trainer.py lessons/my_topic.yaml`. Metrics appended to `memory/metrics/runs.csv`.

## Clients

- **`chat.py`** — REPL. Plain text is a query; `:learn`, `:teach`, `:forget`, `:state`, `:edges`, `:debug` as commands.
- **`trainer.py`** — batch teach + quiz with ground truth. Logs pass rates per topic.
- **`telegram_bot.py`** — Telegram bot. Reads token from `.env`.
- **`brain_client.py`** — thin Python HTTP client. Ten methods.

## Project layout

```
reverie/
├── brain.py              — single entry point
├── run_brain.py          — long-lived server process
├── api.py                — FastAPI routes
├── config.py             — all constants
├── core/
│   ├── expert.py         — LightExpert: scene matrix + Hebbian + gap extraction
│   ├── loader.py         — warm pool, cold storage, LRU
│   ├── oscillator.py     — 25ms heartbeat, query cycles, daydream, replay
│   └── registry.py       — read-only view of loader state
├── language/
│   ├── embedder.py       — one shared MiniLM
│   ├── parser.py         — spaCy: extract_keys, parse_gap, parse_scene
│   └── reconciler.py     — multi-expert answer synthesis + info gain filter
├── learning/
│   ├── hebbian.py        — bidirectional scene decay
│   ├── feedback.py       — query context vector blending
│   └── interaction.py    — three-path learning signal
├── lessons/              — YAML lesson files
├── memory/
│   ├── cold_store/       — serialised expert .npz + .meta files
│   ├── metrics/          — runs.csv
│   └── snapshots/        — reserved for activation snapshots
├── chat.py               — REPL client
├── trainer.py            — batch trainer
├── telegram_bot.py       — Telegram bridge
└── brain_client.py       — Python HTTP wrapper
```

## Design notes

### What Reverie is not

- Not an LLM. There is no transformer, no next-token prediction, no attention over a context window.
- Not a knowledge graph. Experts aren't nodes with properties — they're lemma-keyed scene caches.
- Not an inference engine. Answers come from retrieval + slot extraction, not deduction.
- Not cloud. Runs on one machine, one process, one model weight file (MiniLM).

### What Reverie is

- A **memory substrate** that accumulates knowledge through repeated exposure and corrective feedback
- A **parallel activation engine** where many experts fire simultaneously and converge through reconciliation
- A **self-organising graph** where edges strengthen from co-occurrence and weaken from contradiction
- A **continuous process** that maintains state between queries through daydream + replay

### Open problems

- Scene selection ignores the query's verb — `"Who built Reverie?"` can't distinguish `"I am Reverie"` (high similarity, wrong verb) from `"Tierney built Reverie"` (slightly lower similarity, right verb)
- Short-noun expected answers work well in training; multi-word phrases and abstract concepts don't
- Corrective scenes land but can be outranked by longer-reinforced old scenes — recovery requires targeted `/forget` + re-teach cycles
- No mechanism yet for Reverie to introspect its own live state (e.g. `"What are you thinking about right now?"` retrieves stored beliefs about thinking, not the actual `daydream_state`)

## Licence

Not yet decided. Treat as source-available, all rights reserved, until a licence file lands.
