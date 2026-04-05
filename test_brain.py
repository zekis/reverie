"""Phase 1 smoke test.

Start brain. Store 5 facts. Query 3 questions. Print snapshot.
Verify daydream_state shows relevant concepts warm. Under 30s.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from brain import Brain


FACTS = [
    "The dog barked at the postman.",
    "The cat slept on the sofa.",
    "The postman delivered a parcel to the house.",
    "The parcel contained a book about birds.",
    "Birds fly south in winter.",
]

QUESTIONS = [
    "What did the dog bark at?",
    "Where did the cat sleep?",
    "What did the parcel contain?",
]


def main():
    t0 = time.time()
    print("Booting brain...")
    brain = Brain()
    print(f"Brain online ({time.time()-t0:.1f}s)")

    # Learn
    print("\n--- Learning ---")
    brain.learn_bulk(FACTS)
    print(f"Stats: {brain.stats()}")

    # Let daydream run briefly
    time.sleep(0.5)

    # Query
    print("\n--- Querying ---")
    for q in QUESTIONS:
        t1 = time.time()
        r = brain.query(q)
        dt = (time.time() - t1) * 1000
        _assert_result_contract(r)
        print(f"  Q: {q}")
        print(f"  A: {r['answer']}  (conf={r['confidence']:.3f} "
              f"cycles={r['cycles']} margin={r['margin']:.3f} "
              f"reinf={r['reinforced']} sources={r['sources']} {dt:.0f}ms)")

    # Immediate snapshot — activation from queries
    print("\n--- Snapshot (immediately after queries) ---")
    _print_snapshot(brain)

    # Build edge strength via repeated queries on the same topic.
    # Five reps pushes dog<->bark edges to ~0.41, near self-sustain.
    print("\n--- Reinforcing dog/bark cluster (5 reps) ---")
    for _ in range(5):
        r = brain.query("What did the dog bark at?")
        _assert_result_contract(r)
    _print_edges(brain, ["dog", "bark"])

    # Immediate snapshot after reinforcement
    print("\n--- Snapshot (after reinforcement) ---")
    _print_snapshot(brain)

    # Let daydream traversal run — reinforced cluster should sustain
    time.sleep(1.0)
    print("\n--- Snapshot (after 1s of daydream) ---")
    _print_snapshot(brain)

    print("\n--- Daydream state ---")
    print(f"  active: {brain.daydream_state()}")

    print(f"\nTotal: {time.time()-t0:.1f}s")


def _assert_result_contract(r: dict):
    """Contract: every query result must carry these fields. If any is
    dropped, interaction learning silently degrades — assert loudly."""
    required = ("answer", "confidence", "margin", "reinforced",
                "sources", "cycles")
    for field in required:
        assert field in r, f"oscillator result missing '{field}' field: {r}"


def _print_edges(brain, lemmas):
    for lem in lemmas:
        if lem in brain.loader.warm:
            edges = sorted(brain.loader.warm[lem].edge_weights.items(),
                           key=lambda x: -x[1])[:5]
            print(f"  {lem}: {[(k, round(v, 3)) for k, v in edges]}")


def _print_snapshot(brain):
    snap = brain.snapshot()
    if not snap:
        print("  (empty)")
        return
    for lemma, info in sorted(snap.items(),
                               key=lambda x: -x[1]["activation"])[:10]:
        print(f"  {lemma:15s} act={info['activation']:.3f} "
              f"conf={info['confidence']:.2f} scenes={info['scenes']} "
              f"top=[{info['top_scene']}]")


if __name__ == "__main__":
    main()
