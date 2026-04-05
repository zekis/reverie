"""Phase 2: interaction learning.

Store 20 facts. Ask same question 5 times. Verify:
  - hit_count on source experts increases
  - confidence rises above neutral 0.5
  - edges between co-sources strengthen
  - cycles per query trend down (convergence)

Under 30s.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from brain import Brain


FACTS_20 = [
    "The dog barked at the postman.",
    "The postman delivered a parcel to the house.",
    "The parcel contained a book about birds.",
    "The cat slept on the sofa.",
    "The sofa sits in the living room.",
    "Birds fly south in winter.",
    "Winter brings snow to the mountains.",
    "The mountains are covered in snow.",
    "The child threw a ball in the garden.",
    "The garden has roses and tulips.",
    "Roses bloom in summer.",
    "Summer days are long and warm.",
    "The baker sold bread at the market.",
    "The market opens early in the morning.",
    "The river flows through the valley.",
    "The valley is green in spring.",
    "Spring rain waters the fields.",
    "The farmer grows wheat in the fields.",
    "Wheat makes flour for bread.",
    "The teacher read a story to the class.",
]

QUESTION = "What did the dog bark at?"
REPS = 5


def main():
    t0 = time.time()
    print("Booting brain...")
    brain = Brain()
    print(f"Brain online ({time.time()-t0:.1f}s)")

    print(f"\nLearning {len(FACTS_20)} facts...")
    brain.learn_bulk(FACTS_20)
    print(f"Stats: {brain.stats()}")

    # Baseline — record state before any queries
    print(f"\nBaseline state (before queries):")
    _assert_all_neutral(brain, ["dog", "bark"])

    # Ask same question REPS times, track cycles and state each iteration
    print(f"\nAsking '{QUESTION}' {REPS} times:")
    cycles_history = []
    for i in range(REPS):
        r = brain.query(QUESTION)
        _assert_contract(r)
        cycles_history.append(r["cycles"])
        dog_hits = _hits(brain, "dog")
        dog_conf = _conf(brain, "dog")
        edge = _edge(brain, "dog", "bark")
        print(f"  rep {i+1}: A={r['answer']!r} cycles={r['cycles']} "
              f"margin={r['margin']:.3f} reinf={r['reinforced']} "
              f"| dog.hits={dog_hits} dog.conf={dog_conf:.2f} "
              f"dog->bark={edge:.3f}")

    # Assertions
    print(f"\n--- Assertions ---")

    dog_hits = _hits(brain, "dog")
    bark_hits = _hits(brain, "bark")
    assert dog_hits >= REPS, f"dog.hit_count={dog_hits}, expected >= {REPS}"
    assert bark_hits >= REPS, f"bark.hit_count={bark_hits}, expected >= {REPS}"
    print(f"  [OK] hit_count increased: dog={dog_hits}, bark={bark_hits}")

    dog_conf = _conf(brain, "dog")
    bark_conf = _conf(brain, "bark")
    assert dog_conf > 0.5, f"dog.confidence={dog_conf:.3f}, expected > 0.5"
    assert bark_conf > 0.5, f"bark.confidence={bark_conf:.3f}, expected > 0.5"
    print(f"  [OK] confidence rose: dog={dog_conf:.2f}, bark={bark_conf:.2f}")

    edge_db = _edge(brain, "dog", "bark")
    edge_bd = _edge(brain, "bark", "dog")
    assert edge_db > 0.1, f"dog->bark edge={edge_db:.3f}, expected > 0.1"
    assert edge_bd > 0.1, f"bark->dog edge={edge_bd:.3f}, expected > 0.1"
    assert abs(edge_db - edge_bd) < 1e-6, "edges should be bidirectionally equal"
    print(f"  [OK] bidirectional edges strengthened: "
          f"dog<->bark={edge_db:.3f}")

    # Convergence — last cycles <= first cycles (monotone-ish)
    first_half = sum(cycles_history[:REPS // 2]) / (REPS // 2)
    second_half = sum(cycles_history[REPS // 2:]) / (REPS - REPS // 2)
    assert second_half <= first_half, (
        f"cycles not converging: first_half_avg={first_half:.1f}, "
        f"second_half_avg={second_half:.1f}")
    print(f"  [OK] cycles converging: "
          f"first_half_avg={first_half:.2f} >= "
          f"second_half_avg={second_half:.2f}")

    brain.stop()
    print(f"\nPhase 2 passed ({time.time()-t0:.1f}s)")


# ---- helpers ----

def _assert_contract(r: dict):
    for f in ("answer", "confidence", "margin",
              "reinforced", "sources", "cycles"):
        assert f in r, f"result missing '{f}': {r}"


def _assert_all_neutral(brain, lemmas):
    for lem in lemmas:
        if lem in brain.loader.warm:
            exp = brain.loader.warm[lem]
            assert exp.hit_count == 0 and exp.miss_count == 0, (
                f"{lem} not neutral: hits={exp.hit_count} "
                f"misses={exp.miss_count}")
            print(f"  {lem}: hits=0 misses=0 confidence={exp.confidence:.2f}")


def _hits(brain, lemma):
    return (brain.loader.warm[lemma].hit_count
            if lemma in brain.loader.warm else 0)


def _conf(brain, lemma):
    return (brain.loader.warm[lemma].confidence
            if lemma in brain.loader.warm else 0.5)


def _edge(brain, a, b):
    if a not in brain.loader.warm:
        return 0.0
    return brain.loader.warm[a].edge_weights.get(b, 0.0)


if __name__ == "__main__":
    main()
