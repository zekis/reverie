"""Give the brain an identity. Reinforce it.

Generic self-facts route through generic experts ('thing', 'become').
A specific name — Reverie — creates a dense self-model node. Repeated
identity queries bind brain<->Reverie above the competing edges like
dog<->bark that dominate from earlier tests.

Requires run_brain.py running on :7700.
"""

import time
from brain_client import BrainClient


IDENTITY_FACTS = [
    # First-person anchors — route "What are you?" through the I/be expert
    "I am Reverie.",
    "I am a brain that learns from experience.",
    "I think continuously using oscillating queries.",
    # Descriptive facts — single experts, contextual
    "This brain is called Reverie.",
    "Reverie stores knowledge as slot vectors in expert nodes.",
    "Reverie was built by following failure modes.",
    # Bridging facts — single-clause, no noise words, reverie and learn
    # as the only content words. Both become top-3 keys. Expert scenes
    # sit close enough in embedding space that any question containing
    # either word retrieves both.
    "Reverie learns.",
    "Learning defines Reverie.",
    "Reverie is learning.",
    # prep_from slot for advmod-gap queries like "How does Reverie learn?"
    "Reverie learns from experience.",
    "Reverie thinks from its memory.",
    # prep_by slot for passive-voice queries
    "Reverie was built by following failures.",
]


def main():
    client = BrainClient()
    print("Reaching brain...")
    print(f"  stats: {client.stats()}\n")

    print("--- Teaching identity ---")
    for fact in IDENTITY_FACTS:
        r = client.learn(fact)
        print(f"  stored in {r['stored_in']} experts: {fact}")

    time.sleep(0.3)

    # Questions with explicit ground-truth answers. Every binding query
    # produces either a Rewarded (correct) or Corrected (wrong + new
    # scene written) outcome — real hits/misses, not margin guesses.
    bindings = [
        ("What is Reverie learning?",   "experience"),
        ("What does Reverie store?",    "scenes"),
        ("What defines Reverie?",       "learning"),
        ("What is Reverie made of?",    "experience"),
    ]
    print(f"\n--- Binding 'reverie' <-> 'learn' "
          f"({len(bindings)*3} queries, with ground truth) ---")
    print("  (** = reinf>=2 edge strengthen; ok = matched expected)")
    for i, (q, expected) in enumerate(bindings * 3):
        r = client.query(q, expected=expected)
        reinf = r["reinforced"]
        ans = r["answer"]
        # Same matching rule the brain uses (substring)
        hit = ans and (expected.lower() in ans.lower()
                       or ans.lower() in expected.lower())
        outcome = "ok" if hit else "miss"
        marker = "  **" if reinf >= 2 else "    "
        edge = _edge_weight(client, "reverie", "learn")
        print(f"{marker}{i+1:2d}: [{outcome:4s}] "
              f"reverie<->learn={edge:.3f} "
              f"reinf={reinf} sources={r['sources']} "
              f"A={ans!r} :: {q} (exp={expected!r})")

    print("\n--- Identity questions (post-reinforcement) ---")
    for q in [
        "What is Reverie?",
        "What are you?",
        "What is this brain called?",
        "What does Reverie do?",
        "How does Reverie learn?",
    ]:
        r = client.query(q)
        print(f"\n  Q: {q}")
        print(f"  A: {r['answer']!r}")
        print(f"     conf={r['confidence']:.2f} "
              f"cycles={r['cycles']} "
              f"margin={r['margin']:.3f} "
              f"sources={r['sources']}")

    # Inspect the edges that now define Reverie
    print("\n--- Identity edges ---")
    for lem in ("brain", "reverie"):
        try:
            e = client.edges(lem)
            print(f"\n  {lem}: scenes={e['scenes']} hits={e['hits']} "
                  f"misses={e['misses']} conf={e['confidence']:.2f}")
            for edge in e["edges"][:6]:
                print(f"    -> {edge['to']:20s} w={edge['weight']}")
        except Exception as err:
            print(f"  {lem}: {err}")

    print("\n--- Current daydream ---")
    state = client.state()
    print(f"Thinking about: {state['thinking_about']}")

    # Introspective questions — never explicitly taught the answers.
    # These route through whatever the brain has stored about its own
    # mechanisms (think, know, confident, forget). The answers come
    # from the brain's *model* of itself, not its live runtime state.
    print("\n--- Introspection (untrained questions) ---")
    for q in [
        "What are you thinking about right now?",
        "What do you know most confidently?",
        "What have you forgotten?",
    ]:
        r = client.query(q)
        print(f"\n  Q: {q}")
        print(f"  A: {r['answer']!r}")
        print(f"     conf={r['confidence']:.2f} "
              f"cycles={r['cycles']} "
              f"margin={r['margin']:.3f} "
              f"sources={r['sources']}")


def _edge_weight(client, a: str, b: str) -> float:
    try:
        e = client.edges(a)
        for edge in e["edges"]:
            if edge["to"] == b:
                return edge["weight"]
    except Exception:
        pass
    return 0.0


if __name__ == "__main__":
    main()
