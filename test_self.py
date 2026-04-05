"""Teach the brain about itself.

Self-knowledge is not incidental — it shapes what the brain is.
External facts fill the brain; reflexive facts change what the
brain can say about itself.

Requires run_brain.py running on :7700.
"""

import time
from brain_client import BrainClient


SELF_FACTS = [
    "I am a brain that learns from experience.",
    "I store knowledge as geometric patterns called scenes.",
    "I think continuously even when no one is talking to me.",
    "I become more confident about things I have been asked many times.",
    "I forget things I have only encountered once.",
]

QUESTIONS = [
    "What are you?",
    "How do you learn?",
    "What happens when no one is talking to you?",
    "What do you forget?",
    "When are you confident?",
]


def main():
    client = BrainClient()
    print("Reaching brain...")
    print(f"  stats: {client.stats()}\n")

    print("--- Teaching self-knowledge ---")
    for fact in SELF_FACTS:
        r = client.learn(fact)
        print(f"  stored in {r['stored_in']} experts: {fact}")

    time.sleep(0.5)

    print("\n--- Asking reflexive questions ---")
    for q in QUESTIONS:
        r = client.query(q)
        print(f"\n  Q: {q}")
        print(f"  A: {r['answer']!r}")
        print(f"     conf={r['confidence']:.2f} "
              f"cycles={r['cycles']} "
              f"margin={r['margin']:.3f} "
              f"reinf={r['reinforced']} "
              f"sources={r['sources']}")

    print("\n--- Current self-knowledge state ---")
    state = client.state()
    print(f"Thinking about: {state['thinking_about']}")
    print("Top warm:")
    for node in state["top_warm"]:
        print(f"  {node['lemma']:15s} act={node['activation']:.3f} "
              f"conf={node['confidence']:.2f} scenes={node['scenes']}")


if __name__ == "__main__":
    main()
