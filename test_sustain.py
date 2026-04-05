"""Self-sustain test via the HTTP API.

Requires run_brain.py running on :7700. Six reinforced queries push
the dog<->bark edge past the ~0.5 self-sustain threshold. After 10s
of pure daydream with no new queries, the cluster should still be
active.
"""

import time
from brain_client import BrainClient


def main():
    client = BrainClient()
    print("Reaching brain...")
    print(f"  stats: {client.stats()}")

    print("\nSeeding fact...")
    client.learn("The dog barked at the postman.")

    print("\nReinforcing dog/bark (6 queries):")
    for i in range(6):
        r = client.query("What did the dog bark at?")
        print(f"  {i+1}: answer={r['answer']!r} "
              f"conf={r['confidence']:.2f} "
              f"reinf={r['reinforced']} "
              f"sources={r['sources']}")

    edge = client.edges("dog")
    bark_w = next((e["weight"] for e in edge["edges"] if e["to"] == "bark"),
                  None)
    print(f"\ndog->bark edge weight: {bark_w}")
    print(f"dog activation right now: {edge['activation']:.3f}")

    print("\nWaiting 10s (no queries, pure daydream)...")
    time.sleep(10)

    state = client.state()
    print(f"\nAfter 10s daydream:")
    print(f"  thinking_about: {state['thinking_about']}")
    print(f"  top_warm:")
    for node in state["top_warm"]:
        print(f"    {node['lemma']:15s} act={node['activation']:.3f} "
              f"conf={node['confidence']:.2f} scenes={node['scenes']}")

    if "dog" in state["thinking_about"] and "bark" in state["thinking_about"]:
        print("\n[OK] self-sustain verified — cluster holds without queries")
    else:
        print("\n[INFO] cluster did not sustain — edge weight may not have "
              "crossed threshold")


if __name__ == "__main__":
    main()
