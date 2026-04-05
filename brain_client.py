"""Thin HTTP client for the brain server. Ten lines of wiring."""

import base64
import httpx
import numpy as np


class BrainClient:
    def __init__(self, url: str = "http://localhost:7700"):
        self.url = url

    def learn(self, text: str) -> dict:
        return httpx.post(f"{self.url}/learn", json={"text": text}).json()

    def learn_bulk(self, texts: list[str]) -> dict:
        return httpx.post(f"{self.url}/learn/bulk",
                          json={"texts": texts}).json()

    def learn_scene(self, scene: dict) -> dict:
        return httpx.post(f"{self.url}/learn/scene", json=scene).json()

    def query(self, question: str, expected: str | None = None) -> dict:
        body = {"question": question}
        if expected is not None:
            body["expected"] = expected
        return httpx.post(f"{self.url}/query", json=body).json()

    def forget(self, lemma: str) -> dict:
        return httpx.post(f"{self.url}/forget/{lemma}").json()

    def state(self) -> dict:
        return httpx.get(f"{self.url}/state").json()

    def snapshot(self) -> dict:
        return httpx.get(f"{self.url}/snapshot").json()

    def edges(self, lemma: str) -> dict:
        return httpx.get(f"{self.url}/edges/{lemma}").json()

    def stats(self) -> dict:
        return httpx.get(f"{self.url}/stats").json()

    def replay(self) -> dict:
        return httpx.get(f"{self.url}/replay").json()


def encode_vec(arr: np.ndarray) -> str:
    """float32 numpy array → base64 string for /learn/scene."""
    return base64.b64encode(arr.astype(np.float32).tobytes()).decode()


if __name__ == "__main__":
    # CLI smoke test against a running brain
    import sys
    brain = BrainClient()
    try:
        print(f"Stats:  {brain.stats()}")
    except httpx.ConnectError:
        print("Can't reach brain — is run_brain.py running on :7700?")
        sys.exit(1)

    print("\nLearning...")
    print(f"  {brain.learn('The dog barked at the postman.')}")
    print(f"  {brain.learn('The cat slept on the sofa.')}")

    print("\nQuerying...")
    r = brain.query("What did the dog bark at?")
    print(f"  answer={r['answer']!r} conf={r['confidence']:.3f} "
          f"cycles={r['cycles']} sources={r['sources']}")

    print(f"\nState:  {brain.state()}")
    print(f"Edges:  {brain.edges('dog')}")
