"""Type-agnostic primitive smoke test.

Exercises learn_scene / query_scene without the language adapter.
Uses synthetic low-dim vectors (sensorimotor style) to verify the
refactored core works independent of MiniLM / spaCy.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from brain import Brain


DIM = 10  # tiny sensorimotor-style vectors


def make_scene(sensor_vec, action_vec, keys, label=""):
    """Build a scene dict the way an adapter would."""
    scene_vec = np.concatenate([sensor_vec, action_vec]).astype(np.float32)
    return {
        "source_text": label,
        "slots": [
            {"role": "sensor", "text": f"sensor={label}",
             "vec": sensor_vec.astype(np.float32).tobytes(),
             "is_gap": False},
            {"role": "action", "text": f"action={label}",
             "vec": action_vec.astype(np.float32).tobytes(),
             "is_gap": False},
        ],
        "vec": scene_vec.tobytes(),
        "keys": keys,
    }


def make_gap(sensor_vec, keys):
    """Build a query gap with an action-slot gap."""
    # The scene_vec we search against was sensor+action concatenated.
    # The query half we know is sensor; pad the action half with zeros.
    query_vec = np.concatenate([
        sensor_vec, np.zeros(DIM, dtype=np.float32)
    ]).astype(np.float32)
    return {
        "query_vec": query_vec,
        "subject_vec": np.zeros(2 * DIM, dtype=np.float32),  # no echo filter
        "role": "action",
    }, keys


def main():
    print("Booting language-free brain...")
    brain = Brain(warm_cold=False, use_language=False)
    print(f"  stats: {brain.stats()}")

    # Three synthetic sensor→action associations, teach each one twice
    # to build up the expert pool
    rng = np.random.default_rng(42)
    sensors = {
        "food": rng.normal(size=DIM).astype(np.float32),
        "wall": rng.normal(size=DIM).astype(np.float32),
        "nest": rng.normal(size=DIM).astype(np.float32),
    }
    actions = {
        "grab": rng.normal(size=DIM).astype(np.float32),
        "turn": rng.normal(size=DIM).astype(np.float32),
        "drop": rng.normal(size=DIM).astype(np.float32),
    }
    pairs = [("food", "grab"), ("wall", "turn"), ("nest", "drop")]

    print("\n--- Learning 6 sensorimotor scenes (3 pairs x 2 reps) ---")
    for s_name, a_name in pairs:
        for rep in range(2):
            # tiny jitter per rep
            s = sensors[s_name] + rng.normal(scale=0.05, size=DIM).astype(np.float32)
            a = actions[a_name] + rng.normal(scale=0.05, size=DIM).astype(np.float32)
            scene = make_scene(s, a,
                               keys=["sm_experience", s_name],
                               label=f"{s_name}->{a_name}#{rep}")
            handles = brain.learn_scene(scene, top_k=None)
            assert len(handles) == 2, f"expected 2 keys stored, got {len(handles)}"
    print(f"  stats: {brain.stats()}")

    # Query each sensor and check the returned action vector is closest
    # to the corresponding stored action.
    print("\n--- Querying ---")
    action_matrix = np.stack([actions["grab"], actions["turn"], actions["drop"]])
    action_names = ["grab", "turn", "drop"]

    correct = 0
    for s_name, expected_action in pairs:
        gap, keys = make_gap(sensors[s_name], keys=["sm_experience", s_name])
        result = brain.query_scene(gap, keys)

        # result["answer_vec"] should be the action slot's vector.
        if result.get("answer_vec") is None:
            print(f"  sensor={s_name}  NO ANSWER  "
                  f"(cycles={result['cycles']} sources={result['sources']})")
            continue
        av = np.frombuffer(result["answer_vec"], dtype=np.float32)
        # The answer vec lives in the sensorimotor space. Find nearest
        # canonical action by cosine. (The stored slot vec is sensor+jitter
        # OR action+jitter depending on which slot was picked as gap-fill.)
        av_n = av / (np.linalg.norm(av) + 1e-8)
        sims = action_matrix @ av_n[:DIM] if len(av) == DIM else \
               np.array([
                   np.dot(act / (np.linalg.norm(act) + 1e-8),
                          av / (np.linalg.norm(av) + 1e-8))
                   for act in action_matrix
               ])
        nearest = action_names[int(np.argmax(sims))]
        ok = "OK" if nearest == expected_action else "MISS"
        correct += (nearest == expected_action)
        print(f"  sensor={s_name:5s}  expected={expected_action:5s}  "
              f"got={nearest:5s}  [{ok}]  "
              f"margin={result['margin']:.3f} cycles={result['cycles']} "
              f"sources={result['sources']}")

    print(f"\n{correct}/{len(pairs)} correct")

    # Inspect warm pool state
    print("\n--- Warm pool ---")
    for lem, exp in brain.loader.warm.items():
        print(f"  {lem:20s}  scenes={len(exp.scene_vecs)}  "
              f"act={exp.activation:.3f}  edges={list(exp.edge_weights.items())}")

    brain.oscillator.stop()
    print("\nDone.")
    assert correct == len(pairs), "primitive sensorimotor test failed"


if __name__ == "__main__":
    main()
