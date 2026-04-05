"""Ant simulator — motor babbling on Reverie's sensorimotor primitives.

Phase 1: empty scene memory, random actions, store every sensorimotor
episode. No goal, no reward. Pure exploration.

Phase 2: apply outcome-weighted reinforcement. Food-collecting scenes
get their Hebbian weights boosted; wall-hit scenes decay. This is
reward-shaping the already-built sensorimotor map.

No language. The ant never queries yet — this is the substrate-building
phase, analogous to mammalian motor babbling before goal-directed
behaviour emerges.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from brain import Brain


# ---- World constants ----

GRID_SIZE = 15
NEST = (7, 7)
FOOD_SOURCES = [(5, 5), (5, 9), (9, 5), (9, 9),
                (4, 7), (10, 7), (7, 4), (7, 10)]
FOOD_PER_SOURCE = 20

# Headings as (dx, dy). 0=N, 1=E, 2=S, 3=W.
HEADINGS = [(0, -1), (1, 0), (0, 1), (-1, 0)]

ACTIONS = ["move_forward", "turn_left", "turn_right", "pick_up", "drop"]


# ---- Environment ----

class World:
    def __init__(self):
        self.food = {pos: FOOD_PER_SOURCE for pos in FOOD_SOURCES}
        self.food_delivered = 0

    def is_wall(self, pos):
        x, y = pos
        return x < 0 or x >= GRID_SIZE or y < 0 or y >= GRID_SIZE

    def has_food(self, pos):
        return self.food.get(pos, 0) > 0

    def take_food(self, pos):
        if self.food.get(pos, 0) > 0:
            self.food[pos] -= 1
            return True
        return False


class Ant:
    def __init__(self, world):
        self.world = world
        self.pos = NEST
        self.heading = 0
        self.carrying = False

    def front_cell(self):
        dx, dy = HEADINGS[self.heading]
        return (self.pos[0] + dx, self.pos[1] + dy)

    def sense(self):
        ahead = self.front_cell()
        # Normalised nest displacement (not just sign): gives the brain
        # distance-to-nest, not just direction.
        nest_dx = (NEST[0] - self.pos[0]) / GRID_SIZE
        nest_dy = (NEST[1] - self.pos[1]) / GRID_SIZE
        # Heading as 4-dim one-hot: without this two ants at the same
        # cell facing opposite directions have identical sensor state
        # but need opposite actions — the brain can't form heading-
        # dependent policy without this.
        heading_vec = [0.0, 0.0, 0.0, 0.0]
        heading_vec[self.heading] = 1.0
        return {
            "food_ahead": 1 if self.world.has_food(ahead) else 0,
            "at_nest": 1 if self.pos == NEST else 0,
            "carrying": 1 if self.carrying else 0,
            "wall_ahead": 1 if self.world.is_wall(ahead) else 0,
            "heading_n": heading_vec[0],
            "heading_e": heading_vec[1],
            "heading_s": heading_vec[2],
            "heading_w": heading_vec[3],
            "nest_dx": float(nest_dx),
            "nest_dy": float(nest_dy),
        }

    def execute(self, action: str) -> dict:
        """Runs one action, returns outcome dict."""
        outcome = {"food_collected": 0, "food_delivered": 0,
                   "hit_wall": 0, "noop": 0}
        if action == "move_forward":
            ahead = self.front_cell()
            if self.world.is_wall(ahead):
                outcome["hit_wall"] = 1
            else:
                self.pos = ahead
        elif action == "turn_left":
            self.heading = (self.heading - 1) % 4
        elif action == "turn_right":
            self.heading = (self.heading + 1) % 4
        elif action == "pick_up":
            ahead = self.front_cell()
            if not self.carrying and self.world.take_food(ahead):
                self.carrying = True
                outcome["food_collected"] = 1
            else:
                outcome["noop"] = 1
        elif action == "drop":
            if self.carrying and self.pos == NEST:
                self.carrying = False
                self.world.food_delivered += 1
                outcome["food_delivered"] = 1
            else:
                outcome["noop"] = 1
        return outcome


# ---- Encoding: sensorimotor episode → Reverie scene ----

def encode_sensor(s):
    return np.array([s["food_ahead"], s["at_nest"], s["carrying"],
                     s["wall_ahead"],
                     s["heading_n"], s["heading_e"],
                     s["heading_s"], s["heading_w"],
                     s["nest_dx"], s["nest_dy"]], dtype=np.float32)


def encode_action(action):
    v = np.zeros(len(ACTIONS), dtype=np.float32)
    v[ACTIONS.index(action)] = 1.0
    return v


def encode_outcome(o):
    return np.array([o["food_collected"], o["food_delivered"],
                     o["hit_wall"], o["noop"]], dtype=np.float32)


def context_keys(sensors):
    """Discrete context buckets — partition memory by coarse situation.
    Every episode lands in 'ant_experience' plus whichever contextual
    flags were active. Multiple keys per scene give the reconciler
    multiple voters at query time."""
    keys = ["ant_experience"]
    keys.append("ctx_carrying" if sensors["carrying"] else "ctx_foraging")
    if sensors["food_ahead"]:
        keys.append("ctx_food_ahead")
    if sensors["wall_ahead"]:
        keys.append("ctx_wall_ahead")
    if sensors["at_nest"]:
        keys.append("ctx_at_nest")
    return keys


def build_scene(sensors, action, outcome, step):
    sv = encode_sensor(sensors)
    av = encode_action(action)
    ov = encode_outcome(outcome)
    return {
        "source_text": f"t={step}",
        "slots": [
            {"role": "sensor", "text": str(sensors),
             "vec": sv.tobytes(), "is_gap": False},
            {"role": "action", "text": action,
             "vec": av.tobytes(), "is_gap": False},
            {"role": "outcome", "text": str(outcome),
             "vec": ov.tobytes(), "is_gap": False},
        ],
        "vec": np.concatenate([sv, av, ov]).tobytes(),
        "keys": context_keys(sensors),
    }


# ---- Exploration ----

def explore(brain, n_steps=1000, seed=0):
    """Motor babbling. Random actions, store every episode, no goals."""
    world = World()
    ant = Ant(world)
    rng = np.random.default_rng(seed)
    stats = {"food_collected": 0, "food_delivered": 0, "hit_wall": 0,
             "steps": 0, "unique_positions": set()}
    for step in range(n_steps):
        sensors = ant.sense()
        action = ACTIONS[rng.integers(0, len(ACTIONS))]
        outcome = ant.execute(action)
        scene = build_scene(sensors, action, outcome, step)
        # Babbling phase: accumulate without Hebbian decay. Outcomes
        # haven't been introduced yet, so there's no basis to say
        # any scene contradicts any other.
        brain.learn_scene(scene, top_k=None, apply_hebbian=False)
        stats["food_collected"] += outcome["food_collected"]
        stats["food_delivered"] += outcome["food_delivered"]
        stats["hit_wall"] += outcome["hit_wall"]
        stats["unique_positions"].add(ant.pos)
        stats["steps"] = step + 1
    stats["unique_positions"] = len(stats["unique_positions"])
    stats["food_remaining"] = sum(world.food.values())
    return stats


# ---- Reward shaping ----

def apply_reward_weights(brain, food_boost=1.8, wall_decay=0.5,
                         max_w=2.5, min_w=0.1):
    """Reshape scene weights using each scene's outcome slot. Food-related
    scenes get their weights boosted (Hebbian reinforcement); wall-hit
    scenes decay. Rebuilds expert matrices so queries see the new shape.

    This is post-hoc reward propagation — the experience was stored
    outcome-blind during babbling; now outcomes reach back and reshape
    the memory geometry."""
    counts = {"boosted": 0, "decayed": 0, "neutral": 0}
    for lemma, expert in brain.loader.warm.items():
        changed = False
        for i, slots in enumerate(expert.scene_slots):
            outcome_slot = next(
                (s for s in slots if s.get("role") == "outcome"), None)
            if outcome_slot is None:
                counts["neutral"] += 1
                continue
            ov = np.frombuffer(outcome_slot["vec"], dtype=np.float32)
            if len(ov) < 3:
                counts["neutral"] += 1
                continue
            # ov layout: [food_collected, food_delivered, hit_wall, noop]
            food_signal = ov[0] + (ov[1] if len(ov) > 1 else 0.0)
            wall_signal = ov[2]
            if food_signal > 0.5:
                expert.scene_weights[i] = min(
                    max_w, expert.scene_weights[i] * food_boost)
                counts["boosted"] += 1
                changed = True
            elif wall_signal > 0.5:
                expert.scene_weights[i] = max(
                    min_w, expert.scene_weights[i] * wall_decay)
                counts["decayed"] += 1
                changed = True
            else:
                counts["neutral"] += 1
        if changed:
            expert._rebuild_matrix()
    return counts


# ---- Query: decide an action from sensor state ----

def decide_action(brain, sensors):
    """Ask the brain: given these sensors, what action? Returns
    (action_name, margin, sources, raw_answer_vec)."""
    sv = encode_sensor(sensors)
    # Query vec matches scene_vec layout: [sensor, action?, outcome?]
    query_vec = np.concatenate([
        sv,
        np.zeros(len(ACTIONS), dtype=np.float32),
        np.zeros(4, dtype=np.float32),
    ])
    gap = {
        "query_vec": query_vec,
        "subject_vec": np.zeros(len(query_vec), dtype=np.float32),
        "role": "action",
    }
    keys = context_keys(sensors)
    result = brain.query_scene(gap, keys)

    if result.get("answer_vec") is None:
        return None, result.get("margin", 0.0), result.get("sources", []), None

    av = np.frombuffer(result["answer_vec"], dtype=np.float32)
    # Decode: nearest canonical action one-hot
    if len(av) != len(ACTIONS):
        return None, result.get("margin", 0.0), result.get("sources", []), av
    action_idx = int(np.argmax(av))
    return ACTIONS[action_idx], result["margin"], result["sources"], av


def demo_query(brain):
    """Show the brain's action preferences for a handful of sensor states."""
    probes = [
        ("food ahead, not carrying",
         {"food_ahead": 1, "at_nest": 0, "carrying": 0,
          "wall_ahead": 0, "pheromone": 0.0}),
        ("wall ahead, foraging",
         {"food_ahead": 0, "at_nest": 0, "carrying": 0,
          "wall_ahead": 1, "pheromone": 0.0}),
        ("carrying, at nest",
         {"food_ahead": 0, "at_nest": 1, "carrying": 1,
          "wall_ahead": 0, "pheromone": 0.0}),
        ("open space, foraging",
         {"food_ahead": 0, "at_nest": 0, "carrying": 0,
          "wall_ahead": 0, "pheromone": 0.0}),
        ("carrying, open space",
         {"food_ahead": 0, "at_nest": 0, "carrying": 1,
          "wall_ahead": 0, "pheromone": 0.0}),
    ]
    for label, sensors in probes:
        action, margin, sources, av = decide_action(brain, sensors)
        if av is not None and len(av) == len(ACTIONS):
            dist = " ".join(f"{a}={v:.2f}" for a, v in zip(ACTIONS, av))
        else:
            dist = "(no action vec)"
        print(f"  {label:26s} -> {str(action):13s}  margin={margin:.3f}")
        print(f"    distribution: {dist}")
        print(f"    sources: {sources}")


# ---- Main ----

def main():
    print("Spawning ant brain (sensorimotor, no language)...")
    brain = Brain(warm_cold=False, use_language=False)
    print(f"  initial: {brain.stats()}")

    print("\n--- Phase 1: motor babbling (1000 random steps) ---")
    stats = explore(brain, n_steps=1000)
    print(f"  steps:            {stats['steps']}")
    print(f"  unique positions: {stats['unique_positions']}")
    print(f"  food collected:   {stats['food_collected']}")
    print(f"  food delivered:   {stats['food_delivered']}")
    print(f"  wall hits:        {stats['hit_wall']}")
    print(f"  food remaining:   {stats['food_remaining']}")
    print(f"  brain:            {brain.stats()}")

    print("\n--- Scene pool per expert ---")
    for lem, exp in sorted(brain.loader.warm.items(),
                           key=lambda kv: -len(kv[1].scene_vecs)):
        w = np.array(exp.scene_weights)
        print(f"  {lem:18s} scenes={len(exp.scene_vecs):4d}  "
              f"w: mean={w.mean():.3f} min={w.min():.3f} max={w.max():.3f}")

    print("\n--- Phase 2: reward shaping ---")
    counts = apply_reward_weights(brain)
    print(f"  boosted (food):   {counts['boosted']}")
    print(f"  decayed (wall):   {counts['decayed']}")
    print(f"  neutral:          {counts['neutral']}")

    print("\n--- Scene weights after reward shaping ---")
    for lem, exp in sorted(brain.loader.warm.items(),
                           key=lambda kv: -len(kv[1].scene_vecs)):
        w = np.array(exp.scene_weights)
        n_boost = int((w > 1.0 + 1e-6).sum())
        n_decay = int((w < 1.0 - 1e-6).sum())
        n_unit = int(((w > 1.0 - 1e-6) & (w < 1.0 + 1e-6)).sum())
        print(f"  {lem:18s} boosted={n_boost:4d} decayed={n_decay:4d} "
              f"unit={n_unit:4d}  w: mean={w.mean():.3f} "
              f"min={w.min():.3f} max={w.max():.3f}")

    print("\n--- Phase 3: query the brain ---")
    demo_query(brain)

    brain.oscillator.stop()
    print("\nDone.")


if __name__ == "__main__":
    main()
