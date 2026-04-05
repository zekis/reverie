"""Evolutionary ants — small-population GA over scene pools.

Each ant has its own Brain (scene pool = genome). Each lifetime: ant
uses epsilon-greedy policy, records experience, and applies per-episode
reward shaping. Across generations, fit scene pools are crossed and
mutated; unfit ones disappear.

Contrast with ant_sim.py's motor babbling:
  - babbling: one ant, post-hoc reward shaping, no selection pressure.
  - evolution: population, fitness gradient, scene-pool inheritance.

No backprop, no policy gradient. Just Hebbian reinforcement within a
lifetime plus structural inheritance between lifetimes.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from brain import Brain
from ant_sim import (World, Ant, ACTIONS, encode_sensor, encode_action,
                     encode_outcome, context_keys, build_scene,
                     apply_reward_weights)


# ---- Hyperparameters ----

POP_SIZE = 4
GENERATIONS = 12
LIFETIME_STEPS = 800
MAX_SCENES_PER_EXPERT = 2000
EPSILON_START = 0.9     # exploration rate generation 0
EPSILON_END = 0.2       # exploration rate final generation
MUTATION_RATE = 0.05    # fraction of child scenes perturbed
MUTATION_SCALE = 0.05   # vector jitter magnitude (small: preserve patterns)
TRACE_LEN = 50          # eligibility trace: reward propagates back N steps
DELIVER_BOOST = 1.8     # trace boost per scene when delivery happens
COLLECT_BOOST = 1.25    # trace boost per scene when pickup happens
WALL_DECAY = 0.75       # trace decay per scene when wall hit
BASELINE_DECAY = 0.85   # post-lifetime decay for new unreinforced scenes
SEED = 42


# ---- Ant with a brain ----

def make_brain():
    """Compact per-ant brain. No cold store, no oscillator thread."""
    return Brain(warm_cold=False, use_language=False, run_oscillator=False)


def choose_action(brain, sensors, rng, epsilon):
    """Epsilon-greedy: explore randomly or query the brain."""
    if rng.random() < epsilon:
        return ACTIONS[rng.integers(0, len(ACTIONS))]
    sv = encode_sensor(sensors)
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
    av = result.get("answer_vec")
    if av is None:
        return ACTIONS[rng.integers(0, len(ACTIONS))]
    vec = np.frombuffer(av, dtype=np.float32)
    if len(vec) != len(ACTIONS):
        return ACTIONS[rng.integers(0, len(ACTIONS))]
    return ACTIONS[int(np.argmax(vec))]


def run_lifetime(brain, rng, epsilon):
    """Ant lives for LIFETIME_STEPS with inline eligibility-trace reward
    shaping. Every stored scene is tagged with the step it was stored.
    When a reward event fires (delivery / pickup / wall hit), the last
    TRACE_LEN steps of scenes have their weights updated. This
    propagates credit backward through the action chain — the steps
    that LED TO delivery get reinforced, not just the final step."""
    world = World()
    ant = Ant(world)
    stats = {"food_collected": 0, "food_delivered": 0, "hit_wall": 0,
             "positions": set()}
    # Snapshot per-expert scene counts so we can decay only NEWLY-stored
    # scenes at end of lifetime (not scenes carried over from prior gens).
    start_counts = {id(exp): len(exp.scene_vecs)
                    for exp in brain.loader.warm.values()}
    # Eligibility trace: (expert, scene_idx, step) for recent stores
    trace: list[tuple] = []
    # Reinforcement set: (id(expert), idx) pairs touched by any trace boost
    reinforced: set = set()

    def apply_trace_boost(multiplier):
        for exp, idx, _ in trace:
            reinforced.add((id(exp), idx))
            exp.scene_weights[idx] = float(np.clip(
                exp.scene_weights[idx] * multiplier, 0.1, 2.5))

    for step in range(LIFETIME_STEPS):
        sensors = ant.sense()
        action = choose_action(brain, sensors, rng, epsilon)
        outcome = ant.execute(action)
        scene = build_scene(sensors, action, outcome, step)

        # Store under each context key directly, tracking the insertion
        # index so the trace can reach back and reweight later. Initial
        # weight 0.5 ensures new exploration scenes don't dominate
        # retrieval until trace-reinforced — long-term reinforced scenes
        # from prior generations remain at their earned weights.
        for key in scene["keys"]:
            exp = brain.loader.get_or_create(key)
            exp.store(scene, apply_hebbian=False, initial_weight=0.5)
            trace.append((exp, len(exp.scene_vecs) - 1, step))

        # Trim trace to last TRACE_LEN steps
        while trace and step - trace[0][2] > TRACE_LEN:
            trace.pop(0)

        # Reward dispatch
        if outcome["food_delivered"]:
            apply_trace_boost(DELIVER_BOOST)
        elif outcome["food_collected"]:
            apply_trace_boost(COLLECT_BOOST)
        elif outcome["hit_wall"]:
            apply_trace_boost(WALL_DECAY)

        stats["food_collected"] += outcome["food_collected"]
        stats["food_delivered"] += outcome["food_delivered"]
        stats["hit_wall"] += outcome["hit_wall"]
        stats["positions"].add(ant.pos)

    # Post-lifetime consolidation: new scenes that were never boosted by
    # any trace decay. This compresses the pool toward reinforced
    # experience — biological sleep consolidation applied to episodic
    # memory. Old scenes inherited from prior generations aren't touched.
    for exp in brain.loader.warm.values():
        start = start_counts.get(id(exp), 0)
        for i in range(start, len(exp.scene_weights)):
            if (id(exp), i) not in reinforced:
                exp.scene_weights[i] = max(
                    0.1, exp.scene_weights[i] * BASELINE_DECAY)
    # Single matrix rebuild per expert after all weight changes
    for exp in brain.loader.warm.values():
        if exp.scene_vecs:
            exp._rebuild_matrix()
    stats["positions"] = len(stats["positions"])
    return stats


def fitness(stats):
    """Delivery is the goal; pick-up is a partial credit; movement bonus."""
    return (stats["food_delivered"] * 10.0
            + stats["food_collected"] * 1.0
            + stats["positions"] * 0.05)


# ---- Genetic operators on scene pools ----

def copy_pool(src_brain, dst_brain):
    """Deep-copy every scene from src into dst (additive)."""
    for lemma, src_expert in src_brain.loader.warm.items():
        dst_expert = dst_brain.loader.get_or_create(lemma)
        for i in range(len(src_expert.scene_vecs)):
            dst_expert.scene_vecs.append(src_expert.scene_vecs[i].copy())
            dst_expert.scene_texts.append(src_expert.scene_texts[i])
            dst_expert.scene_slots.append(
                [dict(s) for s in src_expert.scene_slots[i]])
            dst_expert.scene_weights.append(src_expert.scene_weights[i])
        dst_expert._rebuild_matrix()


def crossover(parent_a, parent_b, rng, bias_a=0.5):
    """Create a child brain inheriting scenes from both parents.

    Each scene from each parent is included with probability biased by
    the parent's share. Child's scene weights are preserved — fit
    scenes carry their earned reinforcement into the next generation."""
    child = make_brain()
    for parent, prob in ((parent_a, bias_a), (parent_b, 1.0 - bias_a)):
        for lemma, expert in parent.loader.warm.items():
            dst = child.loader.get_or_create(lemma)
            for i in range(len(expert.scene_vecs)):
                if rng.random() < prob:
                    dst.scene_vecs.append(expert.scene_vecs[i].copy())
                    dst.scene_texts.append(expert.scene_texts[i])
                    dst.scene_slots.append(
                        [dict(s) for s in expert.scene_slots[i]])
                    dst.scene_weights.append(expert.scene_weights[i])
    # Prune: keep at most MAX_SCENES_PER_EXPERT per expert, highest-
    # weight scenes win. Selection pressure compresses the pool.
    for dst in child.loader.warm.values():
        if len(dst.scene_vecs) > MAX_SCENES_PER_EXPERT:
            order = np.argsort(dst.scene_weights)[::-1]
            keep = set(order[:MAX_SCENES_PER_EXPERT].tolist())
            dst.scene_vecs = [v for i, v in enumerate(dst.scene_vecs)
                              if i in keep]
            dst.scene_texts = [t for i, t in enumerate(dst.scene_texts)
                               if i in keep]
            dst.scene_slots = [s for i, s in enumerate(dst.scene_slots)
                               if i in keep]
            dst.scene_weights = [w for i, w in enumerate(dst.scene_weights)
                                 if i in keep]
        if dst.scene_vecs:
            dst._rebuild_matrix()
    return child


def mutate(brain, rng):
    """Random perturbation on a fraction of scene weights + jitter some
    scene vectors. Keeps the pool from stagnating."""
    for expert in brain.loader.warm.values():
        if not expert.scene_vecs:
            continue
        n = len(expert.scene_vecs)
        n_mut = max(1, int(n * MUTATION_RATE))
        for _ in range(n_mut):
            i = rng.integers(0, n)
            # Weight jitter
            expert.scene_weights[i] *= float(rng.uniform(0.7, 1.3))
            expert.scene_weights[i] = float(np.clip(
                expert.scene_weights[i], 0.1, 2.5))
            # Vector jitter (tiny)
            jitter = rng.normal(scale=MUTATION_SCALE,
                                size=expert.scene_vecs[i].shape
                                ).astype(np.float32)
            expert.scene_vecs[i] = expert.scene_vecs[i] + jitter
        expert._rebuild_matrix()


# ---- Evolution loop ----

def run_generation(population, rng, epsilon):
    """Score every brain in the population."""
    results = []
    for brain in population:
        stats = run_lifetime(brain, rng, epsilon)
        results.append((brain, stats, fitness(stats)))
    return results


def prune_brain(brain, weight_threshold: float = 1.0 + 1e-6):
    """Keep only reinforced scenes (weight above threshold). Drops all
    unreinforced context between generations. Reinforced scenes are
    those that were part of a reward trace during some prior lifetime —
    the distilled "what worked" memory."""
    for expert in brain.loader.warm.values():
        if not expert.scene_vecs:
            continue
        weights = np.array(expert.scene_weights)
        mask = weights > weight_threshold
        if not mask.any():
            continue
        keep = set(np.where(mask)[0].tolist())
        expert.scene_vecs = [v for i, v in enumerate(expert.scene_vecs)
                             if i in keep]
        expert.scene_texts = [t for i, t in enumerate(expert.scene_texts)
                              if i in keep]
        expert.scene_slots = [s for i, s in enumerate(expert.scene_slots)
                              if i in keep]
        expert.scene_weights = [w for i, w in enumerate(expert.scene_weights)
                                if i in keep]
        expert._rebuild_matrix()


def breed_next_gen(ranked, rng):
    """Elitism + crossover + a fresh random ant.
    ranked = list of (brain, stats, fitness) sorted desc by fitness.
    The elite is pruned so its pool doesn't bloat across generations —
    reinforced patterns survive, baseline noise gets culled."""
    elite = ranked[0][0]
    prune_brain(elite)
    next_pop = [elite]
    best_fit = ranked[0][2]
    runner_fit = ranked[1][2] if len(ranked) > 1 else best_fit
    total = max(best_fit + runner_fit, 1e-6)
    bias = best_fit / total
    # Two crossover children
    for _ in range(POP_SIZE - 2):
        child = crossover(ranked[0][0], ranked[1][0], rng, bias_a=bias)
        mutate(child, rng)
        next_pop.append(child)
    # One fresh random ant — injects novelty
    next_pop.append(make_brain())
    return next_pop


def evaluate(brain, n_episodes=3, seed_base=1000, eval_epsilon=0.05):
    """Run the brain near-deterministically for N episodes, return mean
    fitness and delivery stats. A small epsilon (0.05) breaks degenerate
    action loops at states the brain hasn't learned a clear preference
    for. Read-only — does NOT modify scene weights or add new scenes."""
    fits = []
    total_deliv = 0
    total_coll = 0
    total_walls = 0
    action_counts = np.zeros(len(ACTIONS), dtype=np.int64)
    for ep in range(n_episodes):
        rng = np.random.default_rng(seed_base + ep)
        world = World()
        ant = Ant(world)
        s = {"food_collected": 0, "food_delivered": 0, "hit_wall": 0,
             "positions": set()}
        for _ in range(LIFETIME_STEPS):
            sensors = ant.sense()
            action = choose_action(brain, sensors, rng, epsilon=eval_epsilon)
            action_counts[ACTIONS.index(action)] += 1
            outcome = ant.execute(action)
            s["food_collected"] += outcome["food_collected"]
            s["food_delivered"] += outcome["food_delivered"]
            s["hit_wall"] += outcome["hit_wall"]
            s["positions"].add(ant.pos)
        s["positions"] = len(s["positions"])
        fits.append(fitness(s))
        total_deliv += s["food_delivered"]
        total_coll += s["food_collected"]
        total_walls += s["hit_wall"]
    return {
        "mean_fitness": sum(fits) / len(fits),
        "delivered": total_deliv,
        "collected": total_coll,
        "walls": total_walls,
        "episodes": n_episodes,
        "action_dist": action_counts / action_counts.sum(),
    }


def total_scenes(brain):
    return sum(len(e.scene_vecs) for e in brain.loader.warm.values())


def main():
    rng = np.random.default_rng(SEED)
    print(f"Evolutionary ants: pop={POP_SIZE}, gens={GENERATIONS}, "
          f"lifetime={LIFETIME_STEPS}")
    population = [make_brain() for _ in range(POP_SIZE)]

    history = []
    for gen in range(GENERATIONS):
        eps = EPSILON_START + (EPSILON_END - EPSILON_START) * (
            gen / max(GENERATIONS - 1, 1))
        ranked = sorted(run_generation(population, rng, eps),
                        key=lambda x: -x[2])
        fits = [f for _, _, f in ranked]
        deliveries = [s["food_delivered"] for _, s, _ in ranked]
        collections = [s["food_collected"] for _, s, _ in ranked]
        hits = [s["hit_wall"] for _, s, _ in ranked]
        scenes = [total_scenes(b) for b, _, _ in ranked]
        print(f"\nGen {gen}  eps={eps:.2f}")
        print(f"  fitness:       {[f'{f:6.2f}' for f in fits]}")
        print(f"  food_delivered:{[f'{d:6d}' for d in deliveries]}")
        print(f"  food_collected:{[f'{c:6d}' for c in collections]}")
        print(f"  wall_hits:     {[f'{h:6d}' for h in hits]}")
        print(f"  scenes:        {[f'{s:6d}' for s in scenes]}")

        # Deterministic evaluation of the top brain — measures the
        # learned policy without exploration noise.
        ev = evaluate(ranked[0][0])
        ad = ev["action_dist"]
        print(f"  EVAL top brain (eps=0.05, {ev['episodes']}ep avg):  "
              f"fitness={ev['mean_fitness']:.2f}  "
              f"delivered={ev['delivered']}  "
              f"collected={ev['collected']}  walls={ev['walls']}")
        print(f"    action dist: " + "  ".join(
            f"{a}={ad[i]:.2f}" for i, a in enumerate(ACTIONS)))

        history.append({
            "gen": gen, "best_fit": fits[0],
            "mean_fit": sum(fits) / len(fits),
            "best_delivered": deliveries[0],
            "mean_delivered": sum(deliveries) / len(deliveries),
            "eval_fit": ev["mean_fitness"],
            "eval_delivered": ev["delivered"],
        })
        if gen < GENERATIONS - 1:
            population = breed_next_gen(ranked, rng)

    print("\n--- Evolution summary ---")
    print(f"  {'gen':>3} {'train_best':>10} {'train_mean':>10} "
          f"{'tr_deliv':>8} {'EVAL_fit':>9} {'EVAL_del':>9}")
    for h in history:
        print(f"  {h['gen']:>3} {h['best_fit']:>10.2f} "
              f"{h['mean_fit']:>10.2f} {h['best_delivered']:>8d} "
              f"{h['eval_fit']:>9.2f} {h['eval_delivered']:>9d}")

    print("\nDone.")


if __name__ == "__main__":
    main()
