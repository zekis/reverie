"""Reverie — all constants."""

# Oscillation
CYCLE_MS = 25
MAX_CYCLES = 8
CONFIDENCE_THRESHOLD = 0.25
QUERY_WEIGHT = 0.75
ANSWER_WEIGHT = 0.25

# Wave modulation — continuous sine modulation of expert contributions.
# Replaces purely excitatory cycles with excitatory/inhibitory rhythm.
# PHASE_INCREMENT is radians per query cycle. pi/2 = quarter-wave per
# cycle, so 8 cycles sweep 2 full oscillations. Every expert peaks
# twice and troughs twice across a full query.
import math
PHASE_INCREMENT = math.pi / 2

# Experts
MAX_WARM_EXPERTS = 500
EMBEDDING_DIM = 384
MAX_SCENES_PER_EXPERT = 200
COOLING_RATE = 0.15

# Hebbian
HEBBIAN_THRESHOLD = 0.4
HEBBIAN_DECAY = 0.95
HEBBIAN_MIN_WEIGHT = 0.1

# Cold storage
import os
_BASE = os.path.dirname(os.path.abspath(__file__))
COLD_STORE_DIR = os.path.join(_BASE, "memory", "cold_store")
SNAPSHOT_DIR = os.path.join(_BASE, "memory", "snapshots")

# Learning
LOCAL_ALPHA_MIN_SAMPLES = 5

# Information gain — suppress answers that echo the question
# gain = 1 - cosine(query, answer). Below threshold, answer is
# discarded. Above threshold, final score is weighted by gain^2
# so modest-gain answers (bare wh-words, short question fragments)
# are still crushed relative to genuinely novel answers.
INFO_GAIN_THRESHOLD = 0.25

# Salience bands — three paths for interaction outcomes.
# High: confident correct → strengthen edges, no new scene
# Low: routine exchange → ignore, no update
# Abstain or wrong (with ground truth) → store corrective scene
HIGH_SALIENCE_MARGIN = 0.40
LOW_SALIENCE_MARGIN = 0.15
MATCH_SIMILARITY = 0.65       # answer↔expected similarity for "correct"

# Daydream traversal — self-sustaining activation along edges
DAYDREAM_TRAVERSAL_STRENGTH = 0.3  # strength of neighbour re-firing
DAYDREAM_MIN_ACTIVATION = 0.05     # below this, nodes don't traverse
DAYDREAM_EDGE_FANOUT = 3           # top-k edges to follow per node

# Replay buffer — hippocampal-style offline consolidation. When the
# graph falls quiet, the most recent interactions re-fire their
# source experts, strengthening edges without new queries.
REPLAY_BUFFER_SIZE = 5             # most recent interactions retained
REPLAY_DECAY = 0.7                 # each replay is this fraction of last
REPLAY_DROP_BELOW = 0.05           # drop interaction when strength below
REPLAY_EDGE_LR = 0.1               # learning rate for edge reinforcement

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# WH detection via spaCy POS tags: WDT (which/what det), WP (who/what pronoun),
# WP$ (whose), WRB (where/when/how/why adverb). No word list needed.
WH_TAGS = {"WDT", "WP", "WP$", "WRB"}
