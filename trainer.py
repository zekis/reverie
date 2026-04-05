"""Load lessons, teach facts, run quizzes, log metrics.

Usage:
  python trainer.py lessons/*.yaml
  python trainer.py lessons/relationship.yaml --reps 5
  python trainer.py lessons/*.yaml --no-teach     # quiz only
  python trainer.py lessons/*.yaml --no-quiz      # teach only

A lesson file is:
  topic: <name>
  reps: <int>                   # how many quiz passes
  facts: [<str>, ...]           # sent to /learn
  quiz:                         # sent to /query with expected=
    - q: <question>
      expected: <answer>

Metrics appended to memory/metrics/runs.csv per run per topic:
  timestamp, topic, reps, hits, total, pass_rate, avg_confidence
"""

import os
import sys
import csv
import time
import glob
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml

from brain_client import BrainClient


METRICS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "memory", "metrics", "runs.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lessons", nargs="+",
                    help="lesson YAML files (glob patterns allowed)")
    ap.add_argument("--reps", type=int, default=None,
                    help="override reps in lesson files")
    ap.add_argument("--no-teach", action="store_true",
                    help="skip /learn, just run quizzes")
    ap.add_argument("--no-quiz", action="store_true",
                    help="skip quizzes, just teach facts")
    args = ap.parse_args()

    # Expand globs
    paths: list[str] = []
    for pattern in args.lessons:
        expanded = sorted(glob.glob(pattern))
        if not expanded:
            print(f"No match: {pattern}")
            continue
        paths.extend(expanded)
    if not paths:
        print("No lesson files found.")
        sys.exit(1)

    client = BrainClient()
    try:
        print(f"Connected: {client.stats()}\n")
    except Exception as e:
        print(f"Can't reach brain: {e}")
        sys.exit(1)

    os.makedirs(os.path.dirname(METRICS_CSV), exist_ok=True)
    run_ts = int(time.time())

    for path in paths:
        with open(path) as f:
            lesson = yaml.safe_load(f)
        topic = lesson.get("topic", os.path.splitext(
            os.path.basename(path))[0])
        reps = args.reps or lesson.get("reps", 3)
        facts = lesson.get("facts", [])
        quiz = lesson.get("quiz", [])

        print(f"=== {topic} ({os.path.basename(path)}) ===")

        if not args.no_teach and facts:
            print(f"Teaching {len(facts)} facts...")
            for fact in facts:
                r = client.learn(fact)
                print(f"  ({r['stored_in']}) {fact}")
            print()

        if args.no_quiz or not quiz:
            continue

        print(f"Quiz: {len(quiz)} questions × {reps} reps")
        hits = 0
        total = 0
        conf_sum = 0.0
        per_question_hits: dict[str, int] = {}

        for rep in range(reps):
            for item in quiz:
                q = item["q"]
                expected = item["expected"]
                r = client.query(q, expected=expected)
                total += 1
                conf_sum += r.get("confidence", 0.0)
                ans = r.get("answer") or ""
                hit = _is_hit(ans, expected)
                if hit:
                    hits += 1
                    per_question_hits[q] = per_question_hits.get(q, 0) + 1
                marker = "ok  " if hit else "miss"
                print(f"  rep{rep+1} [{marker}] "
                      f"{ans!r:30s} exp={expected!r:20s} "
                      f"conf={r.get('confidence', 0):.2f} "
                      f":: {q}")

        pass_rate = hits / total if total else 0.0
        avg_conf = conf_sum / total if total else 0.0
        print(f"\n  Topic result: {hits}/{total} "
              f"pass_rate={pass_rate:.1%} "
              f"avg_conf={avg_conf:.2f}")

        weak = [q for q in {i["q"] for i in quiz}
                if per_question_hits.get(q, 0) < reps]
        if weak:
            print(f"  Weak: {len(weak)} questions didn't hit every rep")
            for q in weak:
                print(f"    ({per_question_hits.get(q, 0)}/{reps}) {q}")

        _append_metrics(run_ts, topic, reps, hits, total,
                        pass_rate, avg_conf)
        print()

    # Persist brain state so lesson effects survive restart
    try:
        import httpx
        r = httpx.post(f"{client.url}/save").json()
        print(f"Saved {r.get('saved', '?')} experts to cold storage.")
    except Exception as e:
        print(f"Save failed: {e}")


def _is_hit(answer: str, expected: str) -> bool:
    if not answer:
        return False
    a, e = answer.strip().lower(), expected.strip().lower()
    return e in a or a in e


def _append_metrics(ts, topic, reps, hits, total, pass_rate, avg_conf):
    write_header = not os.path.exists(METRICS_CSV)
    with open(METRICS_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["timestamp", "topic", "reps", "hits", "total",
                        "pass_rate", "avg_confidence"])
        w.writerow([ts, topic, reps, hits, total,
                    f"{pass_rate:.4f}", f"{avg_conf:.4f}"])


if __name__ == "__main__":
    main()
