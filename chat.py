"""REPL for training and talking to Reverie.

Default mode: type a question, get an answer.
Prefix with ':' for commands (learn, teach, forget, state, etc.).
Paste multi-line text and it'll be split on blank lines.

Requires run_brain.py running on localhost:7700.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import readline  # noqa: history + line editing where available
except ImportError:
    pass

import httpx

from brain_client import BrainClient


HELP = """
Commands (prefix with ':'):
  :learn <text>       teach a fact                 (alias: :l)
  :teach <q> :: <a>   query with ground truth      (alias: :t)
  :forget <lemma>     accelerate decay on expert   (alias: :f)
  :edges <lemma>      show an expert's edges       (alias: :e)
  :state              current daydream state       (alias: :s)
  :snap               full warm snapshot
  :replay             replay buffer contents
  :stats              loader stats
  :save               persist all warm experts
  :debug <text>       show parse of a question
  :help               this message
  :quit               exit                         (alias: :q)

Paste multi-line input — each non-empty line is treated separately.
Blank lines separate inputs.
"""


def main():
    client = BrainClient()
    try:
        stats = client.stats()
    except httpx.ConnectError:
        print("Can't reach brain at http://localhost:7700.")
        print("Is run_brain.py running?")
        sys.exit(1)

    print(f"Connected to Reverie. {stats}")
    print("Type a question. Prefix ':' for commands. :help for details.\n")

    while True:
        try:
            line = input("reverie> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        try:
            if line.startswith(":"):
                if not _command(client, line[1:].strip()):
                    break
            else:
                _ask(client, line)
        except httpx.HTTPStatusError as e:
            print(f"  [error] {e.response.status_code}: {e.response.text}")
        except httpx.ConnectError:
            print("  [error] brain disconnected")
            break
        except Exception as e:
            print(f"  [error] {type(e).__name__}: {e}")


def _ask(client: BrainClient, question: str):
    r = client.query(question)
    ans = r.get("answer")
    if ans is None:
        print(f"  (no answer — cycles={r['cycles']}, sources={r['sources']})")
        return
    print(f"  {ans}")
    print(f"    conf={r['confidence']:.2f} cycles={r['cycles']} "
          f"margin={r['margin']:.3f} sources={r['sources']}")


def _command(client: BrainClient, cmd: str) -> bool:
    """Return False to quit, True otherwise."""
    if not cmd:
        return True
    parts = cmd.split(maxsplit=1)
    op = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if op in ("q", "quit", "exit"):
        return False

    if op in ("h", "help"):
        print(HELP)
        return True

    if op in ("l", "learn"):
        if not arg:
            print("  usage: :learn <text>")
            return True
        r = client.learn(arg)
        print(f"  stored in {r['stored_in']} experts")
        return True

    if op in ("t", "teach"):
        if "::" not in arg:
            print("  usage: :teach <question> :: <expected answer>")
            return True
        q, expected = (x.strip() for x in arg.split("::", 1))
        r = client.query(q, expected=expected)
        ans = r.get("answer")
        match = "[match]" if _matches(ans, expected) else "[mismatch → correction stored]"
        print(f"  {ans!r} {match}")
        print(f"    conf={r['confidence']:.2f} sources={r['sources']}")
        return True

    if op in ("f", "forget"):
        if not arg:
            print("  usage: :forget <lemma>")
            return True
        r = client.forget(arg.strip())
        print(f"  {r}")
        return True

    if op in ("e", "edges"):
        if not arg:
            print("  usage: :edges <lemma>")
            return True
        r = client.edges(arg.strip())
        print(f"  {arg}: scenes={r['scenes']} hits={r['hits']} "
              f"misses={r['misses']} conf={r['confidence']:.2f}")
        for edge in r["edges"][:8]:
            print(f"    -> {edge['to']:20s} w={edge['weight']:.3f}")
        return True

    if op in ("s", "state"):
        r = client.state()
        print(f"  thinking about: {r['thinking_about']}")
        print("  top warm:")
        for n in r["top_warm"]:
            print(f"    {n['lemma']:15s} act={n['activation']:.2f} "
                  f"conf={n['confidence']:.2f} scenes={n['scenes']}")
        return True

    if op == "snap":
        snap = client.snapshot()
        top = sorted(snap.items(),
                     key=lambda x: -x[1]["activation"])[:10]
        for lem, d in top:
            print(f"  {lem:15s} act={d['activation']:.2f} "
                  f"conf={d['confidence']:.2f} scenes={d['scenes']}")
        return True

    if op == "replay":
        r = client.replay()
        print(f"  buffer: {r['size']} entries")
        for e in r["entries"]:
            print(f"    sources={e['sources']} "
                  f"strength={e['strength']} replays={e['replays']}")
        return True

    if op == "stats":
        print(f"  {client.stats()}")
        return True

    if op == "save":
        r = httpx.post(f"{client.url}/save").json()
        print(f"  saved {r['saved']} experts")
        return True

    if op == "debug":
        if not arg:
            print("  usage: :debug <text>")
            return True
        r = httpx.post(f"{client.url}/debug/keys",
                       json={"text": arg}).json()
        print(f"  keys={r['keys']} gap_role={r['gap_role']} "
              f"root={r['gap_root']} subject={r['gap_subject']}")
        return True

    print(f"  unknown command: :{op}. :help for list.")
    return True


def _matches(answer: str | None, expected: str) -> bool:
    if not answer:
        return False
    a, e = answer.lower(), expected.lower()
    return e in a or a in e


if __name__ == "__main__":
    main()
