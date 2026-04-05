"""Telegram bot wrapper for Reverie.

Connects Telegram chat → local brain API. Plain messages become
queries. Slash commands provide training and introspection.

Setup:
  1. Put your bot token in .env:  REVERIE_BOT_TOKEN=...
  2. Make sure run_brain.py is running on :7700
  3. python telegram_bot.py

Long-polling via Telegram Bot API. No external dependencies beyond
httpx (already used by brain_client).
"""

import os
import sys
import time
import html

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import httpx

from brain_client import BrainClient


# ---- token ----

def _load_token() -> str:
    # Prefer env var. Fall back to .env file next to this script.
    tok = os.environ.get("REVERIE_BOT_TOKEN")
    if tok:
        return tok.strip()
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", ".env")
    env_path = os.path.normpath(env_path)
    if os.path.exists(env_path):
        with open(env_path) as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == "REVERIE_BOT_TOKEN":
                    return v.strip().strip('"').strip("'")
    print("No REVERIE_BOT_TOKEN found.")
    print("Set the env var or add it to .env at the repo root:")
    print("  REVERIE_BOT_TOKEN=<your-token-from-BotFather>")
    sys.exit(1)


TOKEN = _load_token()
API = f"https://api.telegram.org/bot{TOKEN}"
CLIENT = BrainClient()

HELP = (
    "Reverie chat:\n"
    " - Send any text → I'll query the brain.\n\n"
    "Commands:\n"
    "/learn <text>         teach a fact\n"
    "/teach <q> :: <a>     quiz with ground truth\n"
    "/forget <lemma>       accelerate decay on an expert\n"
    "/edges <lemma>        show expert edges\n"
    "/state                current daydream state\n"
    "/stats                loader stats\n"
    "/snap                 top warm experts\n"
    "/replay               replay buffer\n"
    "/debug <text>         show parse of a question\n"
    "/help                 this message"
)


# ---- Telegram I/O ----

def _send(chat_id: int, text: str):
    # Telegram limits messages to 4096 chars; chunk if needed.
    for i in range(0, len(text), 4000):
        httpx.post(f"{API}/sendMessage", json={
            "chat_id": chat_id,
            "text": text[i:i + 4000],
            "parse_mode": "HTML",
        }, timeout=10.0)


def _get_updates(offset: int) -> list[dict]:
    try:
        r = httpx.get(f"{API}/getUpdates", params={
            "offset": offset,
            "timeout": 30,
            "allowed_updates": ["message"],
        }, timeout=35.0)
        r.raise_for_status()
        return r.json().get("result", [])
    except httpx.HTTPStatusError as e:
        print(f"[telegram] {e.response.status_code}: {e.response.text}")
        return []
    except httpx.ReadTimeout:
        return []


# ---- Message handling ----

def _handle(chat_id: int, text: str):
    text = text.strip()
    if not text:
        return
    if text.startswith("/"):
        _command(chat_id, text)
    else:
        _ask(chat_id, text)


def _ask(chat_id: int, question: str):
    try:
        r = CLIENT.query(question)
    except httpx.ConnectError:
        _send(chat_id, "Brain offline. Is run_brain.py running?")
        return
    ans = r.get("answer")
    if ans is None:
        _send(chat_id,
              f"<i>no answer</i> (cycles={r['cycles']}, "
              f"sources={r['sources']})")
        return
    body = (
        f"{html.escape(str(ans))}\n\n"
        f"<i>conf={r['confidence']:.2f} "
        f"cycles={r['cycles']} "
        f"margin={r['margin']:.2f} "
        f"sources={r['sources']}</i>"
    )
    _send(chat_id, body)


def _command(chat_id: int, line: str):
    parts = line[1:].split(maxsplit=1)
    if not parts:
        return
    cmd = parts[0].lower().split("@")[0]  # strip @botname if present
    arg = parts[1] if len(parts) > 1 else ""

    if cmd in ("start", "help"):
        _send(chat_id, HELP)
        return

    if cmd == "learn":
        if not arg:
            _send(chat_id, "usage: /learn &lt;text&gt;")
            return
        r = CLIENT.learn(arg)
        _send(chat_id, f"stored in {r['stored_in']} experts")
        return

    if cmd == "teach":
        if "::" not in arg:
            _send(chat_id, "usage: /teach &lt;question&gt; :: &lt;expected&gt;")
            return
        q, exp = (x.strip() for x in arg.split("::", 1))
        r = CLIENT.query(q, expected=exp)
        ans = r.get("answer") or ""
        match = ("✓ match" if exp.lower() in ans.lower()
                 or ans.lower() in exp.lower()
                 else "✗ mismatch — correction stored")
        _send(chat_id,
              f"<b>{html.escape(ans)}</b>  <i>{match}</i>\n"
              f"conf={r['confidence']:.2f} sources={r['sources']}")
        return

    if cmd == "forget":
        if not arg:
            _send(chat_id, "usage: /forget &lt;lemma&gt;")
            return
        try:
            r = CLIENT.forget(arg.strip())
            _send(chat_id,
                  f"forgot <b>{html.escape(arg)}</b>: "
                  f"scenes={r.get('scenes')} "
                  f"weight {r.get('avg_weight_before', 0):.2f}"
                  f"→{r.get('avg_weight_after', 0):.2f}")
        except httpx.HTTPStatusError as e:
            _send(chat_id, f"error: {e.response.text}")
        return

    if cmd == "edges":
        if not arg:
            _send(chat_id, "usage: /edges &lt;lemma&gt;")
            return
        try:
            r = CLIENT.edges(arg.strip())
        except httpx.HTTPStatusError:
            _send(chat_id, f"no expert: {arg}")
            return
        lines = [f"<b>{html.escape(arg)}</b>: scenes={r['scenes']} "
                 f"hits={r['hits']} misses={r['misses']} "
                 f"conf={r['confidence']:.2f}"]
        for e in r["edges"][:8]:
            lines.append(f"  → {e['to']}  w={e['weight']:.3f}")
        _send(chat_id, "\n".join(lines))
        return

    if cmd == "state":
        r = CLIENT.state()
        lines = [f"thinking about: {r['thinking_about']}", "top warm:"]
        for n in r["top_warm"]:
            lines.append(
                f"  {n['lemma']}  act={n['activation']:.2f} "
                f"conf={n['confidence']:.2f} scenes={n['scenes']}"
            )
        _send(chat_id, "\n".join(lines))
        return

    if cmd == "stats":
        _send(chat_id, f"<pre>{html.escape(str(CLIENT.stats()))}</pre>")
        return

    if cmd == "snap":
        snap = CLIENT.snapshot()
        top = sorted(snap.items(),
                     key=lambda x: -x[1]["activation"])[:10]
        lines = [
            f"  {lem}  act={d['activation']:.2f} "
            f"conf={d['confidence']:.2f} scenes={d['scenes']}"
            for lem, d in top
        ]
        _send(chat_id, "\n".join(lines) if lines else "(quiet)")
        return

    if cmd == "replay":
        r = CLIENT.replay()
        lines = [f"buffer: {r['size']} entries"]
        for e in r["entries"]:
            lines.append(
                f"  sources={e['sources']} "
                f"strength={e['strength']} "
                f"replays={e['replays']}"
            )
        _send(chat_id, "\n".join(lines))
        return

    if cmd == "debug":
        if not arg:
            _send(chat_id, "usage: /debug &lt;text&gt;")
            return
        r = httpx.post(f"{CLIENT.url}/debug/keys",
                       json={"text": arg}).json()
        _send(chat_id,
              f"keys={r['keys']}\n"
              f"gap_role={r['gap_role']}\n"
              f"root={r['gap_root']}  subject={r['gap_subject']}")
        return

    _send(chat_id, f"unknown command: /{cmd}")


# ---- Main loop ----

def main():
    # Verify brain reachable before polling
    try:
        stats = CLIENT.stats()
    except httpx.ConnectError:
        print("Brain not running on :7700. Start run_brain.py first.")
        sys.exit(1)

    # Verify Telegram connection
    try:
        me = httpx.get(f"{API}/getMe", timeout=10.0).json()
        if not me.get("ok"):
            print(f"Telegram auth failed: {me}")
            sys.exit(1)
        bot = me["result"]
        print(f"Connected: @{bot['username']} ({bot['first_name']})")
        print(f"Brain stats: {stats}")
        print("Listening for messages. Ctrl+C to stop.\n")
    except Exception as e:
        print(f"Telegram connection failed: {e}")
        sys.exit(1)

    offset = 0
    while True:
        try:
            updates = _get_updates(offset)
            for u in updates:
                offset = u["update_id"] + 1
                msg = u.get("message")
                if not msg or "text" not in msg:
                    continue
                chat_id = msg["chat"]["id"]
                text = msg["text"]
                user = msg.get("from", {}).get("username", "?")
                print(f"[{user}] {text[:80]}")
                _handle(chat_id, text)
        except KeyboardInterrupt:
            print("\nShutting down.")
            break
        except Exception as e:
            print(f"[error] {type(e).__name__}: {e}")
            time.sleep(2)


if __name__ == "__main__":
    main()
