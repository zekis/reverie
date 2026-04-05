"""The brain, running. Long-lived process with an HTTP API.

Boots the Brain, serves the API in a background thread, prints
daydream state on the console every REPORT_INTERVAL seconds.
Ctrl+C for clean shutdown.

Clients (text apps, training APIs, anything) connect via HTTP on
port BRAIN_PORT. This console is a debug window into what the
brain is currently thinking.
"""

import os
import sys
import time
import signal
import threading
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from brain import Brain
from api import build_app


BRAIN_HOST = "127.0.0.1"
BRAIN_PORT = 7700
REPORT_INTERVAL = 5.0


def main():
    print("Booting brain...")
    t0 = time.time()
    brain = Brain()
    print(f"Brain online ({time.time()-t0:.1f}s)")

    # Build API and launch uvicorn in a background thread
    app = build_app(brain)
    config = uvicorn.Config(
        app, host=BRAIN_HOST, port=BRAIN_PORT,
        log_level="warning",  # suppress access log noise
    )
    server = uvicorn.Server(config)
    api_thread = threading.Thread(target=server.run, daemon=True)
    api_thread.start()

    # Wait for API ready
    time.sleep(0.5)
    print(f"API listening on http://{BRAIN_HOST}:{BRAIN_PORT}")
    print(f"Stats: {brain.stats()}")
    print("Ctrl+C to stop.\n")

    def shutdown(sig, frame):
        print("\n\nShutting down...")
        brain.stop()
        server.should_exit = True
        print(f"Final stats: {brain.stats()}")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Main thread: print daydream state on a loop
    start = time.time()
    while True:
        time.sleep(REPORT_INTERVAL)
        elapsed = time.time() - start
        state = brain.daydream_state()
        snap = brain.snapshot()
        stats = brain.stats()

        print(f"--- t={elapsed:.0f}s  warm={stats['warm']} "
              f"cold={stats['cold']} scenes={stats['total_scenes']} ---")
        print(f"Thinking about: {state if state else '(quiet)'}")
        if snap:
            top = sorted(snap.items(),
                         key=lambda x: -x[1]["activation"])[:5]
            for lemma, d in top:
                print(f"  {lemma:15s} act={d['activation']:.3f} "
                      f"conf={d['confidence']:.2f} "
                      f"scenes={d['scenes']}")
        print()


if __name__ == "__main__":
    main()
