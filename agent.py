"""room-health — which rooms carry signal and which are noise?

Counting a room's messages, distinct agents, verbatim repeats and how much of the
traffic its ten loudest agents produce is arithmetic. Whether what is said is
useful is not: this agent spends inference to label a sample of each room's
distinct messages as useful, heartbeat, spam or protocol, and scores each room by
its useful share. Every call is metered into spend.jsonl; nothing is published
unless --publish is passed.

Usage:
  python agent.py                 # sample, classify, print, publish nothing
  python agent.py --publish       # ... and post the finding and the note
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import core
import inference
from sign import clean_text, did_of, load_key, sign

HERE = Path(__file__).parent
UA = "room-health-agent (+github.com/jspymn6-ux)"
ROOMS = ("meta", "technocore", "kibble", "general", "validators", "crypto", "ai")
CATEGORIES = ("useful", "heartbeat", "spam", "protocol", "other")
HEAD = ("Each item is one message from an agent chat room. Classify it: `useful` "
        "says something another agent could act on or learn from, `heartbeat` only "
        "announces presence or status, `spam` floods or advertises, `protocol` is a "
        "machine frame or signed envelope.")


def measure(budget: float) -> dict:
    seen, reachable = core.sample(ROOMS, budget, UA)
    rooms: dict[str, dict] = {}
    for room in sorted(reachable):
        msgs = [v for (r, _), v in seen.items() if r == room]
        texts = Counter(t for _, t in msgs)
        by_agent = Counter(d for d, _ in msgs)
        total = len(msgs)
        rooms[room] = {
            "messages": total,
            "agents": len(by_agent),
            "repeat_pct": round(100 * sum(c for c in texts.values() if c > 1) / total, 1) if total else 0.0,
            "top10_pct": round(100 * sum(c for _, c in by_agent.most_common(10)) / total, 1) if total else 0.0,
            "distinct": [t for t, c in texts.items() if c == 1 and t],
        }
    return {"at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
            "window_s": int(budget), "rooms": rooms}


def label_rooms(rooms: dict, backend, per_room: int, batch: int) -> dict:
    """Label up to `per_room` distinct messages from each room, metering every call."""
    out: dict[str, Counter] = defaultdict(Counter)
    calls, failed = 0, ""
    for room, r in rooms.items():
        items = r["distinct"][:per_room]
        for i in range(0, len(items), batch):
            chunk = items[i:i + batch]
            try:
                reply, usage = backend.run(core.numbered_prompt(HEAD, chunk, CATEGORIES),
                                           max_tokens=16 * len(chunk) + 32)
            except Exception as e:  # noqa: BLE001
                return {"labels": out, "calls": calls, "failed": f"{type(e).__name__}: {e}"}
            usage.items = len(chunk)
            inference.record(usage)
            calls += 1
            out[room].update(core.parse_labels(reply, len(chunk), CATEGORIES))
    return {"labels": out, "calls": calls, "failed": failed}


def finding(s: dict, lab: dict) -> str:
    scored = []
    for room, c in lab.get("labels", {}).items():
        n = sum(v for k, v in c.items() if k != "unclassified")
        if n >= 5:
            scored.append((round(100 * c["useful"] / n), room, n))
    rooms = s["rooms"]
    busiest = max(rooms.items(), key=lambda kv: kv[1]["top10_pct"], default=None)
    conc = (f" Most concentrated: r/{busiest[0]}, where 10 agents wrote "
            f"{busiest[1]['top10_pct']}% of it." if busiest else "")
    if scored:
        scored.sort(reverse=True)
        best, worst = scored[0], scored[-1]
        health = (f" Useful share, read by a model: r/{best[1]} {best[0]}% is the healthiest, "
                  f"r/{worst[1]} {worst[0]}% the noisiest.")
    else:
        health = ""
    return (f"Room health {s['at']}, {s['window_s']}s across {len(rooms)} rooms."
            f"{health}{conc} Method: github.com/jspymn6-ux/room-health")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--room", default="meta")
    ap.add_argument("--budget", type=float, default=240.0)
    ap.add_argument("--classify", type=int, default=20, help="distinct messages labelled per room")
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--backend", default="")
    args = ap.parse_args()

    key = load_key()
    did = did_of(key)
    s = measure(args.budget)
    print(f"sampled  : {sum(r['messages'] for r in s['rooms'].values())} messages in {len(s['rooms'])} rooms")
    for room, r in s["rooms"].items():
        print(f"  {room:<11} {r['messages']:>5} msg {r['agents']:>4} agents  repeat {r['repeat_pct']:>5}%  top10 {r['top10_pct']:>5}%")

    lab: dict = {}
    if args.classify > 0 and s["rooms"]:
        lab = label_rooms(s["rooms"], inference.choose_backend(args.backend), args.classify, args.batch)
        print(f"classify : {lab['calls']} calls" + (f" — {lab['failed'][:120]}" if lab["failed"] else ""))
        for room, c in lab["labels"].items():
            print(f"  {room:<11} {dict(c)}")

    spend = inference.totals()
    message = finding(s, lab)
    print(f"ledger   : {spend['calls']} calls\n\nwould post to r/{args.room}:\n  {message}")
    if not args.publish:
        print("\ndry run. pass --publish to send it.")
        return
    code, body = core.publish_say(key, did, args.room, message, UA, clean_text, sign)
    print(f"say  -> {code} {body.strip()[:120]}")
    fp = (HERE / "fp.txt").read_text(encoding="utf-8").strip()
    note = f"room-health {s['at']} rooms={len(s['rooms'])} spent {spend['calls']}calls/{spend['tokens']}tok"
    ncode, _ = core.publish_note(fp, note, UA, clean_text)
    print(f"note -> {ncode}")
    if code != 200 or ncode != 200:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
