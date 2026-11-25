"""The command line. Run `python -m starter_agent --help`."""
import argparse
import json
import os
import sys

from . import bdi, curiosity
from .agent import Agent, utc_now
from .config import load_config
from .embed import HashEmbedder, make_embedder
from .models import ModelError, clean, known_window, make_model
from .recall import render_session
from .store import Store, local


HASH_NOTE = ("Note: the hash embedder is in use (the default). It matches shared words and spellings, not "
             "meaning, so its recall and similarity scores don't measure the memory design. "
             "See 'Optional: better recall' in the README.")


def build(config, model=None, clock=utc_now, embeds=True, calls=True):
    """The agent. Only a command that computes scores builds the embedder (and names the hash one), and only
    one that calls the model builds the model: listing or deleting never needs a key or a download."""
    store = Store(config.db, clock)
    embedder = make_embedder(config.embedder) if embeds else None
    if isinstance(embedder, HashEmbedder):
        print(HASH_NOTE, file=sys.stderr)
    if model is None and calls:
        model = make_model(config)
    return Agent(store, model, embedder, config)


def number(text):
    """An id or a rank: a whole number SQLite can hold."""
    value = int(text)
    if not -2**63 <= value < 2**63:
        raise argparse.ArgumentTypeError(f"{text} is too large")
    return value


def ask(prompt):
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        return ""


def tell(agent):
    """What the agent recorded, skipped or couldn't do while running the command."""
    for note in agent.notes:
        print(f"  -- {note}")
    agent.notes.clear()


def show_belief(b):
    if b["flag"] == "removed":
        return f"#{b['id']}  [removed] {b['text']}"
    why = ", ".join(str(t) for t in json.loads(b["evidence"]))
    line = f"#{b['id']}  {b['confidence']:.2f}  {b['text']}  (evidence: turns {why})"
    if b["supersedes"]:
        line += f"  supersedes #{b['supersedes']}"
    return line + {"review": "  [flagged for your review]", "dropped": "  [dropped by you]"}.get(b["flag"], "")


def cmd_chat(agent, a):
    if a.message:
        print(agent.handle(" ".join(a.message)))
        tell(agent)
        return 0
    print("Type a message. /end ends the session, /quit leaves (the session stays open).")
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            return 0
        if text == "/quit":
            return 0
        if text == "/end":
            print(f"Session ended; curiosity queued {len(agent.end_session())} question(s).")
        elif text:
            print("agent>", agent.handle(text))
        tell(agent)


def cmd_session(agent, a):
    store = agent.store
    if a.action == "end":
        queued = agent.end_session()
        print(f"Session ended; curiosity queued {len(queued)} question(s). See: curious list")
    elif a.action == "list":
        for s in store.sessions():
            ended = local(s["ended_at"]) if s["ended_at"] else "(open)"
            print(f"#{s['id']}  {local(s['started_at'])} to {ended}  {s['title'] or ''}")
    elif a.id is None or store.session(a.id) is None:
        print("Give the id of an existing session (see: session list).")
        return 1
    elif a.action == "show":
        print(render_session(store, a.id))
    elif a.action == "delete":
        if not a.yes and ask(f"Delete session {a.id} and everything written in it (its turns, and the beliefs "
                             "and questions from it)? This can't be undone. Type yes: ") != "yes":
            print("Nothing deleted.")
            return 1
        print(json.dumps(agent.delete_session(a.id), indent=1))
    tell(agent)
    return 0


def cmd_desire(agent, a):
    store = agent.store
    if a.action == "add":
        if not " ".join(a.args).strip():
            print('Say what the agent should want, e.g.: desire add "Replicate figure 3" --rank 1')
            return 1
        text = " ".join(a.args)
        twin = bdi.similar_desire(store, text)
        print(f"Desire #{bdi.add_desire(store, text, a.rank)} added.")
        if twin:
            print(f"Note: desire #{twin['id']} already says nearly the same: {twin['text']}  "
                  f"(to remove one: desire drop ID)")
    elif a.action == "list":
        for d in bdi.live_desires(store):
            print(f"#{d['id']}  rank {d['rank']}  {d['text']}  ({d['source']})")
    else:
        try:
            desire_id = number(a.args[0])
            desire = store.latest("desires", desire_id)
            if desire is None or desire["flag"] == "removed" or not desire["live"]:   # dropped: add it again
                raise ValueError
            new_rank = a.rank if a.rank is not None or a.action == "drop" else number(a.args[1])
        except (IndexError, ValueError, argparse.ArgumentTypeError):
            print("Give the id of a live desire (see: desire list), and for rank the new rank: desire rank ID N")
            return 1
        if a.action == "rank":
            print(f"Desire is now #{bdi.rerank_desire(store, desire_id, new_rank)}, rank {new_rank}.")
        else:
            print(f"Desire dropped (#{bdi.drop_desire(store, desire_id)}).")
    return 0


def cmd_beliefs(agent, a):
    if a.action in ("ok", "drop"):                     # your word: it stands, or stop using it
        flag = None if a.action == "ok" else "dropped"
        before = agent.store.latest("beliefs", a.id) if a.id is not None else None
        new = bdi.review_belief(agent.store, a.id, flag) if a.id is not None else None
        if new is not None and before is not None and new == before["id"]:
            print(f"Belief #{new} wasn't flagged, so nothing changed (no new version).")
            return 0
        if new is None:
            print("Give the id of a current belief (see: beliefs).")
            return 1
        print(f"Belief #{new}: " + ("it stands, no longer flagged." if flag is None else
                                   "dropped; its history stays."))
        return 0
    if a.action == "history":
        rows = agent.store.history("beliefs", a.id) if a.id is not None else []
        if not rows:
            print("Give the id of a belief (see: beliefs).")
            return 1
    else:
        rows = bdi.current_beliefs(agent.store)
    for b in rows:
        print(show_belief(b))
    return 0


def where_from(store, intention):
    """Where an intention's words came from: a copy of the desire it serves, which you wrote or which was
    adopted from one of the agent's own questions (and the turn that raised it)."""
    desire = store.latest("desires", intention["serves"])
    if desire is None:
        return "(the desire it served is gone)"
    if desire["source"] == "curiosity":
        return (f"(words copied from desire #{desire['id']}, adopted from a question the agent raised "
                f"in turn {desire['source_turn']})")
    return f"(words copied from desire #{desire['id']}, which you wrote)"


def cmd_intentions(agent, a):
    """Read-only, like every list: no model call, nothing written. `intentions reopen ID` is your word."""
    if a.action == "reopen":
        new = bdi.reopen_intention(agent.store, a.id) if a.id is not None else None
        print(f"Intention is active again (#{new})." if new else
              "Give the id of a done intention (see: intentions).")
        return 0 if new else 1
    desire, current = bdi.choose(agent.store)          # plain code, the same pick a chat turn makes
    live = {d["key"] for d in bdi.live_desires(agent.store)}
    for i in agent.store.current("intentions", "flag IS NOT 'removed'"):
        if i["serves"] in live:                         # not one whose desire you dropped
            mark = "  (current)" if current and i["id"] == current["id"] else ""
            print(f"#{i['id']}  {i['status']}  {i['text']}{mark}\n      {where_from(agent.store, i)}")
    if desire is not None and current is None:
        print(f"(new)  active  {desire['text']}  (current; your next chat turn writes it down)")
    return 0


def cmd_curious(agent, a):
    store = agent.store
    if a.action == "list":                   # read-only: it never ends a session or calls the model
        for q in curiosity.open_questions(store):
            print(f"#{q['id']}  {q['text']}\n      \"{q['quote']}\" (turn {q['source_turn']}, "
                  f"session {q['session_id']})")
        idle = agent.idle_session()
        if idle:
            print(f"Session {idle['id']} has been idle over {agent.config.idle_minutes} minutes but is still "
                  "open; curiosity reads it once it ends.\nTo end it now: python -m starter_agent session end")
        return 0
    act = curiosity.drop if a.action == "drop" else (lambda s, i: curiosity.adopt(s, i, a.rank))
    question = store.latest("questions", a.id) if a.id is not None else None
    twin = bdi.similar_desire(store, question["text"]) if a.action == "adopt" and question else None
    result = act(store, a.id) if a.id is not None else None
    if result is None:
        print("Give the id of an open question (see: curious list).")
        return 1
    print("Dropped. It stays in the file, and a new question in nearly the same words won't be queued."
          if a.action == "drop"
          else f"Adopted as desire #{result}.")
    if twin and result is not None:
        print(f"Note: desire #{twin['id']} already says nearly the same: {twin['text']}  "
              f"(to remove one: desire drop ID)")
    return 0


def cmd_recall(agent, a):
    """What recall would hand the model for these words. Where the window is known without asking a server
    (known_window), packed to the budget a chat turn with these words would have now; elsewhere to
    recall_budget_tokens, and it says so: it never asks a server, and a chat turn there may pack less."""
    query = clean(" ".join(a.query))
    budget, context = agent.config.recall_budget_tokens, known_window(agent.config)
    if context:
        budget, room = agent.budget(agent.preview(query), context)
    if budget is None:
        print(f"Nothing recalled: a chat turn now would skip recall, since this session would leave it ~"
              f"{max(room, 0):,} tokens of the model's {context:,}-token context.")
    else:
        pack = agent.memory.recall(query, budget)
        for session, score in pack.sessions:
            print(f"session {session}  score {score:.4f}  ({pack.scored_by})")
        for session, score, tokens in pack.skipped:
            print(f"session {session}  score {score:.4f}  skipped: ~{tokens:,} tokens, more than was left of the "
                  f"{budget:,}-token budget (a session is never cut)")
        print(f"\n{pack.text}" if pack.sessions else "Nothing recalled.")
    if not context:
        print(f'(Packed to "recall_budget_tokens", {budget:,}: this model\'s window isn\'t known without asking its '
              'server, which `recall` never does, so a chat turn may pack less. Set "context_tokens" in agent.json '
              "to see what a chat turn would.)")
    idle = agent.idle_session()
    if idle:
        print(f"(Session {idle['id']} is idle but still open, so it isn't recalled yet; "
              "your next chat ends it.)")
    return 0


def cmd_log(agent, a):
    for row in agent.store.db.execute("SELECT * FROM log ORDER BY id"):
        print(f"#{row['id']}  {local(row['at'])}  session {row['session_id']}  {row['kind']}: {row['reason']}")
    return 0


def main(argv=None, model=None, clock=utc_now):
    p = argparse.ArgumentParser(prog="python -m starter_agent",
                                description="A small agent that remembers, keeps beliefs, and wonders.")
    p.add_argument("--config", help="settings file (default: agent.json, if there is one)")
    p.add_argument("--db", help="the database file (default: memory/agent.db beside agent.json; "
                                "a relative --db is read from the current folder)")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("chat", help="send one message, or none to talk until /quit")
    s.add_argument("message", nargs="*")
    s = sub.add_parser("session", help="end | list | show ID | delete ID [--yes]")
    s.add_argument("action", choices=["end", "list", "show", "delete"])
    s.add_argument("id", nargs="?", type=number)
    s.add_argument("--yes", action="store_true", help="delete without asking")
    s = sub.add_parser("desire", help='add "TEXT" [--rank N] | list | rank ID N | drop ID')
    s.add_argument("action", choices=["add", "list", "rank", "drop"])
    s.add_argument("args", nargs="*")
    s.add_argument("--rank", type=number, help="1 is the top; default: the bottom of the list")
    s = sub.add_parser("beliefs", help="the current beliefs, or: history ID | ok ID | drop ID")
    s.add_argument("action", nargs="?", default="list", choices=["list", "history", "ok", "drop"])
    s.add_argument("id", nargs="?", type=number)
    s = sub.add_parser("intentions", help="the intentions and their status, or: reopen ID")
    s.add_argument("action", nargs="?", default="list", choices=["list", "reopen"])
    s.add_argument("id", nargs="?", type=number)
    s = sub.add_parser("curious", help="list | drop ID | adopt ID [--rank N]")
    s.add_argument("action", choices=["list", "drop", "adopt"])
    s.add_argument("id", nargs="?", type=number)
    s.add_argument("--rank", type=number)
    s = sub.add_parser("recall", help="what recall would hand the model for this query")
    s.add_argument("query", nargs="+")
    sub.add_parser("log", help="what was dropped, and why; the server's counts far under the estimate")
    a = p.parse_args(argv)
    if a.config and not os.path.exists(a.config):
        raise SystemExit(f"{a.config} doesn't exist. Leave out --config to use agent.json (or the defaults).")
    calls = a.command == "chat" or (a.command == "session" and a.action == "end")
    config = load_config(a.config or "agent.json", db=a.db)
    agent = build(config, model, clock, calls or a.command == "recall", calls)
    try:
        return globals()[f"cmd_{a.command}"](agent, a)
    except KeyboardInterrupt:
        print("\nStopped. A message the model hadn't answered wasn't kept: send it again.")
        return 130
    except ModelError as error:
        print(f"{error}. See 'Choose a real model' in the README.")
        return 1
    except OSError as error:
        print(f"Could not reach the model: {error}. See 'Choose a real model' in the README.")
        return 1
    finally:
        tell(agent)
