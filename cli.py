#!/usr/bin/env python3
"""
CLI for the couples agent. Run: python cli.py

One couple login per session. Each line either is a local command (handled
here, never sent to the graph or the model) or is a message from whichever
partner is currently speaking, which runs the per-turn graph once.

Ported from counselling_agent v0.4 (main.py): login flow, /memory, /crisis,
two-step /forget. New: speaker switching, the LangGraph turn loop, and at
session end: the curator (on /quit or /logout), then thread deletion.
"""

import sys
from getpass import getpass

from dotenv import load_dotenv

from app.curator import curate_session
from app.graph import (
    build_graph,
    delete_session_thread,
    make_checkpointer,
    sweep_orphan_threads,
    thread_config,
)
from app.memory import auth, db, memory_store, timeutil
from app.model import OfflineChatModel, get_model
from app.nodes.context import compute_temporal_context
from app.prompts import speaker_message
from app.resources.crisis_resources import format_crisis_message
from app.state import new_session_state

BANNER = """\
========================================================================
 Couples relationship-skills companion (learning project, not therapy)
========================================================================
This is a personal learning project, not couples therapy, a clinician,
or a crisis service. It won't take sides or diagnose either of you.

If either of you is in danger right now, call 000. Type /crisis at any
time for support services.
------------------------------------------------------------------------
"""

HELP_TEXT = """\
Who's speaking:
  /a  or  /b               switch speaker (add text to switch and send)
  Name: message            speak as that partner, e.g. "{a}: I felt ignored"
  /who                     show who is speaking now

Commands:
  /memory                  list what's stored about you as a couple
  /memory <n>              show one memory document in full
  /crisis                  show support services immediately
  /debug                   toggle showing each graph node as it runs
  /logout                  save memory, end this session and log out
  /forget                  permanently delete this couple account and all data
  /quit                    save memory, end the session and exit
  /help                    show this message

Anything else is a message from the current speaker.
"""

# Only the couple's own documents; shared reference notes aren't "about" them.
_MEMORY_SCOPES = {memory_store.SCOPE_COUPLE}


# --- login / registration ---

def _ask(prompt: str, secret: bool = False) -> str | None:
    try:
        return (getpass(prompt) if secret else input(prompt)).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None


def prompt_login() -> tuple[int, str] | None:
    """(couple_id, username), or None if they chose to quit."""
    print("Log in, or enter a new username to create a couple account.\n")
    while True:
        username = _ask("Username (or /quit): ")
        if username is None or username.lower() in ("/quit", "/exit"):
            return None
        if not username:
            continue

        if db.get_couple_by_username(username):
            password = _ask("Password: ", secret=True)
            if password is None:
                return None
            try:
                return auth.login(username, password), username
            except auth.AuthError as e:
                print(f"{e}\n")
                continue

        confirm = _ask(f"No account for '{username}'. Create one for the two of you? (y/N) ")
        if confirm is None:
            return None
        if confirm.lower() != "y":
            continue
        password = _ask("Choose a shared password (min 8 characters): ", secret=True)
        password2 = _ask("Confirm password: ", secret=True)
        if password is None or password2 is None:
            return None
        if password != password2:
            print("Passwords didn't match; let's try again.\n")
            continue
        name_a = _ask("First partner's name: ")
        name_b = _ask("Second partner's name: ")
        if name_a is None or name_b is None:
            return None
        try:
            couple_id = auth.register(username, password, name_a, name_b)
        except auth.AuthError as e:
            print(f"{e}\n")
            continue
        print(f"Account created. Welcome, {name_a.strip()} and {name_b.strip()}.\n")
        return couple_id, username


# --- speaker parsing ---

def parse_turn(line: str, speaker: str, names: dict[str, str]) -> tuple[str, str]:
    """Work out who is speaking and what they said.
    Returns (speaker, text); text is "" for a bare switch like "/a"."""
    lowered = line.lower()
    for cmd, who in (("/a", "partner_a"), ("/b", "partner_b")):
        if lowered == cmd or lowered.startswith(cmd + " "):
            return who, line[len(cmd):].strip()
    head, sep, rest = line.partition(":")
    if sep:
        for who, name in names.items():
            if head.strip().casefold() == name.casefold():
                return who, rest.strip()
    return speaker, line


# --- memory transparency (v0.4 /memory) ---

def show_memory_index(couple_id: int):
    docs = memory_store.list_documents(couple_id, _MEMORY_SCOPES, limit=50)
    if not docs:
        print("\nNothing stored yet. This fills in as you use the app.")
        return
    print(f"\nWhat's stored about you as a couple ({len(docs)} document(s)):\n")
    for d in docs:
        rel = timeutil.humanize_delta(d["updated_at"])
        print(f"  [{d['id']}] {d['doc_type']:<18} {d['title']}  (updated {rel})")
    print("\nUse /memory <id> to read one in full.")


def show_memory_document(couple_id: int, doc_id: int):
    doc = memory_store.get_document(doc_id)
    if not doc or doc["couple_id"] != couple_id or doc["doc_scope"] not in _MEMORY_SCOPES:
        print(f"\nNo document with id {doc_id} for this account.")
        return
    rel = timeutil.humanize_delta(doc["updated_at"])
    print(f"\n--- {doc['title']} ({doc['doc_type']}, v{doc['version']}), updated {rel} ---")
    if doc["doc_type"] in memory_store.LIVING_TYPES:
        history = memory_store.get_document_history(couple_id, doc["doc_key"])
        if len(history) > 1:
            since = timeutil.humanize_delta(history[0]["created_at"])
            print(f"(Tracked since {since}; {len(history)} versions on file)")
    print(doc["content"])


def confirm_forget(couple_id: int, username: str) -> bool:
    print(f"\nThis will PERMANENTLY delete the couple account '{username}' and every "
          f"memory document, exercise record and safety log for BOTH of you. "
          f"This cannot be undone.")
    first = _ask("Type the username again to confirm: ")
    if first != username:
        print("Didn't match; cancelled.")
        return False
    second = _ask("Are you sure? (yes/no) ")
    if (second or "").lower() != "yes":
        print("Cancelled.")
        return False
    db.delete_couple_data(couple_id)
    print("Done. The account and all associated data have been deleted.")
    return True


# --- one turn through the graph ---

def run_turn(graph, turn_input: dict, config: dict, debug: bool = False):
    """Run the graph once and print what the couple should see.

    stream_mode="updates" yields {node_name: partial_update} after each node
    runs, so we can show progress (tool use, and every node in /debug mode).
    The reply itself is printed only after output_check has run: streaming
    tokens straight from the model would show text the backstop might then
    replace."""
    for chunk in graph.stream(turn_input, config, stream_mode="updates"):
        for node, update in chunk.items():
            if debug:
                print(f"  [node: {node}]")
            if node == "converse" and update:
                for call in update["messages"][-1].tool_calls:
                    args = ", ".join(f"{k}={v!r}" for k, v in call["args"].items())
                    print(f"  [checking memory: {call['name']}({args})]")

    # Code-written text (crisis resources) first, then the model's reply,
    # unless the IPV protocol ended the session (no model reply that turn).
    state = graph.get_state(config).values
    for text in state.get("display", []):
        print("\n" + text)
    if not state.get("session_ended"):
        print("\n" + state["messages"][-1].text)
    return state


def run_session(couple_id: int, username: str, model, checkpointer) -> str:
    """Returns 'quit' or 'logout'."""
    names = db.get_partner_names(couple_id)
    session_id = db.start_session(couple_id)
    config = thread_config(session_id)
    graph = build_graph(model, checkpointer)
    _, _, gap = compute_temporal_context(couple_id, session_id)

    a, b = names["partner_a"], names["partner_b"]
    if gap:
        print(f"\nWelcome back, {a} and {b}. It's been {gap} since your last session.")
    else:
        print(f"\nWelcome, {a} and {b}.")
    print(f"Type /help for commands. Speaking now: {a} (switch with /a, /b or '{b}: ...').")
    if isinstance(model, OfflineChatModel):
        print("[No API key configured: commands and the safety check work, chat replies don't.]")

    speaker, first_turn, debug, result = "partner_a", True, False, "quit"
    # Curate only on a deliberate /quit or /logout. Not on Ctrl+C/EOF or a
    # crash, not after /forget, and not after the IPV protocol ended it.
    curate = False
    try:
        while True:
            line = _ask(f"\n{names[speaker]}> ")
            if line is None:
                break
            if not line:
                continue

            cmd, _, arg = line.partition(" ")
            cmd = cmd.lower()
            if cmd in ("/quit", "/exit"):
                curate = True
                break
            if cmd == "/logout":
                result, curate = "logout", True
                break
            if cmd == "/help":
                print("\n" + HELP_TEXT.format(a=a))
                continue
            if cmd == "/who":
                print(f"Speaking now: {names[speaker]}")
                continue
            if cmd == "/debug":
                debug = not debug
                print(f"Node tracing {'on' if debug else 'off'}.")
                continue
            if cmd == "/crisis":
                print("\n" + format_crisis_message())
                continue
            if cmd == "/memory":
                if arg.strip().isdigit():
                    show_memory_document(couple_id, int(arg))
                else:
                    show_memory_index(couple_id)
                continue
            if cmd == "/forget":
                if confirm_forget(couple_id, username):
                    return "quit"  # session row is gone with the account
                continue

            speaker, text = parse_turn(line, speaker, names)
            if not text:
                print(f"Speaking now: {names[speaker]}")
                continue

            turn = {"messages": [speaker_message(text, speaker, names)], "speaker": speaker}
            if first_turn:
                turn = {**new_session_state(couple_id, session_id, names), **turn}
            try:
                state = run_turn(graph, turn, config, debug)
                first_turn = False
            except Exception as e:  # API/network errors: report, keep the session alive
                print(f"\n[Something went wrong reaching the model: {type(e).__name__}: {e}]",
                      file=sys.stderr)
                continue
            if state.get("session_ended"):  # IPV protocol: wrap up, no curator
                print("\n(This session has ended. Take care.)")
                result = "logout"
                break
    finally:
        # Curator first (it reads the session from the checkpoint), THEN
        # always close the session and delete its message history.
        if curate and not first_turn:
            try:
                run_curator(graph, config, model, couple_id, session_id)
            except KeyboardInterrupt:  # impatient Ctrl+C: skip memory, still clean up
                print("\n[Skipped saving memory.]")
        if db.get_partner_names(couple_id):  # account still exists (not /forget)
            db.end_session(session_id)
        delete_session_thread(checkpointer, session_id)
    return result


def run_curator(graph, config: dict, model, couple_id: int, session_id: int):
    """Curate the session's memory and show the couple what was remembered
    (v0.4's transparency rule). Never raises: exit must not be blocked."""
    try:
        state = graph.get_state(config).values
    except Exception:
        return
    if not state.get("messages") or state.get("session_ended"):
        return  # nothing said / IPV protocol: write nothing at all
    print("\n[Saving what to remember from this session...]")
    notes = curate_session(model, couple_id, session_id, state["messages"],
                           safety_flagged=bool(state.get("safety_flags")),
                           now_text=state.get("now_text"),
                           temporal_context=state.get("temporal_context"))
    for note in notes:
        print(f"  - {note}")
    if notes:
        print("  (See /memory next time to read or check it.)")


def main():
    load_dotenv()
    db.init_db()
    memory_store.seed_reference_documents()
    checkpointer = make_checkpointer()
    leftover = sweep_orphan_threads(checkpointer)  # no session is active at startup
    if leftover:
        print(f"[Deleted {len(leftover)} leftover conversation thread(s) from an earlier crash.]")
    model = get_model()
    print(BANNER)

    while True:
        login = prompt_login()
        if login is None:
            break
        if run_session(*login, model, checkpointer) == "quit":
            break
        print("\nLogged out.\n")
    print("\nTake care of each other.")


if __name__ == "__main__":
    main()
