r"""
End-of-session memory curator.

At /quit the curator reads the session and decides what's worth keeping.
The MODEL proposes; CODE validates and writes. Ported from
counselling_agent v0.4 (agent/curator.py).

The graph (a separate, tiny graph run once at session end; no checkpointer,
so it leaves nothing behind):

    START -> gather --(nothing to curate)--> END
                    \-> propose -> apply -> END

Change from v0.4: v0.4 let the model look up existing documents with the
memory tools (up to 6 calls) before a forced final tool call. But the
curator always needs the same thing, the couple's current living documents
(at most 5), so CODE fetches them into the prompt and the model makes ONE
structured-output call. Don't make the model fetch what code already knows
it needs.

The code half:

  - MemoryProposal: the shape of what the model must return. In v0.4 this
    was a JSON schema on a forced tool call. Opus 5.5 rejects forced tool
    calls, so step 3 uses STRUCTURED OUTPUT instead: the API constrains the
    model's reply to this schema. Pydantic then validates it again here.
  - format_session_material: the session's messages as plain text for the
    curator, with safety-flagged turns removed.
  - apply_proposal: writes a proposal to the memory store.

What the model never decides: which documents may be written (the
Literal below), the scope (always 'couple'), or the document keys.
"""

import json
from typing import Literal, TypedDict, get_args

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from app.memory import db, memory_store
from app.nodes.safety import check_message

# Living documents the curator may update. profile_overview has its own
# field; the append-only types (session_note, exercise_record, assessment)
# are written by code.
UpdatableType = Literal["shared_goals", "agreements", "conflict_patterns", "appreciations"]
UPDATABLE_TYPES = sorted(get_args(UpdatableType))


class DocumentUpdate(BaseModel):
    doc_type: UpdatableType
    title: str
    content: str = Field(description="The FULL updated document: merge what was already "
                                     "on file with what's new. Not a diff.")
    reason: str = Field(default="", description="One line for the developer: why this "
                                                "update. Not shown to the couple.")


class MemoryProposal(BaseModel):
    """This session's memory updates."""
    profile_overview: str = Field(description="Under ~180 words. The only document shown "
                                              "at the start of every future session.")
    session_note: str = Field(description="2-4 sentences: what happened in this session.")
    document_updates: list[DocumentUpdate] = Field(
        default_factory=list,
        description="Only for documents this session actually gave new information about. "
                    "Usually empty or one or two.")


# --- session material ---

def format_session_material(messages: list[AnyMessage],
                            exercise_records: list[dict] | None = None,
                            flagged_ids: list[str] | None = None) -> str:
    """The session as text for the curator. "" if nothing to curate.

    Leaves out:
      - tool calls and tool results (that's memory already on file);
      - SAFETY-FLAGGED TURNS, and the reply to each. The curator is told a
        flag occurred (see step 3), but never sees or writes the content,
        because both partners can read couple memory with /memory.

    Which turns were flagged: the IDs safety_gate recorded (flagged_ids),
    which covers flags from the regex AND the LLM classifier. The regex is
    re-run too, as a second net (e.g. a thread from before IDs were kept).
    Originally this only re-ran the regex; adding the classifier silently
    opened a gap, found while explaining the design in a walkthrough.
    """
    flagged = set(flagged_ids or [])
    lines: list[str] = []
    skipping = False
    for m in messages:
        if isinstance(m, HumanMessage):
            skipping = m.id in flagged or check_message(m.text).triggered
            if not skipping:
                lines.append(m.text)  # already "Name: text"
        elif isinstance(m, AIMessage) and not skipping and m.text.strip():
            lines.append(f"Companion: {m.text}")

    for rec in exercise_records or []:
        lines.append(f"\nExercise completed: {rec['exercise_key']}")
        lines.extend(f"    {k}: {v}" for k, v in rec["data"].items())

    return "\n".join(lines)


# --- writing ---

def apply_proposal(couple_id: int, session_id: int, proposal: MemoryProposal) -> list[str]:
    """Write a validated proposal to the store. Returns short notes for the
    CLI to print, so the couple sees what was remembered (v0.4's
    transparency rule). Scope is always 'couple'; keys are set here."""
    notes = []
    scope = memory_store.SCOPE_COUPLE

    overview = proposal.profile_overview.strip()
    if overview:
        memory_store.upsert_document(couple_id, "profile_overview", "profile_overview",
                                     "Profile overview", overview, doc_scope=scope)
        notes.append("Updated your profile overview.")

    note = proposal.session_note.strip()
    if note:
        memory_store.upsert_document(couple_id, "session_note", f"session_note:{session_id}",
                                     "Session note", note, doc_scope=scope)
        notes.append("Logged a note for this session.")

    for update in proposal.document_updates:
        content = update.content.strip()
        if not content:
            continue
        memory_store.upsert_document(couple_id, update.doc_type, update.doc_type,
                                     update.title.strip() or update.doc_type, content,
                                     doc_scope=scope)
        label = update.doc_type.replace("_", " ")
        reason = update.reason.strip().rstrip(".")
        notes.append(f"Updated {label}" + (f" ({reason})." if reason else "."))

    return notes


# ======================================================================
# The curator graph
# ======================================================================

CURATOR_SYSTEM_PROMPT = """You are the memory curator for a relationship-skills \
companion used by a couple together. You run at the end of a session to decide what \
is worth remembering long-term. You are NOT talking to the couple now.

BOTH partners can read everything you write (via /memory), so:
- Stay neutral. Never assign blame, never say who was right, never take sides.
- Never diagnose or label either person or the relationship.
- Describe conflict as a shared pattern the two of them get caught in, using their \
own words for it where they have them.

You'll be given the couple's current living documents and this session's \
conversation. Return:
- profile_overview: under ~180 words, the only note shown to the companion at the \
start of every future session. Keep it current and short; point to other documents \
rather than repeating them.
- session_note: 2-4 sentences on what happened in THIS session.
- document_updates: only for documents this session gave real new information \
about (most sessions touch none or one). Each update is the FULL document: merge \
what's on file with what's new, don't write a diff. Types: shared_goals (shared \
plans and meaning, e.g. marriage, family), agreements (what they agreed to try, in \
their words), conflict_patterns (their recurring cycle, in their words), \
appreciations (what each values in the other).

Rules:
- If you're told a safety flag occurred, don't write anything about it or hint at \
it. Write only about the rest of the session.
- Write like a considerate colleague's notes, not clinical jargon.
- If nothing substantive happened, keep document_updates empty and the other two \
brief and honest about that.
- Trust the date and time-since-last-session you're given; you are not reliable at \
computing time gaps. Don't write specific dates into documents (they're timestamped \
in storage)."""

SAFETY_FLAG_NOTE = ("NOTE: the safety system flagged part of this session. That part "
                    "has been removed from what you see. Don't mention or hint at it.")


class CuratorState(TypedDict, total=False):
    # Input (set by the caller)
    couple_id: int
    session_id: int
    messages: list[AnyMessage]       # the session, from the checkpoint
    safety_flagged: bool
    flagged_ids: list[str]           # IDs of messages the safety gate flagged
    now_text: str | None
    temporal_context: str | None
    # Set by nodes
    prompt: str | None               # gather: None means nothing to curate
    proposal: MemoryProposal | None  # propose
    notes: list[str]                 # apply


def _session_exercise_records(couple_id: int, session_id: int) -> list[dict]:
    return [{"exercise_key": r["exercise_key"], "data": json.loads(r["data_json"])}
            for r in db.list_exercise_records(couple_id)
            if r["session_id"] == session_id]


def _existing_documents(couple_id: int) -> str:
    parts = []
    for doc_type in ("profile_overview", *UPDATABLE_TYPES):
        doc = memory_store.get_latest_by_key(couple_id, doc_type)
        parts.append(f"[{doc_type}]\n{doc['content'] if doc else '(none yet)'}")
    return "\n\n".join(parts)


def gather(state: CuratorState) -> dict:
    """CODE assembles everything the model needs: no tool calls."""
    material = format_session_material(
        state["messages"], _session_exercise_records(state["couple_id"], state["session_id"]),
        state.get("flagged_ids"))
    if not material.strip():
        return {"prompt": None}

    time_line = f"TODAY: {state['now_text']}." if state.get("now_text") else ""
    if state.get("temporal_context"):
        time_line += f" {state['temporal_context']}"
    sections = [
        time_line,
        SAFETY_FLAG_NOTE if state.get("safety_flagged") else "",
        f"CURRENT DOCUMENTS ON FILE:\n{_existing_documents(state['couple_id'])}",
        f"THIS SESSION:\n{material}",
    ]
    return {"prompt": "\n\n".join(s for s in sections if s)}


def route_after_gather(state: CuratorState) -> str:
    return "propose" if state.get("prompt") else "end"


def make_propose_node(model):
    # STRUCTURED OUTPUT (method="json_schema"): the API constrains the reply
    # to MemoryProposal's JSON schema, and LangChain parses it into a
    # MemoryProposal. Not "function_calling", LangChain's default, which
    # forces a tool call: Opus 5.5 rejects forced tool calls.
    structured = model.with_structured_output(MemoryProposal, method="json_schema")

    def propose(state: CuratorState) -> dict:
        proposal = structured.invoke([SystemMessage(content=CURATOR_SYSTEM_PROMPT),
                                      HumanMessage(content=state["prompt"])])
        return {"proposal": proposal}

    return propose


def apply(state: CuratorState) -> dict:
    return {"notes": apply_proposal(state["couple_id"], state["session_id"], state["proposal"])}


def build_curator_graph(model):
    g = StateGraph(CuratorState)
    g.add_node("gather", gather)
    g.add_node("propose", make_propose_node(model))
    g.add_node("apply", apply)
    g.add_edge(START, "gather")
    g.add_conditional_edges("gather", route_after_gather, {"propose": "propose", "end": END})
    g.add_edge("propose", "apply")
    g.add_edge("apply", END)
    return g.compile()  # no checkpointer: nothing from the session is kept


def curate_session(model, couple_id: int, session_id: int, messages: list[AnyMessage],
                   safety_flagged: bool = False, now_text: str | None = None,
                   temporal_context: str | None = None,
                   flagged_ids: list[str] | None = None) -> list[str]:
    """Run the curator. Returns the notes to show the couple ([] if nothing
    was written). NEVER raises: a failed curation must not block exit
    (v0.4 rule). The cost of failing is one session's memory, not the app."""
    try:
        out = build_curator_graph(model).invoke(
            {"couple_id": couple_id, "session_id": session_id, "messages": messages,
             "safety_flagged": safety_flagged, "flagged_ids": flagged_ids or [],
             "now_text": now_text,
             "temporal_context": temporal_context},
            {"run_name": "curator"})
        return out.get("notes", [])
    except Exception:
        return []
