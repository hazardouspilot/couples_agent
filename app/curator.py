"""
End-of-session memory curator, part 1: the pure-code half.

At /quit the curator reads the session and decides what's worth keeping.
The MODEL proposes; CODE validates and writes. This module holds the code
half, ported from counselling_agent v0.4 (agent/curator.py):

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

from typing import Literal

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from pydantic import BaseModel, Field

from app.memory import memory_store
from app.nodes.safety import check_message

# Living documents the curator may update. profile_overview has its own
# field; the append-only types (session_note, exercise_record, assessment)
# are written by code.
UpdatableType = Literal["shared_goals", "agreements", "conflict_patterns", "appreciations"]


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
                            exercise_records: list[dict] | None = None) -> str:
    """The session as text for the curator. "" if nothing to curate.

    Leaves out:
      - tool calls and tool results (that's memory already on file);
      - SAFETY-FLAGGED TURNS: any partner message the safety regex flags,
        and the reply to it. The same check_message as safety_gate, re-run
        here, so no extra state is needed. The curator is told a flag
        occurred (see step 3), but never sees or writes the content, because
        both partners can read couple memory with /memory.
    """
    lines: list[str] = []
    skipping = False
    for m in messages:
        if isinstance(m, HumanMessage):
            skipping = check_message(m.text).triggered
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
        notes.append(f"Updated {label}" + (f" ({update.reason})." if update.reason else "."))

    return notes
