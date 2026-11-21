"""
Memory retrieval for the model: the three read-only operations behind the
`search_memory`, `get_memory_document` and `list_memory_documents` tools.

Ported from counselling_agent v0.4 (agent/memory_tools.dispatch), minus the
Anthropic tool schemas. This module is plain Python on purpose: next
session, app/tools/memory.py wraps each function with LangChain's @tool and
a ToolNode runs them. Keeping the logic here means it's tested without any
LLM or framework, and swapping frameworks wouldn't touch it.

Access control lives HERE, in code, not in the prompt:
  - Every function takes couple_id from the session (graph state), never
    from the model. The @tool wrappers will hide it from the model's view
    of the tool's arguments, so the model cannot even ask for another couple.
  - Only JOINT_SESSION_SCOPES (couple + reference) are ever returned in v1;
    partner_private is invisible, including by guessing a doc_id.
  - "Not found" and "not yours" return the identical error, so the model
    (or anyone reading its output) can't probe which ids exist.

Results are JSON-serialisable dicts. Errors are returned as {"error": ...}
rather than raised, so the model sees what went wrong and can try again,
instead of the turn crashing.

Every couple document carries a precomputed `relative_time` ("3 weeks
ago") from timeutil; the model never subtracts timestamps itself.
Reference notes get None: "updated 3 weeks ago" means nothing for theory.
"""

from app.memory import memory_store as ms
from app.memory import timeutil

ALLOWED_SCOPES = ms.JOINT_SESSION_SCOPES  # v1 = joint sessions only

MAX_SNIPPET_RESULTS = 8
MAX_LIST_RESULTS = 50
MAX_DOCUMENT_CHARS = 4000  # guardrail so one huge document can't flood the context

_NOT_FOUND = "No document with id {}."


def _relative(updated_at: str, doc_scope: str) -> str | None:
    if doc_scope == ms.SCOPE_REFERENCE:
        return None
    return timeutil.humanize_delta(updated_at)


def _check_doc_type(doc_type) -> dict | None:
    if doc_type is not None and doc_type not in ms.ALL_TYPES:
        return {"error": f"Unknown doc_type '{doc_type}'. Valid: {sorted(ms.ALL_TYPES)}"}
    return None


def _can_read(doc: dict, couple_id: int) -> bool:
    if doc["doc_scope"] not in ALLOWED_SCOPES:
        return False
    return doc["doc_scope"] == ms.SCOPE_REFERENCE or doc["couple_id"] == couple_id


def search_memory(couple_id: int, query: str, doc_type: str | None = None) -> dict:
    """Keyword search; returns short snippets with doc_ids to fetch in full."""
    query = (query or "").strip()
    if not query:
        return {"error": "query must not be empty"}
    if err := _check_doc_type(doc_type):
        return err

    results = ms.search(query, couple_id, ALLOWED_SCOPES, doc_type=doc_type,
                        limit=MAX_SNIPPET_RESULTS)
    if not results:
        return {"results": [], "note": "No matches."}
    return {
        "results": [
            {
                "doc_id": r["doc_id"],
                "title": r["title"],
                "doc_type": r["doc_type"],
                "snippet": r["snippet"],
                "updated_at": r["updated_at"],
                "relative_time": _relative(r["updated_at"], r["doc_scope"]),
            }
            for r in results
        ]
    }


def get_memory_document(couple_id: int, doc_id) -> dict:
    """Full text of one document, if this couple may read it."""
    try:
        doc_id = int(doc_id)  # models sometimes send "12" instead of 12
    except (TypeError, ValueError):
        return {"error": f"doc_id must be an integer, got {doc_id!r}."}

    doc = ms.get_document(doc_id)
    if not doc or not _can_read(doc, couple_id):
        return {"error": _NOT_FOUND.format(doc_id)}  # same message either way

    content = doc["content"]
    truncated = len(content) > MAX_DOCUMENT_CHARS
    if truncated:
        content = content[:MAX_DOCUMENT_CHARS] + "\n[...truncated...]"

    result = {
        "doc_id": doc["id"],
        "title": doc["title"],
        "doc_type": doc["doc_type"],
        "content": content,
        "truncated": truncated,
        "updated_at": doc["updated_at"],
        "relative_time": _relative(doc["updated_at"], doc["doc_scope"]),
    }

    if doc["doc_type"] in ms.LIVING_TYPES and doc["couple_id"] is not None:
        history = ms.get_document_history(doc["couple_id"], doc["doc_key"], doc["doc_scope"])
        if len(history) > 1:
            result["first_recorded_at"] = history[0]["created_at"]
            result["first_recorded_relative"] = timeutil.humanize_delta(history[0]["created_at"])
            result["version_count"] = len(history)
        if doc["superseded"]:
            # New vs v0.4: say so when the model fetched an old version.
            result["superseded"] = True
            result["latest_doc_id"] = history[-1]["id"]

    return result


def list_memory_documents(couple_id: int, doc_type: str | None = None) -> dict:
    """Latest version of each document this couple may read, newest first."""
    if err := _check_doc_type(doc_type):
        return err
    docs = ms.list_documents(couple_id, ALLOWED_SCOPES, doc_type=doc_type,
                             limit=MAX_LIST_RESULTS)
    return {
        "documents": [
            {
                "doc_id": d["id"],
                "title": d["title"],
                "doc_type": d["doc_type"],
                "updated_at": d["updated_at"],
                "relative_time": _relative(d["updated_at"], d["doc_scope"]),
            }
            for d in docs
        ]
    }


# Name -> function, same names the model will see as tools.
TOOL_FUNCTIONS = {
    "search_memory": search_memory,
    "get_memory_document": get_memory_document,
    "list_memory_documents": list_memory_documents,
}


def dispatch(couple_id: int, tool_name: str, tool_input: dict) -> dict:
    """v0.4-compatible entry point: run a tool by name. The curator and
    tests use it; in the graph, ToolNode routes by name instead."""
    fn = TOOL_FUNCTIONS.get(tool_name)
    if fn is None:
        return {"error": f"Unknown tool: {tool_name}"}
    try:
        return fn(couple_id, **(tool_input or {}))
    except TypeError as e:  # unexpected or missing arguments from the model
        return {"error": f"Bad arguments for {tool_name}: {e}"}
