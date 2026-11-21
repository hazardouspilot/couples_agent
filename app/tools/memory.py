"""
The memory tools the chat model can call: thin LangChain wrappers over
app/memory/retrieval.py, where the logic and the scope checks live.

LANGCHAIN CONCEPT: @tool
`@tool` turns a Python function into a Tool object with a name, a
description and a JSON schema the model reads. The name is the function
name, the description is the docstring, and the argument schema comes from
the type hints. v0.4 wrote that JSON schema by hand; here it's generated.

LANGGRAPH CONCEPT: InjectedState
`couple_id: Annotated[int, InjectedState("couple_id")]` means: when
LangGraph's ToolNode runs this tool, it fills `couple_id` from the graph
state, and the argument is REMOVED from the schema the model sees. So the
model cannot see, choose or fake which couple's memory it reads. Access
control lives in code, not in the prompt.
"""

from typing import Annotated, Literal

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from app.memory import retrieval

# Spelled out (not built from memory_store.ALL_TYPES) because a Literal
# must be static for the schema. tests/test_tools.py checks they match.
DocType = Literal[
    "agreements", "appreciations", "assessment", "conflict_patterns", "exercise_record",
    "profile_overview", "reference", "session_note", "shared_goals",
]


@tool
def search_memory(
    query: str,
    couple_id: Annotated[int, InjectedState("couple_id")],
    doc_type: DocType | None = None,
) -> dict:
    """Search this couple's saved memory (past session notes, shared goals,
    agreements, recurring conflict patterns, appreciations, completed
    exercises) and shared reference notes by keyword. Returns short snippets
    with a doc_id; use get_memory_document to read one in full. Use this when
    something the couple says might connect to an earlier session.

    Args:
        query: Keywords to search for, e.g. a topic, a person, a feeling.
        doc_type: Optional: restrict to one document type.
    """
    return retrieval.search_memory(couple_id, query, doc_type)


@tool
def get_memory_document(
    doc_id: int,
    couple_id: Annotated[int, InjectedState("couple_id")],
) -> dict:
    """Read the full text of one memory document by its doc_id (from
    search_memory or list_memory_documents). For documents that are updated
    over time (e.g. agreements), also says how long it has been tracked.

    Args:
        doc_id: The document's id.
    """
    return retrieval.get_memory_document(couple_id, doc_id)


@tool
def list_memory_documents(
    couple_id: Annotated[int, InjectedState("couple_id")],
    doc_type: DocType | None = None,
) -> dict:
    """List the couple's current memory documents (latest version of each),
    most recently updated first. Useful to see what's on file before
    searching or reading.

    Args:
        doc_type: Optional: restrict to one document type.
    """
    return retrieval.list_memory_documents(couple_id, doc_type)


MEMORY_TOOLS = [search_memory, get_memory_document, list_memory_documents]
