# couples_agent
Read SPEC.md first; it is the source of truth. Build in the order of its weekend plan.
Reference implementation (read-only, do not modify): ~/Repos/counselling_agent
- Port behaviour from it, especially safety.py, memory_store.py, memory_tools.py,
  curator.py, timeutil.py. Keep its tests passing after the port.
Rules:
- Safety gate is the first node on every path; crisis resources are emitted by code.
- All date arithmetic in Python (timeutil); never ask the model to compute time gaps.
- No raw transcripts persisted; delete the LangGraph thread after the curator runs.
- Never commit .env or real keys; .env.example holds placeholders only.
- Explain each LangGraph concept you use in a short comment where it first appears
  (StateGraph, reducers, conditional edges, ToolNode, interrupt/Command, checkpointer).
- Run `python -m pytest` and `python evals/run_evals.py` before saying a step is done.