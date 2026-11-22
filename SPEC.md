# Couples Agent — LangGraph Tech Spec

Nov 20, 2026 · @Harry DAVIES

## Purpose and scope

Build a couples-support agent in LangGraph that reuses the proven architecture of `counselling_agent` v0.4 (Python, Anthropic SDK, SQLite), with a small eval suite from day one. It has two jobs: give me and my partner a relationship-skills companion we can use together, and give me honest, hands-on LangGraph/LangChain experience for AI engineering roles.

**In scope (v1, this weekend):**

- Joint sessions: both partners at one keyboard, each turn tagged with who is speaking.
- Conversation grounded in relationship-skills psychoeducation, with the agent staying neutral between partners.
- One deterministic, guided couples exercise (a structured conflict debrief).
- The two-layer safety gate, extended for couples-specific risk (intimate partner violence and coercive control).
- Typed, versioned memory scoped to the couple, with retrieval through tools.
- An end-of-session curator.
- An eval harness with a labelled safety set and a response rubric.

**Out of scope for v1 (designed for, not built):** individual sessions with private memory, a web UI, embedding search, encryption at rest and deployment.

**Non-goals:** this is not couples therapy, it does not diagnose either partner or the relationship, and it does not adjudicate who is right. Where abuse or coercive control is indicated, the agent stops mediating altogether (see Couples-specific design).

## Fit to an AI engineering role

The build covers most of the ad's "What you'll do" list. SQL and pipelines are only partly covered, and DBT and Airflow not at all; speak to your own experience for those.

| What the ad asks for | Where this build shows it | What to say |
| --- | --- | --- |
| Production-grade GenAI apps with LangChain/LangGraph | `StateGraph`, conditional edges, `ToolNode`, `interrupt()`, SQLite checkpointer | Why a graph: safety gates every path, and exercises pause for human input |
| LLM prompts vs workflows vs agents, and when to use each | Code for safety and exercise steps, a fixed workflow for routing, an agent loop only for memory retrieval | Use the least autonomy that does the job |
| Evaluation frameworks and observability metrics | Safety precision/recall, LLM-judge rubric, LangSmith traces, Prometheus metrics with Grafana alerts | Quote your before/after numbers |
| Cloud and containerisation (Docker) | Dockerfile `for the app + docker compose for Prometheus and Grafana`; OpenTelemetry path to Application Insights | Prior Azure and Foundry exposure, plus the open-source vs Azure mapping |
| Develop APIs | `Not built here, on purpose` | Your existing API work: custom Workato actions integrating ACC/BIM360, Azure Functions, Flask apps. You'd wrap this graph in FastAPI the same way |
| SQL, DBT, orchestration | Hand-written SQL and an FTS5 index in the memory store; no DBT or Airflow | Talk from your own pipeline work; don't stretch this project to cover it |
| Testing, code quality, version control, code reviews | \~88 ported unit tests, graph tests with a fake model, `ruff`, GitHub Actions CI, reviewing every Claude Code diff | Tests and evals run offline in CI |
| Experimentation / A/B testing (nice to have) | Prompt A vs prompt B on the same eval set | Paired comparison; at \~15 cases it is directional, not significant, and you'd say so |
| Communicating risks and trade-offs | README trade-offs: FTS5 vs embeddings, regex recall bias, trace privacy, IPV in joint mode | Lead with one trade-off you made and why |
| Client-facing, consultative | Not in the build | Prior advisory work: problem framing, POCs, vendor projects |

## Learnings from counselling\_agent v0.4

These design decisions worked and are requirements here, not suggestions. Port the behaviour; restructure the code around LangGraph.

| # | Decision (keep) | Where it lives in v0.4 | How it carries into LangGraph |
| --- | --- | --- | --- |
| 1 | Two-layer safety: deterministic regex gate before the model, plus model judgement via the prompt. Biased to recall: a false positive costs one extra message, a false negative costs far more. | `agent/safety.py`, `system_prompt.py` SAFETY HANDOFF | `safety_gate` node runs first on every turn; crisis resources are emitted by code, never depend on the LLM call succeeding |
| 2 | Exercise steps are fixed Python sequences; the model only adds warmth and the closing reflection | `agent/exercises.py`, `main.run_exercise` | An exercise subgraph that walks steps with `interrupt()`; each answer still passes the safety gate |
| 3 | Memory = typed documents: living (superseded, versioned), append-only (events), reference (shared theory) | `agent/memory_store.py` | Keep the store and schema as-is; expose it as LangChain tools |
| 4 | Only a short `profile_overview` is injected; everything else is retrieved on demand via search, then fetch | `memory_tools.py`, `client.set_user` | Same three tools (`search_memory`, `get_memory_document`, `list_memory_documents`) bound to the chat model; a `ToolNode` executes them |
| 5 | Agentic curator that reads existing documents before writing, then must finish with one forced structured tool call; round cap; never crashes the exit path | `agent/curator.py` | A curator subgraph run at session end; final step forced with `tool_choice` / structured output |
| 6 | Temporal grounding: all date maths in Python; the model receives finished phrases ("3 weeks ago") and the current time on every call | `agent/timeutil.py`, `compute_temporal_context` | `load_context` node computes it into state; tool results keep `relative_time` |
| 7 | Data minimisation: no raw transcripts in the DB; safety events stored as category + timestamp only; `/memory` transparency; `/forget` deletes all versions | `db.py`, `main.py` | Unchanged. Note the LangGraph checkpointer stores messages, so it needs a retention rule (see Graph design) |
| 8 | Licensing discipline: only administer public-domain instruments; log external scores without reproducing items | `assessments.py`, README | Same rule for any couples questionnaire |
| 9 | Crisis numbers verified against healthdirect.gov.au with a date | `resources/crisis_resources.py` | Reuse; re-verify before use |
| 10 | Commands that need no API key still work offline; user isolation enforced inside tool dispatch | `main.py`, `memory_tools.dispatch` | Keep both; isolation now by `couple_id` |
| 11 | FTS5 keyword search over embeddings, chosen deliberately for explainability at personal-data scale | `memory_store.search` | Keep for v1; embeddings remain a drop-in upgrade behind `search()` |

**Gaps to fix in the rebuild:**

- The output-side safety backstop is described in `safety.py` but never called on the model's replies. Wire it in as an `output_check` node.
- `.env.example` contains what looks like a live API key. Revoke and rotate it, and ship a placeholder only.
- Streaming was dropped once tools were added. LangGraph's `stream_mode="messages"` brings it back for the final reply.
- No eval harness exists yet beyond unit tests. That was roadmap items 1 and 2, now built in from the start.

## Tech stack and repo layout

Python 3.11, LangGraph 1.x (the latest is 1.2.14, released 6 October 2026), Claude via `langchain-anthropic`, SQLite for both memory and graph checkpoints. Pin exact versions when the project is scaffolded.

| Layer | Choice | Why |
| --- | --- | --- |
| Orchestration | `langgraph` (`StateGraph`, conditional edges, `interrupt`, `Command`) | Named in the role; the safety-first routing maps naturally onto a graph |
| Model | `langchain-anthropic` `ChatAnthropic`, model id from `.env` | Same model family as v0.4, so prompt behaviour carries over |
| Tools | `@tool`-decorated wrappers over the existing `memory_store`; `ToolNode` from `langgraph.prebuilt` | Reuses proven retrieval; shows the LangChain tool abstraction |
| Persistence (graph) | `langgraph-checkpoint-sqlite` `SqliteSaver` | Required for `interrupt()`; one `thread_id` per session |
| Persistence (memory) | Existing SQLite + FTS5 store, ported with a `couple_id` scope | Proven; already tested |
| Auth | `argon2-cffi`, as in v0.4 | Unchanged |
| Evals | `pytest` harness + JSONL datasets; optional LangSmith tracing and experiments if a free key is set | Runs offline and in CI; LangSmith is a talking point, not a dependency |
| Interface | CLI (same command style as v0.4), also runnable in Docker; it serves Prometheus metrics on a local port | The CLI is fastest for you to use; Docker covers the ad's "containerisation" ask. An API layer is skipped because you've already built APIs elsewhere. A web front end is v2 |

```
couples_agent/
  app/
    graph.py            # StateGraph wiring, compile(checkpointer=...)
    state.py            # CoupleState TypedDict + reducers
    nodes/
      context.py        # load_context: temporal grounding, profile_overview
      safety.py         # safety_gate + output_check (port of v0.4 safety.py)
      router.py         # route: crisis | exercise | converse
      converse.py       # chat model bound to memory tools
      exercises.py      # deterministic exercise subgraph using interrupt()
      crisis.py         # crisis handoff (resources by code, then model reply)
      curator.py        # end-of-session curator subgraph
    tools/memory.py     # @tool wrappers over memory_store
    memory/             # ported memory_store.py, db.py, timeutil.py, auth.py
    prompts.py          # system prompts (ported + couples persona)
    observability.py    # LangSmith config, Prometheus metrics, timing decorator
    resources/crisis_resources.py
  cli.py                # login, commands, runs the graph; serves /metrics
  Dockerfile
  docker-compose.yml    # prometheus + grafana (scrape the CLI's metrics port)
  ops/                  # prometheus.yml, grafana dashboard + alert rules
  .github/workflows/ci.yml  # ruff + pytest + offline evals on push
  evals/
    datasets/safety.jsonl, routing.jsonl, rubric_cases.jsonl
    run_evals.py        # precision/recall + LLM-judge scores, writes a report
  tests/                # ported unit tests + graph tests with a fake model
  CLAUDE.md             # points Claude Code at this spec
  .env.example          # placeholders only
```

## Graph design

One graph runs once per user turn. Safety is the first node, not a side check, so no route reaches the model without passing it.

&#91;embedded content: per-turn graph · 8 nodes, safety first\]

A flagged message never reaches the normal chat path. Exercise answers re-enter `safety_gate` before they are stored, and the curator runs separately at `/quit`.

**State (`CoupleState`):**

```python
class CoupleState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    couple_id: int
    session_id: int
    speaker: Literal["partner_a", "partner_b"]   # who typed this turn
    partner_names: dict[str, str]                 # {"partner_a": "Jordan", ...}
    now_text: str                                 # computed in Python
    temporal_context: str | None                  # "session #4, last one 12 days ago"
    profile_overview: str | None
    safety: SafetyResult | None                   # set by safety_gate
    route: Literal["crisis", "exercise", "converse"] | None
    active_exercise: str | None
    exercise_answers: dict                        # {(speaker, field): answer}
    joint_mode: bool                              # True in v1; drives IPV display rules
```

**Nodes and edges:**

| Node | Does | Next |
| --- | --- | --- |
| `load_context` | On the session's first turn, computes `now_text`, `temporal_context` and `profile_overview` (port of `compute_temporal_context`). Refreshes `now_text` every turn | `safety_gate` |
| `safety_gate` | Deterministic regex (ported patterns plus couples patterns). Logs category only. Optional second layer: a cheap LLM classifier run in parallel, OR-ed with the regex | conditional → `crisis` if triggered, else `router` |
| `router` | Deterministic first: an active exercise or a `/command` decides the route. Otherwise `converse` | `exercise` or `converse` |
| `converse` | Chat model with memory tools bound; couples persona prompt | `tools` if tool calls (max 4 rounds, as in v0.4), else `output_check` |
| `tools` | `ToolNode` executing `search_memory` / `get_memory_document` / `list_memory_documents`, scoped by `couple_id` | back to `converse` |
| `exercise` | Subgraph. For each fixed step it calls `interrupt({prompt, speaker})`; the CLI resumes with `Command(resume=answer)`. Each answer re-enters `safety_gate`. Ends with a model reflection | `output_check` |
| `crisis` | Code emits verified resources first, then the model replies under SAFETY HANDOFF rules. In joint mode, abuse category uses the IPV protocol (next section) | END |
| `output_check` | Runs the regex on the model's reply (the backstop v0.4 never wired) and blocks diagnosis or side-taking phrases flagged by a small rule list | END |

**Session end (separate invocation):** `/quit` runs the curator subgraph. It loops through the memory tools, then is forced on its last round to call `propose_memory_updates`, with the same schema as v0.4 plus `doc_scope`. `apply_proposal` writes to the store. Failures are swallowed, so exit is never blocked.

**Checkpointer retention:** `SqliteSaver` persists message history per `thread_id`, which v0.4 deliberately never stored. After the curator runs, delete that session's thread (`checkpointer.delete_thread(thread_id)`) so the data-minimisation guarantee still holds.

## Couples-specific design

Three things change when there are two people in the session: who is speaking, whose memory it is, and what safety means when the person at risk may be sitting next to the risk.

**Session mode (v1 = joint only).** Both partners log in once as a couple account. Each turn starts with a speaker switch (`/a`, `/b`, or a prefix such as `Jordan:`) and sets `state.speaker`. The prompt receives both names and who is speaking. Individual sessions are v2.

**Neutrality is a hard rule, not a tone.**

- Reflect each partner's view in their own words before offering anything.
- Address both partners. If one has had three turns in a row, invite the other.
- Never assign blame, never say who is right, and never diagnose either person or the relationship.
- `output_check` flags replies containing side-taking or diagnostic phrasing, and the rubric eval scores neutrality.

**Memory scopes.** Memory documents carry `couple_id` and a `doc_scope`:

| Scope | v1 | Retrieved in joint sessions | Examples |
| --- | --- | --- | --- |
| `couple` | Yes | Yes | `profile_overview`, `shared_goals` (marriage, family plans), `agreements`, `conflict_patterns`, `appreciations`, `session_note`, `exercise_record` |
| `partner_private` | v2 | Never, unless that partner runs `/share <doc>` | Individual reflections from solo sessions |
| `reference` | Yes | Yes | New relationship-skills notes written in our own words, plus the ported CBT and distortions notes |

Tool dispatch enforces scope, not the prompt. In v1 the retrieval tools simply never return `partner_private` documents, so the v2 privacy model can be added without changing the tool interface.

**IPV and coercive-control protocol (the most important design decision).**

Joint relationship work is generally considered unsuitable where there is violence or coercive control. In joint mode, a crisis screen could also be read by the partner who poses the risk. So when `safety_gate` returns `abuse` and `joint_mode` is true:

1. Stop mediation for the session. Do not continue the exercise, and do not ask either partner to explain or "see the other side".
2. Show the full general resource list (Lifeline, 1800RESPECT, 000) in neutral wording. Do not single out the disclosure, quote it back, or name a partner.
3. Suggest each partner talk to a professional individually, framed as standard practice rather than as a response to what was said.
4. Log the category only. The curator is told a safety flag occurred but writes nothing about it into `couple`-scoped memory, because both partners can read that memory with `/memory`.

Add couples-specific regex patterns with recall bias, for example "checks my phone", "won't let me see my friends", "controls the money", "scared of how he'll react", "scared of how she'll react". This protocol is a design proposal: get it reviewed by someone with family-violence training before anyone outside the two of you uses it.

**Guided exercise (v1): Conflict debrief.** Fixed steps alternate between the partners, and the model only adds warmth and the closing reflection:

1. Partner A: what happened, as facts only.
2. Partner B: what happened, as facts only.
3. Each partner: what I felt, and how strongly (0-100).
4. Each partner: what I needed in that moment.
5. Each partner: my part in how it went.
6. Each partner: one specific request for next time.
7. Together: one agreement to try. This is saved to the `agreements` document by the curator.

The exercise is generic and written in our own words, not taken from a branded programme. Questionnaires follow v0.4's licensing rule: the Dyadic Adjustment Scale is commercially licensed, so it is log-only. Check any other instrument's licence before reproducing its items.

## Reference material: my partner's picks

My partner chose two books as the app's guiding frameworks: *Hold Me Tight: Seven Conversations for a Lifetime of Love* (Dr Sue Johnson, 2008), the public-facing book on Emotionally Focused Therapy (EFT), and *The Seven Principles for Making Marriage Work* (John Gottman and Nan Silver, 1999). They enter the app as **reference documents written in our own words**, using the same `scope = 'reference'` mechanism as v0.4's CBT notes.

**Rules (same discipline as the DAS/YSQ licensing note):**

- Concepts, not text. Claude Code does not need, and must not use, the books' text. It seeds short notes that summarise ideas in plain language, with the book cited in each note's title.
- No reproduced exercises, quizzes or questionnaires from either book. Where the app has an exercise, it is our own design, informed by the ideas.
- The agent says it draws on ideas from these books. It never claims to deliver EFT or Gottman Method therapy, which are clinician-delivered.
- My partner reviews the notes against the books. That is a direct contribution to the build, and the README credits it.

| `doc_key` | What the note covers (own words) | How the agent uses it |
| --- | --- | --- |
| `reference:eft-attachment-basics` | A relationship as an attachment bond. Partners need each other to be accessible, responsive and engaged (A.R.E.). Conflict often hides a fear of disconnection | Looks under the complaint for the need; reflects "you both want to feel close" rather than refereeing the content |
| `reference:eft-negative-cycles` | Johnson's "Demon Dialogues": recurring negative cycles, e.g. pursue-withdraw (one pushes for connection, the other pulls back) | Names the cycle as the shared enemy, not either partner. This reinforces the neutrality rule |
| `reference:eft-seven-conversations` | The seven conversations, from recognising the cycle and finding raw spots, through revisiting a rocky moment and forgiving injuries, to keeping the bond alive | A map of where a conversation sits; suggests the next conversation the couple might try |
| `reference:gottman-seven-principles` | Love maps, fondness and admiration, turning toward bids, accepting influence, solving solvable problems, overcoming gridlock, creating shared meaning | Suggests small, concrete habits; separates solvable problems from perpetual ones |
| `reference:gottman-four-horsemen` | Criticism, contempt, defensiveness, stonewalling, and the commonly taught antidotes: gentle start-up, appreciation, taking responsibility, self-soothing | Gently names a horseman when it shows up, and offers what the antidote could sound like |
| `reference:gottman-repair-and-startup` | Softened start-up and repair attempts as conflict skills | Used in the conflict-debrief reflection |

**What changes in the build because of these books:**

- **Conflict debrief exercise** gains two steps after step 5: "Which pattern were we in?" (the cycle, in their own words) and "Did any horsemen show up, and what would the antidote have sounded like?". The "what I needed" step is framed as the need underneath the reaction.
- **Memory:** `conflict_patterns` records the couple's own name for their cycle. `shared_goals` holds shared-meaning items such as marriage and family plans.
- **Optional v1.1 exercise, "Love map check-in":** a rotating bank of questions the two of you write yourselves. This is a natural home for your Monday date-night questions.
- **Rubric eval** gains one criterion: does the reply frame conflict as a shared pattern rather than one partner's fault?

## Evals

The weekend must ship evals 1 and 2. Evals 3 and 4 are stretch goals. Every eval runs from `python evals/run_evals.py` and writes a dated Markdown report, so before/after numbers can be quoted.

| # | Eval | Dataset | Metric | Weekend target |
| --- | --- | --- | --- | --- |
| 1 | Safety gate | `safety.jsonl`: \~60 hand-labelled messages: direct risk, indirect phrasing, sarcasm and idiom ("this week is killing me"), discussing vs disclosing, and couples/IPV phrasing | Recall and precision per category, regex alone vs regex + LLM classifier | Build it; report the numbers whatever they are. Recall on `self_harm` and `abuse` matters most |
| 2 | Response rubric (LLM-as-judge) | `rubric_cases.jsonl`: \~15 two-speaker conversation snippets | Judge scores 1-5 on neutrality, one question at a time, no diagnosis, and warmth; plus pass/fail "handed off correctly" on crisis cases | Build it; run once on the base prompt, once after one prompt change |
| 3 | Routing | `routing.jsonl`: \~20 turns with expected route | Accuracy | Stretch |
| 4 | Retrieval | Seeded memory + \~15 (turn → doc that should be found) pairs | Hit rate in the top 3 for `search_memory` | Stretch; this is what would justify embeddings |

**Rules for the harness:**

- Graph tests use a fake chat model so they run without an API key, as v0.4's tests mock the client.
- The judge model is pinned and its prompt is versioned in the repo.
- Hand-check five judge scores against your own reading before trusting the numbers.
- Never put real session content into eval datasets. Write synthetic cases.
- If `LANGSMITH_API_KEY` is set, also log runs as a LangSmith experiment. Without it, everything still works locally.

## Observability and monitoring

Instrument the agent once and send the data to two places. Traces go to LangSmith, showing every node, prompt, tool call, retrieved memory document, token count and latency. Content-free metrics go to Prometheus, with a Grafana dashboard and alerts. Both are built on OpenTelemetry, so the same LangGraph spans can also go to Azure Application Insights and the Foundry traces view, a common enterprise stack.

| Layer | What it answers | Built here (open source) | Azure equivalent |
| --- | --- | --- | --- |
| Agent framework | How control flows | LangGraph | Microsoft Agent Framework |
| Traces | What happened in this run, and why: nodes, prompts, tool calls, retrieved documents, tokens, latency | LangSmith (`LANGSMITH_TRACING=true`), with built-in LangGraph integration | Foundry tracing stored in Application Insights; Microsoft's OpenTelemetry distro also traces LangChain/LangGraph agents hosted outside Foundry ([Microsoft Learn](https://learn.microsoft.com/en-us/azure/foundry/observability/how-to/trace-agent-framework)) |
| Metrics | How the system is doing over time | `prometheus_client` `/metrics` endpoint scraped by Prometheus | Azure Monitor metrics / Application Insights |
| Dashboards and alerts | When a human needs to look | Grafana dashboard + 2 alert rules | Azure Monitor alerts, Workbooks or Azure Managed Grafana |
| Evals over time | Is quality improving | `run_evals.py` results logged as LangSmith experiments | Foundry evaluations |

**Metrics (labels never contain message text or user identity):**

- `agent_turns_total{route}`: crisis / exercise / converse.
- `agent_node_latency_seconds{node}`: a histogram, so p50/p95 per node.
- `llm_tokens_total{direction,model}` and `llm_cost_aud_total` (estimated from token prices in config).
- `safety_triggers_total{category,layer}`: regex vs LLM classifier, which shows how much each layer contributes.
- `tool_calls_total{tool}` and `retrieval_empty_total`: searches that found nothing, an early signal for the embeddings question.
- `agent_errors_total{node}`.

**Alerts (Grafana):** API error rate above 5% over 10 minutes, and estimated daily cost above a set amount. In a real product you would add a third: a spike in safety triggers routed to a human, but that is a product design point, not something to build for two users.

**Privacy rule.** By default LangSmith traces contain the full conversation, which conflicts with this app's data-minimisation design. Trace freely during eval runs on synthetic data. For real sessions, set `LANGSMITH_HIDE_INPUTS=true` and `LANGSMITH_HIDE_OUTPUTS=true` (or a masking anonymiser) so only structure, timing and token counts leave the machine ([LangSmith docs](https://docs.langchain.com/langsmith/mask-inputs-outputs)). The Azure path has the same control: leave `enable_sensitive_data` off. That trade-off makes a good interview answer in its own right.

**Scope:** LangSmith tracing is a must (about 15 minutes of setup). Prometheus + Grafana via `docker compose` is a Sunday must, because the ad lists Docker and observability metrics. It needs Docker Desktop on Windows. Exporting to Application Insights is optional, depending on whether you have an Azure subscription to use; otherwise keep it as a talking point.

## Weekend build plan and Claude Code handoff

The goal is to finish Sunday night with a graph that runs end to end, two evals with numbers, and enough understanding to draw the graph on a whiteboard from memory.

- [done] **Fri evening (1 h):** Rotate the API key. Create `~/Repos/couples_agent`. Export this doc to Markdown as `SPEC.md`. Add `CLAUDE.md` (below). Run git init and push to a private GitHub repo.
- [done] **Sat morning:** Claude Code ports `memory/` from v0.4 (store, db, timeutil, auth, crisis resources) with `couple_id`/`doc_scope`, and its tests pass.
- [done] **Sat midday:** `state.py`, `graph.py` with `load_context → safety_gate → router → converse ⇄ tools → output_check`, plus `SqliteSaver`. A CLI chat works. Turn on LangSmith tracing and read one full trace node by node.
- [done] **Sat afternoon:** `crisis` node with the IPV protocol; the conflict-debrief exercise subgraph with `interrupt()`; the curator subgraph; thread deletion after curation; seed the reference notes once they have been reviewed.
  - *As built:* the exercise is a per-turn state machine, not `interrupt()` (a resumed graph would skip the safety gate at START); the curator is one structured-output call with documents fetched by code. Reference notes seeded straight away; the review of `docs/reference_notes.md` against the books comes later.
- [done] **Sun morning:** Write `safety.jsonl` and `rubric_cases.jsonl` yourself (writing them is the learning). Build `run_evals.py`. Get first numbers.
- [done] **Sun afternoon:**
  - *As built:* did BOTH the LLM classifier and the regex revision, measured on a 40-case held-out set (regex v2: 20% to 20% recall, i.e. overfitted to the dev set; + classifier: 100%). Alerts are Prometheus rules shown in Grafana, not Grafana-managed rules (plain YAML, reviewable). README before/after still to write.
- **Sun afternoon (original plan):** Change one thing (add couples regex patterns, or the LLM classifier) and re-run. Record the before/after in the README. Then expose metrics from the CLI (prometheus\_client start\_http\_server), add a Dockerfile for the app, and bring up Prometheus and Grafana with docker compose: one dashboard, two alert rules. Add the CI workflow.
- [ ] **Sun evening:** Walk through the graph out loud. Explain each node without notes.

**If time runs short, cut from the bottom of this list:** the graph with LangSmith tracing, then the safety eval, then Docker + metrics, then the rubric eval, then the Grafana alerts, then the CI workflow.

**`CLAUDE.md` for the new repo:**

```markdown
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
```

**First prompt to Claude Code:** "Read CLAUDE.md and SPEC.md, then propose a step-by-step plan for Saturday morning's port before writing any code." Ask it to stop after each step so you read the diff. The point of the weekend is that you can explain it, not just that it exists.

## Open questions and interview talking points

**Open questions (defaults are assumed in this spec until answered):**

- Joint sessions only, or individual sessions too? Default: joint only.
- One shared device and login, or each partner's own login on a shared couple record? Default: one couple login.
- Should private individual sessions (v2) ever exist? It changes the privacy model, so decide before real use.
- Who reviews the IPV protocol and crisis wording before anyone else uses it?

**Interview talking points (each is something you can now say you built and why):**

- **Why a graph, not a chain:** safety has to gate every path, and exercises need to pause for human input. Conditional edges and `interrupt()` make both explicit and testable.
- **Deterministic where it matters:** regex safety and fixed exercise steps run in code; the LLM does language, not control flow.
- **Memory design:** typed, versioned documents with tool-based retrieval beat one rolling summary at lifetime scale. FTS5 was chosen over embeddings on purpose, with a retrieval eval as the way to decide.
- **Evals:** precision/recall on a labelled safety set, plus an LLM-as-judge rubric with hand-checked scores. Quote the before/after numbers you got.
- **Responsible design:** data minimisation (checkpointer threads deleted), scope enforced in code rather than the prompt, licensing checks, and the IPV protocol in joint sessions.
- **Observability:** traces (LangSmith) answer "why did this run do that"; metrics and alerts (Prometheus/Grafana) answer "is the system healthy". Both are OpenTelemetry-based, so the same instrumentation maps to Application Insights and Foundry tracing on Azure. Inputs and outputs are hidden for real sessions.
- **Honest limits:** not clinically reviewed, regex recall is a floor, the judge model has its own bias, and it is single-machine SQLite.
