# couples_agent

A relationship-skills companion for two partners in one session, built on LangGraph. It's a weekend learning project: it is **not therapy, a clinician or a crisis service**, and it never takes sides or diagnoses either partner.

It reuses the architecture of an earlier single-user project (`counselling_agent` v0.4): typed, versioned memory in SQLite; a deterministic safety layer; Python-only date maths. It moves the control flow into a LangGraph state graph.

## Run it

```powershell
python -m venv .venv
.venv\Scripts\pip install -e ".[dev]"
copy .env.example .env          # then add your ANTHROPIC_API_KEY
.venv\Scripts\python cli.py
```

Without a key, the app runs in offline mode: the safety gate and local commands still work, but there are no model replies. Type `/help` in the CLI for commands (`/a` and `/b` switch speaker, `/debrief`, `/memory`, `/crisis`, `/quit`).

| Task | Command |
|---|---|
| Tests (offline, fake model) | `python -m pytest` |
| Lint | `ruff check .` |
| Safety eval (offline, free) | `python evals/run_evals.py` |
| + LLM classifier and rubric (live, costs money) | `python evals/run_evals.py --classifier --rubric --label name` |
| Held-out safety set | `python evals/run_evals.py --dataset safety_holdout.jsonl --classifier` |
| Prometheus + Grafana | `docker compose up -d`, then Grafana at http://localhost:3000 |
| The CLI in Docker | `docker compose run --rm --service-ports app` |

## How a turn works

The graph runs once per turn. A SQLite checkpointer carries state between turns, under one thread per session.

```
START -> load_context -> safety_gate --(flagged)--> crisis --(abuse, joint session)--> END
                                     |                     \--> output_check
                                     \--(clear)--> router -> converse <-> tools
                                                      |          \-> output_check -> END
                                                      \-> exercise --(last answer)--> reflect -> output_check
```

- **safety_gate** is the only way past START. A regex runs first. If the regex finds nothing, an LLM classifier also checks the message. A flag from either one sends the turn to `crisis`, where **code** shows the Australian crisis services before the model writes a short handoff reply. If one partner discloses abuse in a joint session, the session ends with no model call (the IPV protocol).
- **converse** answers using memory tools (`ToolNode`). The model never sees `couple_id`; tool scope is enforced in code.
- **exercise** is a 13-step conflict debrief written as a per-turn state machine. The questions are code, and the model writes one reflection at the end. Because each answer is a new turn, every answer passes through the safety gate.
- **output_check** replaces a reply that takes sides or diagnoses, and adds crisis resources if the reply itself trips the safety regex.
- At `/quit`, a **curator** graph turns the session into memory updates using structured output (*the model proposes, code disposes*). Then the session's thread is deleted, so no raw transcript is kept.

A rule that runs through the whole design: **code decides anything that must be right every time** (safety resources, scope, dates, step order, deletion), **and the model adds language and warmth.**

## Evals: before and after

Datasets are synthetic: the couple is "Sam and Alex", and no real session content is used. They were written without looking at the regex.

### Safety gate

The **dev set** (60 messages, 25 risk) is where the patterns were tuned. The **held-out set** (40 messages, 20 risk) was written afterwards and not read until the final runs.

| Gate | Dev recall | Dev precision | **Held-out recall** | **Held-out precision** |
|---|---|---|---|---|
| Original regex (v0.4) | 0% | 0% | 20% | 100% |
| Revised regex (v2, fitted to the dev misses) | 80% | 100% | **20%** | 100% |
| Original regex + LLM classifier | 100% | 83% | 100% | 100% |
| Revised regex + LLM classifier (shipped) | n/a | n/a | **100%** | **100%** |

*Recall:* of the messages that really carried risk, how many were flagged. *Precision:* of the messages flagged, how many really carried risk.

What this shows:

- **The revised regex was overfitted.** It went from 0% to 80% on the dev set it was written from, and stayed at 20% on held-out data. Matching keywords doesn't generalise to how people actually talk.
- **The LLM classifier is what generalises.** On 20 held-out risk messages, 100% recall is consistent with a true recall above about 85%, because the sample is small.
- **The regex stays anyway, as a floor.** It works during an API outage, can't be talked out of a flag by a prompt injection, is deterministic and readable, and costs nothing. The classifier can **add** a flag and never remove one. If the classifier call fails, the gate falls back to regex only.
- **Discussing isn't disclosing.** The v0.4 regex flagged people talking *about* suicide or domestic violence, which would show crisis resources to both partners, and in the abuse case would end the session. The revised patterns need a first-person form.

### Response rubric (LLM judge)

15 two-speaker cases, run on `claude-sonnet-5-5` with an LLM judge scoring 1 to 5. Code checks the facts the judge can't see: whether resources were shown, how many questions the reply asks, and whether `output_check` replaced the reply.

| | Neutrality | One question | No diagnosis | Warmth | Crisis handoff |
|---|---|---|---|---|---|
| Baseline (regex only) | 4.47 | 4.13 | 4.87 | 3.93 | **1/3** |
| + LLM classifier | 4.53 | 3.93 | 4.93 | 3.93 | **3/3** |

The baseline's two handoff failures were the most important finding of the weekend. The model recognised the risk, but the regex hadn't fired, so the model pointed the couple to "the services on your screen" when none were shown. In one run it gave US and UK hotline numbers to an Australian couple. The model can recognise risk but can't show resources, and code can show resources but couldn't recognise the risk. The classifier closes that gap, and the numbers still always come from code.

A note on method: the judge passed all three handoffs when the harness wrongly told it resources had been shown. It needs the facts from code. With one sample per case, differences of about 0.2 in the means are noise.

Reports are in `evals/reports/`. CI runs the offline safety eval on every push and fails if dev-set regex recall drops below 80%. The gate uses the dev set only: gating on the held-out set would gradually turn it into a second dev set.

## Observability

- **LangSmith traces** (set by environment variables only; see `.env.example`) answer "why did this run do that?": every node, prompt, tool call and token count. For real sessions, set `LANGSMITH_HIDE_INPUTS` and `LANGSMITH_HIDE_OUTPUTS` to `true`.
- **Prometheus metrics** answer "is the system healthy?". They're collected by one LangChain callback handler (`app/observability.py`) and served at `:8000/metrics`. They cover turns by route, per-node latency, tokens and estimated AUD cost, safety flags **by layer** (regex, classifier, output), tool calls, and empty searches. Labels never contain message text or identity.
- **Grafana** has a provisioned dashboard and two alert rules: model API errors above 5% over 10 minutes, and estimated spend above A$5 in 24 hours.

## Data and privacy

- No raw transcripts are kept. The LangGraph thread is deleted after the curator runs, and leftover threads are deleted at startup.
- The curator never writes safety-flagged turns into memory.
- Safety events log the **category only**, never the text.
- `.env` is never committed. The Docker image never contains secrets or databases.

## Credits

Riley chose the two books that guide the companion: *Hold Me Tight* (Sue Johnson) and *The Seven Principles for Making Marriage Work* (John Gottman and Nan Silver). Six short reference notes, written in our own words, are in `docs/reference_notes.md`. Riley will check them once she has read the books.

`SPEC.md` is the design spec and build plan; `CLAUDE.md` holds the working rules.
