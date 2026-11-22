"""
Typed, versioned memory documents for a couple, plus shared reference notes.

Ported from counselling_agent v0.4 (agent/memory_store.py). The design is
unchanged; the document types and the scope model are new for couples.

Why typed documents instead of one rolling summary: a single paragraph
rewritten every session silently loses information over time. Splitting by
TYPE gives each kind of information its own space, and only what's relevant
to a conversation is retrieved (via tools) instead of everything being
stuffed into the prompt.

Three kinds of document:
  - LIVING (profile_overview, shared_goals, agreements, conflict_patterns,
    appreciations): the current understanding of one thing. An update
    SUPERSEDES the previous version (new row, version+1, old row flagged)
    rather than editing in place, so history is never lost.
  - APPEND-ONLY (session_note, exercise_record, assessment): events. Each
    gets its own row and its own unique doc_key; nothing is superseded.
  - REFERENCE: shared theory notes, not tied to any couple.

Scopes (doc_scope), and who enforces them:
  - couple: shared couple memory; both partners can read it via /memory.
  - partner_private: one partner only. Designed for v2; v1 never returns it.
  - reference: shared notes, couple_id is NULL.
  Enforcement is layered: the DB schema rejects inconsistent rows (db.py),
  every listing/search function here REQUIRES the caller to say which
  scopes it may see, and retrieval.py (tool dispatch) only ever passes
  JOINT_SESSION_SCOPES. The prompt is never relied on for privacy.

Retrieval is SQLite FTS5 keyword search, not embeddings: built into
sqlite3, fast at personal-data scale, and exact/explainable. Embeddings
would catch paraphrases ("my boss" vs "my manager") and can be swapped in
behind search() without changing its signature.
"""

import json
import re
import sqlite3

from app.memory import db

# --- document types ---

LIVING_TYPES = frozenset({
    "profile_overview",   # short always-injected summary of the couple
    "shared_goals",       # marriage, family plans, shared-meaning items
    "agreements",         # what they've agreed to try (e.g. from the conflict debrief)
    "conflict_patterns",  # recurring cycles, in the couple's own words
    "appreciations",      # things each values in the other
})

APPEND_ONLY_TYPES = frozenset({
    "session_note",
    "exercise_record",
    "assessment",         # log-only for licensed instruments (e.g. DAS)
})

REFERENCE_TYPE = "reference"

ALL_TYPES = LIVING_TYPES | APPEND_ONLY_TYPES | {REFERENCE_TYPE}

# --- scopes ---

SCOPE_COUPLE = "couple"
SCOPE_PARTNER_PRIVATE = "partner_private"
SCOPE_REFERENCE = "reference"
ALL_SCOPES = frozenset({SCOPE_COUPLE, SCOPE_PARTNER_PRIVATE, SCOPE_REFERENCE})

# What a joint session (both partners present, v1's only mode) may ever see.
JOINT_SESSION_SCOPES = frozenset({SCOPE_COUPLE, SCOPE_REFERENCE})


def _row_to_doc(row) -> dict:
    d = dict(row)
    d["metadata"] = json.loads(d["metadata_json"]) if d.get("metadata_json") else {}
    return d


def _check_type_and_scope(doc_type: str, doc_scope: str):
    if doc_type not in ALL_TYPES:
        raise ValueError(f"Unknown doc_type: {doc_type}")
    if doc_scope not in ALL_SCOPES:
        raise ValueError(f"Unknown doc_scope: {doc_scope}")
    # The 'reference' type and the 'reference' scope always go together.
    if (doc_type == REFERENCE_TYPE) != (doc_scope == SCOPE_REFERENCE):
        raise ValueError("doc_type 'reference' requires doc_scope 'reference' and vice versa")


def _scope_filter(couple_id, allowed_scopes) -> tuple[str, list]:
    """SQL condition (on table alias `md`) limiting rows to what the caller
    may see: reference docs if allowed, plus THIS couple's docs in the
    allowed non-reference scopes. Returns ("0", []) (matches nothing) if
    nothing is allowed."""
    allowed = frozenset(allowed_scopes)
    unknown = allowed - ALL_SCOPES
    if unknown:
        raise ValueError(f"Unknown doc_scope(s): {sorted(unknown)}")

    clauses, params = [], []
    owned = sorted(allowed - {SCOPE_REFERENCE})
    if owned and couple_id is not None:
        placeholders = ", ".join("?" for _ in owned)
        clauses.append(f"(md.couple_id = ? AND md.doc_scope IN ({placeholders}))")
        params += [couple_id, *owned]
    if SCOPE_REFERENCE in allowed:
        clauses.append("md.doc_scope = 'reference'")
    if not clauses:
        return "0", []
    return "(" + " OR ".join(clauses) + ")", params


# --- writes ---

def upsert_document(couple_id, doc_type: str, doc_key: str, title: str, content: str,
                    metadata: dict | None = None, doc_scope: str = SCOPE_COUPLE) -> dict:
    """Create a document, or a new version of an existing living document.
    Returns the new row as a dict. All in one transaction: the supersede,
    the insert and the FTS index row commit together or not at all."""
    _check_type_and_scope(doc_type, doc_scope)
    metadata_json = json.dumps(metadata) if metadata else None

    with db.get_conn() as conn:
        version = 1
        if doc_type in LIVING_TYPES:
            existing = conn.execute(
                "SELECT id, version FROM memory_documents WHERE couple_id = ? "
                "AND doc_scope = ? AND doc_key = ? AND superseded = 0",
                (couple_id, doc_scope, doc_key),
            ).fetchone()
            if existing:
                version = existing["version"] + 1
                conn.execute(
                    "UPDATE memory_documents SET superseded = 1 WHERE id = ?",
                    (existing["id"],),
                )

        now = db.now_iso()
        cur = conn.execute(
            "INSERT INTO memory_documents (couple_id, doc_scope, doc_type, doc_key, "
            "title, content, metadata_json, version, superseded, created_at, "
            "updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)",
            (couple_id, doc_scope, doc_type, doc_key, title, content, metadata_json,
             version, now, now),
        )
        doc_id = cur.lastrowid
        conn.execute(
            "INSERT INTO memory_fts (doc_id, couple_id, doc_scope, doc_type, title, "
            "content) VALUES (?, ?, ?, ?, ?, ?)",
            (doc_id, couple_id, doc_scope, doc_type, title, content),
        )
        row = conn.execute("SELECT * FROM memory_documents WHERE id = ?", (doc_id,)).fetchone()
        return _row_to_doc(row)


# --- reads ---

def get_document(doc_id: int) -> dict | None:
    """Raw fetch by id with NO access check. Callers that serve a user or
    the model must check couple_id/doc_scope themselves (retrieval.py does)."""
    with db.get_conn() as conn:
        row = conn.execute("SELECT * FROM memory_documents WHERE id = ?", (doc_id,)).fetchone()
        return _row_to_doc(row) if row else None


def get_latest_by_key(couple_id, doc_key: str,
                      doc_scope: str = SCOPE_COUPLE) -> dict | None:
    with db.get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM memory_documents WHERE couple_id IS ? AND doc_scope = ? "
            "AND doc_key = ? AND superseded = 0 ORDER BY version DESC LIMIT 1",
            (couple_id, doc_scope, doc_key),
        ).fetchone()
        return _row_to_doc(row) if row else None


def get_document_history(couple_id, doc_key: str,
                         doc_scope: str = SCOPE_COUPLE) -> list[dict]:
    """All versions of a living document, oldest first."""
    with db.get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM memory_documents WHERE couple_id IS ? AND doc_scope = ? "
            "AND doc_key = ? ORDER BY version ASC",
            (couple_id, doc_scope, doc_key),
        ).fetchall()
        return [_row_to_doc(r) for r in rows]


def list_documents(couple_id, allowed_scopes, doc_type: str | None = None,
                   limit: int = 100) -> list[dict]:
    """Latest (non-superseded) documents the caller may see, newest first.
    `allowed_scopes` has no default on purpose: every caller must decide."""
    where, params = _scope_filter(couple_id, allowed_scopes)
    sql = f"SELECT md.* FROM memory_documents md WHERE md.superseded = 0 AND {where}"
    if doc_type:
        sql += " AND md.doc_type = ?"
        params.append(doc_type)
    sql += " ORDER BY md.updated_at DESC, md.id DESC LIMIT ?"
    params.append(limit)

    with db.get_conn() as conn:
        return [_row_to_doc(r) for r in conn.execute(sql, params).fetchall()]


_WORD_RE = re.compile(r"\w+")


def _as_plain_terms(query: str) -> str:
    """Turn free text into a safe FTS5 query: each word double-quoted, so
    characters FTS5 treats as syntax (apostrophes, hyphens, colons, stray
    quotes) can't cause an error. Space-separated terms are AND-ed, which
    matches what the raw query would have meant."""
    return " ".join(f'"{w}"' for w in _WORD_RE.findall(query))


def search(query: str, couple_id, allowed_scopes, doc_type: str | None = None,
           limit: int = 8) -> list[dict]:
    """Full-text search over title + content. Returns the latest
    (non-superseded) matches the caller may see, best match first, each
    with a short highlighted snippet.

    The query is tried as written first, so FTS5 syntax the model may use
    (OR, "exact phrase", prefix*) still works. If FTS5 rejects it as
    malformed, it's retried as plain quoted words; v0.4 just returned []
    there, which silently lost queries like "Jordan's family"."""
    if not query or not query.strip():
        return []

    where, scope_params = _scope_filter(couple_id, allowed_scopes)
    # snippet(table, column 1 = content, open/close marks, ellipsis, ~12 tokens).
    # `rank` is FTS5's built-in bm25 relevance score (lower = better).
    sql = (
        "SELECT f.doc_id, md.title, md.doc_type, md.doc_scope, "
        "snippet(memory_fts, 1, '[', ']', ' ... ', 12) AS snippet, md.updated_at "
        "FROM memory_fts f JOIN memory_documents md ON md.id = f.doc_id "
        f"WHERE memory_fts MATCH ? AND md.superseded = 0 AND {where}"
    )
    tail_params = []
    if doc_type:
        sql += " AND md.doc_type = ?"
        tail_params.append(doc_type)
    sql += " ORDER BY rank LIMIT ?"
    tail_params.append(limit)

    with db.get_conn() as conn:
        for match in (query, _as_plain_terms(query)):
            if not match:
                break
            try:
                rows = conn.execute(sql, [match, *scope_params, *tail_params]).fetchall()
                return [dict(r) for r in rows]
            except sqlite3.OperationalError:
                continue  # FTS5 syntax error: try the sanitised form
    return []


# --- reference documents (shared, not personal), seeded once ---
# The first six are the couples notes: our own words, from the two books Riley
# chose (Hold Me Tight; The Seven Principles). Readable copy for her review:
# docs/reference_notes.md. The last three are ported from v0.4.

REFERENCE_DOCS = [
    {
        "doc_key": "reference:eft-attachment-basics",
        "title": "EFT basics: the relationship as an attachment bond",
        "content": (
            "Source: Sue Johnson, Hold Me Tight (2008). Emotionally Focused "
            "Therapy (EFT) sees a committed relationship as an attachment bond: "
            "a basic need for closeness and safety, not a weakness. What partners "
            "need from each other is A.R.E.: Accessible (can I reach you?), "
            "Responsive (will you respond when I need you?), Engaged (will you "
            "stay close and involved?). Many arguments about chores, money or "
            "phones carry a deeper question: 'Are you there for me? Do I matter "
            "to you?' When it feels unanswered, partners protest (criticise, "
            "push) or protect themselves (go quiet, withdraw). Use: look under "
            "the complaint for the need and reflect it to both partners, rather "
            "than refereeing the content. Draw on these ideas; never claim to "
            "deliver EFT, which trained therapists do."
        ),
    },
    {
        "doc_key": "reference:eft-negative-cycles",
        "title": "EFT: negative cycles (Demon Dialogues)",
        "content": (
            "Source: Sue Johnson, Hold Me Tight (2008). Johnson calls the "
            "recurring negative patterns couples get stuck in Demon Dialogues. "
            "Find the Bad Guy: mutual blame, each attacking to protect "
            "themselves. The Protest Polka: one partner pursues (pushes, "
            "criticises, demands contact) and the other withdraws (goes quiet, "
            "shuts down); the most common cycle, each move triggering more of "
            "the other. Freeze and Flee: both withdraw and the relationship goes "
            "cold. Key idea: the cycle is the enemy, not the partner; both moves "
            "make sense as attempts to protect the bond. Use: name the cycle as "
            "something the couple are caught in together ('the more one of you "
            "pushes, the more the other pulls back'), never as one partner's "
            "fault. Use the couple's own name for their cycle if they have one."
        ),
    },
    {
        "doc_key": "reference:eft-seven-conversations",
        "title": "EFT: the seven conversations",
        "content": (
            "Source: Sue Johnson, Hold Me Tight (2008). Seven conversations that "
            "build a more secure bond, in order: 1. Recognising the Demon "
            "Dialogues (spotting the cycle together). 2. Finding the raw spots "
            "(old sensitivities a partner's words can rub against). 3. "
            "Revisiting a rocky moment (going back over a fight calmly to "
            "understand it). 4. Hold me tight (reaching for each other and "
            "sharing needs softly and directly). 5. Forgiving injuries "
            "(repairing moments where one felt abandoned or betrayed). 6. "
            "Bonding through sex and touch. 7. Keeping your love alive (caring "
            "for the bond, building a shared story). Use: a map of where a "
            "conversation sits and a gentle next step. The app's conflict "
            "debrief is close in spirit to conversation 3 but is its own design; "
            "don't run the book's conversations as a programme."
        ),
    },
    {
        "doc_key": "reference:gottman-seven-principles",
        "title": "Gottman: the seven principles",
        "content": (
            "Source: John Gottman and Nan Silver, The Seven Principles for "
            "Making Marriage Work (1999). 1. Enhance your love maps: keep "
            "learning each other's inner world. 2. Nurture fondness and "
            "admiration: notice and say what you appreciate. 3. Turn toward each "
            "other: respond to small bids for attention. 4. Let your partner "
            "influence you: be open to their view, share decisions. 5. Solve "
            "your solvable problems: gentle start-up, repair attempts, "
            "compromise. 6. Overcome gridlock: on deep recurring disagreements, "
            "aim for dialogue and understanding the dream behind each position. "
            "7. Create shared meaning: rituals, roles, goals and values. Many "
            "conflicts are perpetual (lasting differences that won't be "
            "'solved'); the goal there is to keep talking about them kindly. "
            "Use: suggest small concrete habits (a daily check-in, noticing "
            "bids, one appreciation) and help tell solvable from perpetual."
        ),
    },
    {
        "doc_key": "reference:gottman-four-horsemen",
        "title": "Gottman: the four horsemen and their antidotes",
        "content": (
            "Source: John Gottman and Nan Silver, The Seven Principles for "
            "Making Marriage Work (1999). Four especially harmful patterns, each "
            "with a commonly taught antidote. Criticism (attacking the person, "
            "'you always...', not the behaviour) -> gentle start-up: 'I feel... "
            "about... I need...'. Contempt (mockery, sarcasm, eye-rolling, "
            "superiority; the most damaging) -> build a culture of appreciation. "
            "Defensiveness (counter-attacking or playing the victim) -> take "
            "responsibility, even for a small part. Stonewalling (shutting down, "
            "often when flooded) -> self-soothing: take a break of about 20 "
            "minutes, then come back. Use: name a horseman gently and without "
            "blame, and offer what the antidote might sound like. Never label "
            "either partner as 'a stonewaller' or 'contemptuous'."
        ),
    },
    {
        "doc_key": "reference:gottman-repair-and-startup",
        "title": "Gottman: gentle start-up and repair attempts",
        "content": (
            "Source: John Gottman and Nan Silver, The Seven Principles for "
            "Making Marriage Work (1999). Start-up: how a hard conversation "
            "begins tends to set how it ends; a harsh start-up (blame, sarcasm) "
            "usually goes badly, a gentle one (your own feeling and a specific "
            "need, no attack) gives it a better chance. Repair attempts: "
            "anything that de-escalates (humour, 'let's start over', 'I see your "
            "point', a touch); what matters most is whether the other partner "
            "can accept it, which is easier with a store of goodwill. Flooding: "
            "when one partner is physically overwhelmed, a productive "
            "conversation isn't possible; taking a break and agreeing when to "
            "come back is a skill, not a defeat. Use: suggest what a gentle "
            "start-up could sound like, point out a repair attempt kindly, or "
            "suggest an agreed pause."
        ),
    },
    {
        "doc_key": "reference:cognitive-distortions",
        "title": "Cognitive distortions cheat sheet",
        "content": (
            "All-or-nothing thinking: seeing things in black-and-white "
            "categories, no middle ground (e.g. 'If I don't get this "
            "perfect, I've completely failed').\n"
            "Overgeneralization: one event treated as an endless pattern "
            "(e.g. 'I got rejected once, I always get rejected').\n"
            "Mental filter: dwelling on one negative detail while ignoring "
            "everything positive.\n"
            "Discounting the positive: insisting positives don't count "
            "(e.g. 'they only said that to be nice').\n"
            "Jumping to conclusions: mind reading or fortune telling "
            "without evidence (e.g. 'she hasn't replied, she must be mad').\n"
            "Catastrophizing: expecting the worst-case outcome, or "
            "treating a setback as unbearable.\n"
            "Emotional reasoning: assuming that because you feel something, "
            "it must be true ('I feel like a failure, so I must be one').\n"
            "'Should' statements: rigid rules that create guilt or "
            "frustration when unmet ('I should never make mistakes').\n"
            "Labeling: a global negative label instead of describing the "
            "specific behavior ('I'm an idiot' vs 'I made a mistake').\n"
            "Personalization: blaming yourself for something you weren't "
            "fully responsible for."
        ),
    },
    {
        "doc_key": "reference:cbt-core-model",
        "title": "CBT core model (thoughts-feelings-behavior)",
        "content": (
            "Cognitive Behavioral Therapy's central idea: a situation "
            "triggers an automatic thought, the thought produces an "
            "emotion, and the emotion influences behavior, which often "
            "feeds back into the situation. The leverage point is the "
            "automatic thought: it's usually the fastest, least examined "
            "link in the chain, and testing it against actual evidence "
            "(not suppressing or arguing with the emotion directly) is "
            "what a thought record does. Behavioral activation works the "
            "other end of the same loop: instead of waiting for motivation "
            "(an emotion) to produce action, it deliberately schedules a "
            "small action first, on the premise that behavior can shift "
            "mood just as much as mood shifts behavior."
        ),
    },
    {
        "doc_key": "reference:schema-therapy-basics",
        "title": "Schema therapy: early maladaptive schemas (overview)",
        "content": (
            "Schema therapy (Jeffrey Young) proposes that unmet core "
            "childhood needs can produce 'early maladaptive schemas': "
            "broad, self-reinforcing patterns of thought, feeling and bodily "
            "sensation about oneself and relationships that persist into "
            "adulthood. Common domains include: Disconnection & Rejection "
            "(e.g. abandonment, mistrust/abuse, emotional deprivation), "
            "Impaired Autonomy (e.g. dependence, vulnerability to harm), "
            "Impaired Limits (e.g. entitlement, insufficient "
            "self-control), Other-Directedness (e.g. subjugation, "
            "self-sacrifice), and Overvigilance/Inhibition (e.g. "
            "unrelenting standards, negativity). The standard schema "
            "questionnaires are commercially licensed and are not "
            "administered here. In joint sessions this is background "
            "vocabulary only: a partner may name a pattern in their own "
            "words, but the agent never labels either partner with a schema."
        ),
    },
]


def seed_reference_documents():
    """Idempotent: skips any reference doc_key that already exists."""
    with db.get_conn() as conn:
        existing = {
            r["doc_key"]
            for r in conn.execute(
                "SELECT doc_key FROM memory_documents WHERE doc_scope = 'reference'"
            ).fetchall()
        }
    for doc in REFERENCE_DOCS:
        if doc["doc_key"] in existing:
            continue
        upsert_document(
            couple_id=None,
            doc_type=REFERENCE_TYPE,
            doc_key=doc["doc_key"],
            title=doc["title"],
            content=doc["content"],
            doc_scope=SCOPE_REFERENCE,
        )
