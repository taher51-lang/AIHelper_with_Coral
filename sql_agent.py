"""
sql_agent.py
────────────
LLM-powered agent that translates a creator's natural-language request into
a valid SQL query targeting Coral's GitHub schema.

Uses Google Gemini (via the `google-genai` SDK) with **native structured
output** so the model is forced to return a JSON object matching our
Pydantic `SQLOutput` schema — no post-hoc regex parsing needed.

Usage (as a library):
    from sql_agent import generate_coral_query
    sql = await generate_coral_query("Show me my latest merged PRs")

Usage (standalone test):
    python sql_agent.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys

from pydantic import BaseModel, Field
from groq import AsyncGroq

# ── Logging ──────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
MODEL_ID = "llama-3.3-70b-versatile"

# ── Pydantic schema for structured LLM output ───────────────────────────────
# Gemini's `responseSchema` will enforce this shape at decoding time, so the
# model *cannot* return free-form text — only a valid JSON matching these
# two fields.

class SQLOutput(BaseModel):
    """Structured output returned by the LLM."""

    rationale: str = Field(
        ...,
        description=(
            "A short (1-2 sentence) explanation of why these specific "
            "tables and columns were chosen for the query."
        ),
    )
    sql_query: str = Field(
        ...,
        description=(
            "A single, syntactically correct SQL SELECT statement that "
            "targets only the allowed Coral GitHub tables and columns. "
            "Must end with a semicolon."
        ),
    )


# ── System prompt ────────────────────────────────────────────────────────────
# This is the single source of truth for what the model is allowed to query.
# By exhaustively listing every table + column we support, we stop the model
# from hallucinating table or column names that don't exist in Coral.

SYSTEM_PROMPT = """\
You are a SQL assistant for a creator-analytics dashboard.

Your ONLY job is to convert natural-language requests into valid SQL queries
that run against **Coral** — a local SQL interface for GitHub data.

═══════════════════════════════════════════════════════════════
AVAILABLE TABLES AND COLUMNS  (you MUST NOT invent any others)
═══════════════════════════════════════════════════════════════

1. github.user_repos
   ├── name               (text)   — repository name
   ├── description         (text)   — repo description (can be NULL)
   ├── language            (text)   — primary programming language
   ├── stargazers_count    (int)    — number of stars
   ├── forks_count         (int)    — number of forks
   ├── open_issues_count   (int)    — number of open issues
   ├── visibility          (text)   — "public" or "private"
   ├── created_at          (text)   — ISO-8601 creation timestamp
   ├── updated_at          (text)   — ISO-8601 last-updated timestamp
   ├── pushed_at           (text)   — ISO-8601 last-push timestamp
   ├── archived            (bool)   — whether the repo is archived
   └── default_branch      (text)   — e.g. "main" or "master"

2. github.user_issues
   ├── number              (int)    — issue number
   ├── title               (text)   — issue title
   ├── body                (text)   — issue body / description
   ├── state               (text)   — "open" or "closed"
   ├── created_at          (text)   — ISO-8601 creation timestamp
   ├── closed_at           (text)   — ISO-8601 close timestamp (NULL if open)
   └── author_association  (text)   — e.g. "OWNER", "COLLABORATOR"

3. github.events
   ├── type                (text)   — event type (PushEvent, PullRequestEvent,
   │                                  IssuesEvent, CreateEvent, WatchEvent, etc.)
   ├── created_at          (text)   — ISO-8601 timestamp of the event
   ├── repo__name          (text)   — "owner/repo" format
   └── public              (bool)   — whether the event is public

4. github.gists
   ├── description         (text)   — gist description
   ├── public              (bool)   — whether the gist is public
   ├── created_at          (text)   — ISO-8601 creation timestamp
   └── updated_at          (text)   — ISO-8601 last-updated timestamp

5. github.notifications
   ├── reason              (text)   — e.g. "subscribed", "mention", "assign"
   ├── unread              (bool)   — whether it is unread
   ├── updated_at          (text)   — ISO-8601 timestamp
   └── subject__title      (text)   — title of the notification subject

═══════════════════════════════════════════════════════════════
RULES
═══════════════════════════════════════════════════════════════

• ONLY reference the tables and columns listed above.
• Do NOT use subqueries, CTEs, or window functions — keep it simple.
• Always include a LIMIT clause (max 25 rows) unless the user explicitly
  asks for a count / aggregate.
• Use ORDER BY when it makes sense (e.g. most recent first).
• If the user's request is ambiguous, pick the most reasonable interpretation
  and explain your choice in the `rationale` field.
• The `sql_query` must be a single SELECT statement ending with a semicolon.
• These tables do NOT require any WHERE filters — they return the
  authenticated user's data automatically.
• When the user says "updates", "activity", or "latest", prefer
  github.events ordered by created_at DESC — it captures all activity types.
• When the user asks about "repos", "projects", or "portfolio", use
  github.user_repos.
• For text matching (like repository names or issue titles), ALWAYS use 
  case-insensitive `ILIKE '%...%'` matching instead of strict equality `=`. 
  Use very short, single-word keywords (e.g., `ILIKE '%RAG%'`) to avoid 
  mismatches caused by spaces vs hyphens in repository names.

You must respond in valid JSON format adhering to this structure:
{
  "rationale": "Explanation here",
  "sql_query": "SELECT ...;"
}
"""


# ── Core async function ─────────────────────────────────────────────────────

async def generate_coral_query(user_prompt: str) -> str:
    """Translate a natural-language request into a Coral SQL query.

    Parameters
    ----------
    user_prompt:
        The creator's free-form request (e.g. "Show me my latest updates").

    Returns
    -------
    str
        A raw SQL string ready to be passed to `coral_client.run_coral_sql`.

    Raises
    ------
    ValueError
        If the ``GEMINI_API_KEY`` environment variable is not set.
    RuntimeError
        If the Gemini API call fails or returns an unparsable response.
    """

    # ── 1. Resolve API key ──────────────────────────────────────────────────
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing API key. Set the GROQ_API_KEY "
            "environment variable before calling generate_coral_query()."
        )

    # ── 2. Create a Groq client ─────────────────────────────────────────────
    client = AsyncGroq(api_key=api_key)

    # ── 3. Call the model ───────────────────────────────────────────────────
    logger.info("Sending prompt to %s: %.120s…", MODEL_ID, user_prompt)

    try:
        response = await client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
    except Exception as exc:
        logger.error("Groq API call failed: %s", exc)
        raise RuntimeError(f"Groq API call failed: {exc}") from exc

    # ── 4. Parse the structured response ────────────────────────────────────
    raw_text = response.choices[0].message.content
    if not raw_text:
        raise RuntimeError("Groq returned an empty response.")

    logger.debug("Raw Groq response: %s", raw_text)

    try:
        parsed = SQLOutput.model_validate_json(raw_text)
    except Exception as exc:
        logger.error("Failed to parse Groq response into SQLOutput: %s", exc)
        raise RuntimeError(
            f"Could not parse structured output from Groq: {exc}"
        ) from exc

    logger.info("Rationale: %s", parsed.rationale)
    logger.info("Generated SQL: %s", parsed.sql_query)

    return parsed.sql_query


# ── Standalone smoke-test ────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    # ── Quick API-key check ─────────────────────────────────────────────────
    if not os.environ.get("GROQ_API_KEY"):
        print(
            "\n❌  No GROQ_API_KEY found in environment.\n"
            "   Set one before running this test:\n\n"
            "     export GROQ_API_KEY='your-key-here'\n"
            "     python sql_agent.py\n"
        )
        sys.exit(1)

    # ── Test prompts ────────────────────────────────────────────────────────
    TEST_PROMPTS = [
        "Show me my latest updates",
        "How many open issues do I have?",
        "List my recently merged pull requests",
    ]

    async def _run_tests() -> None:
        exit_code = 0
        for prompt in TEST_PROMPTS:
            print("\n" + "─" * 60)
            print(f"Prompt: \"{prompt}\"")
            print("─" * 60)
            try:
                sql = await generate_coral_query(prompt)
                print(f"\n✅  Generated SQL:\n    {sql}")
            except Exception as exc:
                print(f"\n❌  Error: {exc}")
                exit_code = 1
        if exit_code:
            sys.exit(exit_code)

    asyncio.run(_run_tests())
