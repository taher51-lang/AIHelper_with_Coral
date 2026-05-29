"""
coral_client.py
───────────────
Thin wrapper around the Coral CLI (`coral sql`).

Provides a single public function, `run_coral_sql`, that executes an
arbitrary SQL query against a locally-running Coral instance and returns
the results as a list of Python dictionaries.

Usage (as a library):
    from coral_client import run_coral_sql
    rows = run_coral_sql("SELECT title, state FROM github.issues LIMIT 5;")

Usage (standalone test):
    python coral_client.py
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys

# ── Logging ──────────────────────────────────────────────────────────────────
# Configure a module-level logger so callers (e.g. FastAPI) can control the
# verbosity via standard Python logging configuration.
logger = logging.getLogger(__name__)


def run_coral_sql(query: str) -> list[dict]:
    """Execute a SQL query via the Coral CLI and return parsed rows.

    Parameters
    ----------
    query:
        A valid SQL string that Coral understands (e.g. queries against
        ``github.issues``, ``github.pull_requests``, etc.).

    Returns
    -------
    list[dict]
        Each dict represents one row, with column names as keys.
        Returns an empty list on any error (subprocess failure, JSON
        parse failure, unexpected output shape).
    """

    # 1. Build the command ────────────────────────────────────────────────────
    #    `--format json` asks Coral to emit machine-readable JSON to stdout.
    cmd: list[str] = ["coral", "sql", "--format", "json", query]

    # 2. Execute ─────────────────────────────────────────────────────────────
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,   # grab both stdout and stderr
            text=True,             # decode bytes → str automatically
            check=True,            # raise CalledProcessError on non-zero exit
        )
    except FileNotFoundError:
        # `coral` binary is not on $PATH
        logger.error(
            "The 'coral' command was not found. "
            "Make sure Coral is installed and available on your PATH."
        )
        return []
    except subprocess.CalledProcessError as exc:
        # Coral returned a non-zero exit code
        logger.error(
            "Coral query failed (exit code %d).\n"
            "  ┌─ command : %s\n"
            "  ├─ stderr  : %s\n"
            "  └─ stdout  : %s",
            exc.returncode,
            " ".join(cmd),
            (exc.stderr or "").strip(),
            (exc.stdout or "").strip(),
        )
        return []

    # 3. Parse JSON output ───────────────────────────────────────────────────
    raw_output = result.stdout.strip()

    if not raw_output:
        logger.warning("Coral returned empty output for query: %s", query)
        return []

    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        logger.error(
            "Failed to parse Coral output as JSON.\n"
            "  ┌─ error  : %s\n"
            "  └─ output : %.500s",       # truncate to 500 chars for safety
            exc,
            raw_output,
        )
        return []

    # 4. Normalise to list[dict] ─────────────────────────────────────────────
    #    Coral *should* return a JSON array of objects, but we guard against
    #    unexpected shapes (a single object, nested wrapper, etc.).
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        # Some CLI tools wrap rows inside a key like "results" or "rows".
        # Try common wrapper keys first; fall back to returning a one-item list.
        for key in ("results", "rows", "data", "records"):
            if key in parsed and isinstance(parsed[key], list):
                logger.debug("Unwrapped results from '%s' key.", key)
                return parsed[key]
        return [parsed]

    # Truly unexpected type – log and bail.
    logger.error("Unexpected JSON type from Coral: %s", type(parsed).__name__)
    return []


# ── Standalone smoke-test ────────────────────────────────────────────────────
if __name__ == "__main__":
    # Wire up basic console logging so we can see errors when running directly.
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    # ── Test 1: list your GitHub repos (no required WHERE filters) ───────
    TEST_QUERIES: list[tuple[str, str]] = [
        (
            "Repos",
            "SELECT name, stargazers_count FROM github.user_repos LIMIT 3;",
        ),
        (
            "Issues",
            "SELECT title, state FROM github.user_issues LIMIT 3;",
        ),
    ]

    exit_code = 0

    for label, query in TEST_QUERIES:
        print("\n" + "─" * 60)
        print(f"[{label}] Running test query:\n  {query}")
        print("─" * 60)

        rows = run_coral_sql(query)

        if rows:
            print(f"\n✅  Success — received {len(rows)} row(s):\n")
            for i, row in enumerate(rows, start=1):
                print(f"  Row {i}:")
                for col, val in row.items():
                    print(f"    {col}: {val}")
        else:
            print(f"\n⚠️  [{label}] No rows returned (check the logs above).")
            exit_code = 1

    sys.exit(exit_code)
