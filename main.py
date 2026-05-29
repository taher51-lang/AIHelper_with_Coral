"""
main.py
───────
FastAPI backend that orchestrates the full Coral → LLM → Marketing pipeline.

Flow:
  1. Creator sends a natural-language intent  →  POST /api/generate-campaign
  2. sql_agent.py   translates intent → SQL
  3. coral_client.py  executes the SQL against local Coral
  4. marketing_engine.py  transforms raw data → multi-platform campaign
  5. (optional) webhook alert fires in the background
  6. Full MarketingCampaign JSON returned to the caller

Run:
    set -a && source .env && set +a
    .venv/bin/uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os
import textwrap
from contextlib import asynccontextmanager

import httpx
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Local modules ────────────────────────────────────────────────────────────
from coral_client import run_coral_sql
from sql_agent import generate_coral_query
from marketing_engine import transform_data_to_campaign, MarketingCampaign
from data_synthesizer import synthesize_github_data

# ── Load .env early (before anything reads os.environ) ───────────────────────
load_dotenv()

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  Lifespan — runs once on startup / shutdown
# ═══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(application: FastAPI):
    """Log startup diagnostics, then yield control to the app."""

    logger.info("=" * 60)
    logger.info("  🚀  Coral AI Creator & Marketing Dashboard")
    logger.info("=" * 60)
    logger.info("  Coral CLI  :  %s", "coral sql --format json")
    logger.info("  Gemini key :  %s", "SET ✅" if os.getenv("GEMINI_API_KEY") else "MISSING ❌")
    logger.info("  Webhook    :  %s",
                "configured ✅" if _get_webhook_url() else "not set (skipping)")
    logger.info("=" * 60)
    yield
    logger.info("👋  Shutting down.")


# ═══════════════════════════════════════════════════════════════════════════════
#  FastAPI app
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Coral AI Creator & Marketing Dashboard",
    description="Turn your GitHub activity into polished marketing campaigns.",
    version="0.1.0",
    lifespan=lifespan,
)

# ── CORS — allow any local frontend (Vite dev-server, etc.) ─────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Request / Response models
# ═══════════════════════════════════════════════════════════════════════════════

class CampaignRequest(BaseModel):
    """Incoming request from the frontend dashboard."""

    user_intent: str = Field(
        ...,
        min_length=3,
        examples=["Find my recent feature updates from this week and make a launch post"],
        description="A natural-language description of what the creator wants to promote.",
    )
    tone: str = Field(
        default="Professional",
        examples=["Professional", "Casual", "Technical", "Hype"],
        description="The desired tone for the generated marketing copy.",
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Webhook alert (background utility)
# ═══════════════════════════════════════════════════════════════════════════════

def _get_webhook_url() -> str | None:
    """Return the first available webhook URL from environment variables."""
    return os.getenv("DISCORD_WEBHOOK_URL") or os.getenv("SLACK_WEBHOOK_URL")


async def send_webhook_alert(campaign_name: str, linkedin_hook: str) -> None:
    """Fire an async HTTP POST to Discord / Slack notifying the team.

    Runs as a FastAPI background task so it never blocks the response.
    Silently logs errors — a webhook failure should never crash the API.
    """
    webhook_url = _get_webhook_url()
    if not webhook_url:
        logger.debug("No webhook URL configured — skipping alert.")
        return

    # ── Build a clean markdown message ──────────────────────────────────────
    # Discord and Slack both render markdown in webhook payloads.
    message = textwrap.dedent(f"""\
        🎯 **New Campaign Generated**

        **Campaign:** {campaign_name}

        **LinkedIn Hook Preview:**
        > {linkedin_hook}

        Head to the dashboard to preview and publish! 🚀
    """).strip()

    # Discord uses {"content": ...}, Slack uses {"text": ...}.
    is_discord = "discord.com" in webhook_url
    payload = {"content": message} if is_discord else {"text": message}

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(webhook_url, json=payload)
            resp.raise_for_status()
        logger.info("✅ Webhook alert sent to %s.", "Discord" if is_discord else "Slack")
    except Exception as exc:
        logger.warning("⚠️  Webhook alert failed (non-fatal): %s", exc)


# ═══════════════════════════════════════════════════════════════════════════════
#  Health-check endpoint
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/health", tags=["System"])
async def health_check():
    """Quick liveness probe for the dashboard frontend."""
    return {
        "status": "healthy",
        "gemini_key": "set" if os.getenv("GEMINI_API_KEY") else "missing",
        "webhook": "configured" if _get_webhook_url() else "not_set",
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Main campaign generation endpoint
# ═══════════════════════════════════════════════════════════════════════════════

@app.post(
    "/api/generate-campaign",
    response_model=MarketingCampaign,
    tags=["Campaigns"],
    summary="Generate a multi-platform marketing campaign from GitHub data",
)
async def generate_campaign(
    request: CampaignRequest,
    background_tasks: BackgroundTasks,
) -> MarketingCampaign:
    """
    End-to-end pipeline:
    Intent → SQL → Coral data → Marketing campaign → JSON response.
    """

    intent = request.user_intent
    tone = request.tone

    # ── Step A: Natural Language → SQL ──────────────────────────────────────
    logger.info("━" * 60)
    logger.info("📝  STEP A  ▸  Translating intent to SQL")
    logger.info("   Intent : %s", intent)
    logger.info("   Tone   : %s", tone)
    logger.info("━" * 60)

    try:
        sql_query = await generate_coral_query(intent)
    except ValueError as exc:
        # Missing API key
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"LLM failed to generate SQL: {exc}",
        ) from exc

    logger.info("✅  Generated SQL: %s", sql_query)

    # ── Step B: SQL → Coral Data ───────────────────────────────────────────
    logger.info("━" * 60)
    logger.info("🔍  STEP B  ▸  Executing query against Coral")
    logger.info("   Query  : %s", sql_query)
    logger.info("━" * 60)

    raw_data = run_coral_sql(sql_query)

    logger.info("📦  Coral returned %d row(s)", len(raw_data))

    # ── Step C: Empty-data guard ───────────────────────────────────────────
    if not raw_data:
        logger.warning("⚠️  STEP C  ▸  No data returned from Coral")
        raise HTTPException(
            status_code=404,
            detail={
                "error": "no_data",
                "message": (
                    "Coral returned no results for the generated query. "
                    "This could mean the data doesn't exist yet in your "
                    "GitHub repositories, or the query needs refinement."
                ),
                "generated_sql": sql_query,
                "hint": "Try a broader request like 'Show me all my recent activity'.",
            },
        )

    logger.info("✅  STEP C  ▸  Data payload validated (%d rows)", len(raw_data))
    if raw_data:
        logger.debug("   Sample row: %s", raw_data[0])

    # ── Step C.5: Optional README Fetching ────────────────────────────────
    # If the intent mentions "readme" and we have repository rows, fetch it.
    if "readme" in intent.lower():
        logger.info("📚  STEP C.5 ▸  Attempting to fetch README.md for repositories")
        async with httpx.AsyncClient() as client:
            for row in raw_data:
                owner = row.get("owner__login")
                repo = row.get("name")
                if owner and repo:
                    url = f"https://raw.githubusercontent.com/{owner}/{repo}/main/README.md"
                    try:
                        resp = await client.get(url, timeout=5.0)
                        if resp.status_code == 200:
                            row["readme_content"] = resp.text
                            logger.info("✅  Fetched README for %s/%s", owner, repo)
                        else:
                            # Try master branch fallback
                            url_master = f"https://raw.githubusercontent.com/{owner}/{repo}/master/README.md"
                            resp_master = await client.get(url_master, timeout=5.0)
                            if resp_master.status_code == 200:
                                row["readme_content"] = resp_master.text
                                logger.info("✅  Fetched README (master) for %s/%s", owner, repo)
                    except Exception as exc:
                        logger.warning("Failed to fetch README for %s/%s: %s", owner, repo, exc)

    # ── Step D: Raw Data → Marketing Campaign ─────────────────────────────
    logger.info("━" * 60)
    logger.info("🎨  STEP D  ▸  Transforming data into marketing campaign")
    logger.info("   Rows   : %d", len(raw_data))
    logger.info("   Tone   : %s", tone)
    logger.info("━" * 60)

    synthesized_context = synthesize_github_data(raw_data)

    try:
        campaign = await transform_data_to_campaign(synthesized_context, intent, tone)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Marketing engine failed: {exc}",
        ) from exc

    logger.info("━" * 60)
    logger.info("🏆  PIPELINE COMPLETE")
    logger.info("   Campaign : %s", campaign.campaign_name)
    logger.info("   LinkedIn : %d chars", len(campaign.linkedin.body))
    logger.info("   Tweets   : %d tweets", len(campaign.twitter.tweets))
    logger.info("   IG prompt: %.80s…", campaign.instagram.image_generation_prompt)
    logger.info("━" * 60)

    # ── Fire webhook in the background (non-blocking) ─────────────────────
    background_tasks.add_task(
        send_webhook_alert,
        campaign.campaign_name,
        campaign.linkedin.hook,
    )

    return campaign
