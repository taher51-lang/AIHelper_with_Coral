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
import json
from contextlib import asynccontextmanager

import httpx
from groq import AsyncGroq
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Local modules ────────────────────────────────────────────────────────────
from coral_client import run_coral_sql
from sql_agent import generate_coral_query
from marketing_engine import transform_data_to_campaign, MarketingCampaign, CampaignResult, refine_campaign
from data_synthesizer import synthesize_github_data

# ── Load .env early (before anything reads os.environ) ───────────────────────
load_dotenv()

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
)
logger = logging.getLogger(__name__)

# Silence noisy HTTP and Groq logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("groq").setLevel(logging.WARNING)


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


class RefineRequest(BaseModel):
    """Incoming request to refine a generated campaign."""
    previous_result: CampaignResult = Field(..., description="The full previous campaign and strategy result.")
    feedback: str = Field(..., description="The user's feedback or instructions for changes.")


class TranslatedEvent(BaseModel):
    raw_type: str = Field(..., description="The raw event type (e.g. PushEvent).")
    raw_repo: str = Field(..., description="The repository name.")
    plain_english: str = Field(..., description="A 1-sentence plain-English translation of the event(s) (e.g. 'Pushed 5 commits fixing authentication bugs').")
    icon: str = Field(..., description="A single emoji representing the event.")

class TimelineLLMResponse(BaseModel):
    theme: str = Field(..., description="A short 2-5 word theme for the week.")
    journal_entry: str = Field(..., description="A 2-sentence narrative summarizing the week's activity.")
    unsung_hero: str = Field(..., description="Highlight one impactful but unglamorous action (e.g. closing an old issue).")
    translated_events: list[TranslatedEvent] = Field(..., description="Chronological translation of events.")


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
#  Repos endpoint (For Repo Spotlight)
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/repos", tags=["System"])
async def get_repos():
    """Fetch public repositories for the Repo Spotlight dropdown."""
    repos = run_coral_sql(
        "SELECT name, owner__login, stargazers_count "
        "FROM github.user_repos WHERE visibility = 'public' "
        "ORDER BY pushed_at DESC LIMIT 50;"
    )
    return sorted(
        [{"name": r.get("name"), "owner": r.get("owner__login"), "stars": r.get("stargazers_count")} for r in repos],
        key=lambda x: int(x.get("stars", 0) or 0),
        reverse=True
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Developer Impact Score endpoint
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/developer-score", tags=["Analytics"])
async def developer_score():
    """Compute a gamified Developer Impact Score from Coral GitHub data.

    Weights:
      - Activity (events)    : 50%
      - Reach (stars + forks): 25%
      - Reliability (issues) : 25%
    """

    # ── Query Coral for three dimensions ────────────────────────────────────
    repos = run_coral_sql(
        "SELECT name, stargazers_count, forks_count, language "
        "FROM github.user_repos WHERE visibility = 'public' LIMIT 25;"
    )
    events = run_coral_sql(
        "SELECT 'CreateEvent' as type FROM github.user_repos UNION ALL SELECT 'PushEvent' as type FROM github.user_repos WHERE pushed_at IS NOT NULL LIMIT 25;"
    )
    issues = run_coral_sql(
        "SELECT title, state FROM github.user_issues LIMIT 25;"
    )

    # ── 1. ACTIVITY score (50% weight) ──────────────────────────────────────
    total_events = len(events)
    # Normalize: 25 events (our limit) = 100
    activity_raw = min(100, int((total_events / 25) * 100))

    # Count event types for breakdown
    event_types: dict[str, int] = {}
    for e in events:
        t = e.get("type", "Unknown")
        event_types[t] = event_types.get(t, 0) + 1

    # ── 2. REACH score (25% weight) ─────────────────────────────────────────
    total_stars = sum(int(r.get("stargazers_count", 0) or 0) for r in repos)
    total_forks = sum(int(r.get("forks_count", 0) or 0) for r in repos)
    # Logarithmic scale: 50 stars+forks = ~100
    import math
    reach_raw = min(100, int(math.log(max(total_stars + total_forks, 1) + 1, 1.05)))
    reach_raw = min(reach_raw, 100)

    # Language breakdown
    languages: dict[str, int] = {}
    for r in repos:
        lang = r.get("language")
        if lang:
            languages[lang] = languages.get(lang, 0) + 1

    # ── 3. RELIABILITY score (25% weight) ───────────────────────────────────
    total_issues = len(issues)
    closed_issues = sum(1 for i in issues if str(i.get("state", "")).lower() in ("closed", "merged"))
    reliability_raw = int((closed_issues / max(total_issues, 1)) * 100)

    # ── Composite score ─────────────────────────────────────────────────────
    composite = int(
        activity_raw * 0.50 +
        reach_raw * 0.25 +
        reliability_raw * 0.25
    )

    # ── Level system ────────────────────────────────────────────────────────
    if composite >= 90:
        level = "🏆 Elite Builder"
    elif composite >= 75:
        level = "🔥 Master Shipper"
    elif composite >= 60:
        level = "⚡ Active Developer"
    elif composite >= 40:
        level = "🌱 Growing Builder"
    else:
        level = "🌟 Rising Star"

    return {
        "composite_score": composite,
        "level": level,
        "dimensions": {
            "activity": {
                "score": activity_raw,
                "weight": "50%",
                "total_events": total_events,
                "event_breakdown": event_types,
            },
            "reach": {
                "score": reach_raw,
                "weight": "25%",
                "total_stars": total_stars,
                "total_forks": total_forks,
                "languages": languages,
            },
            "reliability": {
                "score": reliability_raw,
                "weight": "25%",
                "total_issues": total_issues,
                "closed_issues": closed_issues,
            },
        },
        "top_repos": sorted(
            [{"name": r.get("name"), "stars": r.get("stargazers_count", 0),
              "language": r.get("language")} for r in repos],
            key=lambda x: int(x.get("stars", 0) or 0),
            reverse=True,
        )[:5],
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Main campaign generation endpoint
# ═══════════════════════════════════════════════════════════════════════════════

@app.post(
    "/api/generate-campaign",
    response_model=CampaignResult,
    tags=["Campaigns"],
    summary="Generate a multi-platform marketing campaign from GitHub data",
)
async def generate_campaign(
    request: CampaignRequest,
    background_tasks: BackgroundTasks,
) -> CampaignResult:
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
                    # Try common branch names and cases
                    branches = ["main", "master"]
                    filenames = ["README.md", "readme.md", "Readme.md"]
                    
                    readme_fetched = False
                    for branch in branches:
                        if readme_fetched: break
                        for filename in filenames:
                            url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{filename}"
                            try:
                                resp = await client.get(url, timeout=5.0)
                                if resp.status_code == 200:
                                    row["readme_content"] = resp.text
                                    logger.info("✅  Fetched README for %s/%s (%s/%s)", owner, repo, branch, filename)
                                    print(f"\n\n{'='*80}\n[EXACT README CONTENT FETCHED FOR {owner}/{repo}]\n{'='*80}\n{resp.text}\n{'='*80}\n\n")
                                    readme_fetched = True
                                    break
                            except Exception as exc:
                                logger.debug("Failed to fetch %s on branch %s: %s", filename, branch, exc)
                    
                    if not readme_fetched:
                        logger.warning("Failed to fetch any README for %s/%s", owner, repo)

    # ── Step D: Raw Data → Marketing Campaign ─────────────────────────────
    logger.info("━" * 60)
    logger.info("🎨  STEP D  ▸  Transforming data into marketing campaign")
    logger.info("   Rows   : %d", len(raw_data))
    logger.info("   Tone   : %s", tone)
    logger.info("━" * 60)

    synthesized_context = synthesize_github_data(raw_data)

    try:
        result = await transform_data_to_campaign(synthesized_context, intent, tone)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Marketing engine failed: {exc}",
        ) from exc

    campaign = result.campaign

    logger.info("━" * 60)
    logger.info("🏆  PIPELINE COMPLETE")
    logger.info("   Campaign : %s", campaign.campaign_name)
    logger.info("   Reasoning: %s", result.reasoning.selected_marketing_angle)
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

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Refine campaign endpoint
# ═══════════════════════════════════════════════════════════════════════════════

@app.post(
    "/api/refine-campaign",
    response_model=CampaignResult,
    tags=["Campaigns"],
    summary="Refine a generated campaign using user feedback",
)
async def refine_campaign_endpoint(
    request: RefineRequest,
    background_tasks: BackgroundTasks,
) -> CampaignResult:
    """Iteratively refine an existing campaign based on user feedback."""
    
    logger.info("━" * 60)
    logger.info("🔄  REFINING CAMPAIGN")
    logger.info("   Feedback: %s", request.feedback)
    logger.info("━" * 60)

    try:
        result = await refine_campaign(request.previous_result, request.feedback)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Refinement failed: {exc}",
        ) from exc

    logger.info("✅ Refinement complete.")
    
    # ── Fire webhook in the background (non-blocking) ─────────────────────
    background_tasks.add_task(
        send_webhook_alert,
        result.campaign.campaign_name,
        result.campaign.linkedin.hook,
    )

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Timeline endpoint
# ═══════════════════════════════════════════════════════════════════════════════

TIMELINE_SYSTEM_PROMPT = """\\
You are an elite Developer Relations advocate and technical storyteller.
Your job is to take a chronological list of raw GitHub events and translate them into a compelling "Developer Journal".

You will receive a JSON list of events.
1. Determine an overarching "theme" for the week.
2. Write a 2-sentence journal entry summarizing the progress.
3. Identify the "unsung hero" action (a bug fix, closing an old issue, refactoring, etc).
4. Group and translate the raw events into a readable plain-English timeline, assigning a fitting emoji icon to each.

Return a strictly formatted JSON object matching this JSON schema:
{schema}

Do not make up events.
"""

@app.get("/api/timeline", tags=["Analytics"])
async def timeline():
    """Fetch recent events from Coral and generate a narrated timeline."""
    
    # 1. Fetch raw events
    events = run_coral_sql(
        "SELECT 'CreateEvent' as type, created_at, name as repo FROM github.user_repos "
        "UNION ALL SELECT 'PushEvent' as type, pushed_at as created_at, name as repo "
        "FROM github.user_repos WHERE pushed_at IS NOT NULL "
        "ORDER BY created_at DESC LIMIT 30;"
    )

    if not events:
        raise HTTPException(status_code=404, detail="No recent activity found.")

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="Missing GROQ_API_KEY.")

    client = AsyncGroq(api_key=api_key)

    # 2. Call LLM to narrate the timeline
    user_message = f"Here is the raw activity data:\n```json\n{json.dumps(events, indent=2)}\n```"

    try:
        system_prompt = TIMELINE_SYSTEM_PROMPT.format(schema=json.dumps(TimelineLLMResponse.model_json_schema(), indent=2))
        response = await client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.4,
            response_format={"type": "json_object"}
        )
        
        raw_text = response.choices[0].message.content
        if not raw_text:
            raise RuntimeError("Empty response from Groq.")
            
        llm_output = TimelineLLMResponse.model_validate_json(raw_text)
        return llm_output.model_dump()
        
    except Exception as exc:
        logger.error("Timeline generation failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Timeline generation failed: {exc}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Content Calendar endpoint
# ═══════════════════════════════════════════════════════════════════════════════

class CalendarPost(BaseModel):
    day: str = Field(..., description="The day of the week (e.g. Monday, Wednesday, Friday)")
    theme: str = Field(..., description="The theme of the post (e.g. Deep Dive, Throwback)")
    platform: str = Field(..., description="The social media platform (e.g. Twitter, LinkedIn)")
    content: str = Field(..., description="The generated post copy.")

class CalendarLLMResponse(BaseModel):
    title: str = Field(..., description="A catchy title for the week's calendar")
    posts: list[CalendarPost] = Field(..., description="A list of 3-5 scheduled posts")

CALENDAR_SYSTEM_PROMPT = """\
You are an elite Developer Relations strategist.
Your job is to generate a multi-day social media content calendar for a developer based on their GitHub data.

You will receive a JSON context with different slices of the developer's data (e.g., most active repo, most starred repo, older repo).
Create a 3-day content calendar (e.g., Monday, Wednesday, Friday).
Vary the platforms (Twitter/LinkedIn) and themes (e.g., Deep Dive, Throwback, Hype).

Return a strictly formatted JSON object matching this JSON schema:
{schema}

Do not make up data that isn't in the provided context, but be creative with the marketing copy.
"""

@app.get("/api/calendar", tags=["Analytics"])
async def calendar():
    """Generate a multi-day content calendar using different data slices."""
    
    # 1. Fetch data slices
    top_stars = run_coral_sql(
        "SELECT name, stargazers_count, description FROM github.user_repos ORDER BY stargazers_count DESC LIMIT 1;"
    )
    recent_pushes = run_coral_sql(
        "SELECT name as repo FROM github.user_repos WHERE pushed_at IS NOT NULL ORDER BY pushed_at DESC LIMIT 1;"
    )
    old_repo = run_coral_sql(
        "SELECT name, created_at, description FROM github.user_repos ORDER BY created_at ASC LIMIT 1;"
    )

    context = {
        "most_starred": top_stars,
        "most_active_recently": recent_pushes,
        "throwback": old_repo
    }

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="Missing GROQ_API_KEY.")

    client = AsyncGroq(api_key=api_key)

    user_message = f"Here is the developer's data context:\n```json\n{json.dumps(context, indent=2)}\n```"

    try:
        system_prompt = CALENDAR_SYSTEM_PROMPT.format(schema=json.dumps(CalendarLLMResponse.model_json_schema(), indent=2))
        response = await client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        
        raw_text = response.choices[0].message.content
        if not raw_text:
            raise RuntimeError("Empty response from Groq.")
            
        llm_output = CalendarLLMResponse.model_validate_json(raw_text)
        return llm_output.model_dump()
        
    except Exception as exc:
        logger.error("Calendar generation failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Calendar generation failed: {exc}")
