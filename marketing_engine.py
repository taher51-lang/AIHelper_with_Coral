"""
marketing_engine.py
───────────────────
A Two-Stage Reasoning Chain that transforms clean synthesized data 
into polished, platform-specific marketing campaigns using Gemini 3.1 Pro.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os

from pydantic import BaseModel, Field
from groq import AsyncGroq

logger = logging.getLogger(__name__)

# Constants
MODEL_ID = "llama-3.3-70b-versatile"

# ═══════════════════════════════════════════════════════════════════════════════
#  Pydantic Schemas
# ═══════════════════════════════════════════════════════════════════════════════

class CampaignStrategy(BaseModel):
    """Internal Strategy object generated in Stage 1."""
    selected_marketing_angle: str = Field(
        ...,
        description="The chosen narrative angle (e.g., Deep-Dive Educational, High-Hype Feature Launch, Problem-Solution Refactor)."
    )
    core_narrative_hook: str = Field(
        ...,
        description="A 1-2 sentence core hook summarizing the overarching story."
    )
    key_value_proposition: str = Field(
        ...,
        description="The primary technical or user value delivered by this work."
    )

class LinkedInPost(BaseModel):
    hook: str = Field(..., description="The opening 1-2 lines. Must be attention-grabbing.")
    body: str = Field(..., description="The full post body. Educational and value-driven.")
    hashtags: str = Field(..., description="3-6 relevant hashtags separated by spaces.")

class TwitterThread(BaseModel):
    tweets: list[str] = Field(..., description="Ordered tweets. Each MUST be strictly <= 280 characters.")

class InstagramConcept(BaseModel):
    image_generation_prompt: str = Field(..., description="A vivid text prompt for an AI image generator.")
    caption: str = Field(..., description="An optimized caption with 10-15 hashtags.")

class MarketingCampaign(BaseModel):
    campaign_name: str = Field(..., description="Catchy campaign name (5-8 words).")
    linkedin: LinkedInPost
    twitter: TwitterThread
    instagram: InstagramConcept

# ═══════════════════════════════════════════════════════════════════════════════
#  System Prompts
# ═══════════════════════════════════════════════════════════════════════════════

STAGE_1_SYSTEM_PROMPT = """\
You are an elite AI Content Strategist. Your job is to analyze technical developer 
activity metrics and define a core marketing strategy. 

You will receive:
1. A synthesized context payload containing summary metrics, categorized activity, and keywords.
2. The original goal of the creator.
3. The requested target tone.

Analyze the 'computed_impact_score' to understand the gravity of the work, and use the 
arrays inside 'categorized_activity' to identify the most compelling narrative angle (e.g., 
Deep-Dive Educational, High-Hype Feature Launch, or Problem-Solution Refactor). 
Return a strictly formatted CampaignStrategy JSON matching this schema:
{
  "selected_marketing_angle": "string",
  "core_narrative_hook": "string",
  "key_value_proposition": "string"
}
"""

STAGE_2_SYSTEM_PROMPT = """\
You are an elite Technical Copywriter. Your job is to translate an approved 
Campaign Strategy and synthesized data into concrete platform copy.

You will receive:
1. The synthesized context payload.
2. The approved Campaign Strategy (angle, hook, and value proposition).
3. The original goal and target tone.

PLATFORM GUIDELINES:
- LinkedIn: Professional, educational, thought-leadership. 150-300 words.
- Twitter Thread: Punchy, concise. EVERY SINGLE TWEET MUST BE <= 280 CHARACTERS. No exceptions.
- Instagram: Vivid image generation prompt and short punchy caption with hashtags.

CRITICAL RULES:
- Never fabricate data. Translate technical jargon into clear benefits.
- NEVER literally mention internal schema keys like "computed impact score", "summary metrics", or "standard activity". These are for your internal reasoning only.
- Return a strictly formatted MarketingCampaign JSON matching this structure:
{
  "campaign_name": "string",
  "linkedin": {"hook": "string", "body": "string", "hashtags": "string"},
  "twitter": {"tweets": ["string"]},
  "instagram": {"image_generation_prompt": "string", "caption": "string"}
}
"""

# ═══════════════════════════════════════════════════════════════════════════════
#  Core Async Chain
# ═══════════════════════════════════════════════════════════════════════════════

async def transform_data_to_campaign(
    synthesized_context: dict,
    original_goal: str,
    target_tone: str,
) -> MarketingCampaign:
    """Transform synthesized data into a campaign using a Two-Stage Reasoning Chain."""
    
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Missing API key. Set GROQ_API_KEY.")

    client = AsyncGroq(api_key=api_key)

    # ── STAGE 1: The Strategist Pass ─────────────────────────────────────────
    logger.info("🧠 STAGE 1: Generating Campaign Strategy...")
    
    stage_1_message = (
        f"## Creator's Goal\n{original_goal}\n\n"
        f"## Target Tone\n{target_tone}\n\n"
        f"## Synthesized Context\n"
        f"```json\n{json.dumps(synthesized_context, indent=2, default=str)}\n```"
    )

    try:
        response_1 = await client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": STAGE_1_SYSTEM_PROMPT},
                {"role": "user", "content": stage_1_message},
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        raw_text_1 = response_1.choices[0].message.content
        if not raw_text_1:
            raise RuntimeError("Empty response from Groq in Stage 1.")
        strategy = CampaignStrategy.model_validate_json(raw_text_1)
    except Exception as exc:
        logger.error("Stage 1 (Strategy) failed: %s", exc)
        raise RuntimeError(f"Stage 1 (Strategy) failed: {exc}") from exc

    logger.info("✅ Strategy Selected: %s", strategy.selected_marketing_angle)

    # ── STAGE 2: The Copywriter Pass ─────────────────────────────────────────
    logger.info("✍️ STAGE 2: Generating Platform Copy...")
    
    stage_2_message = (
        f"{stage_1_message}\n\n"
        f"## Approved Campaign Strategy\n"
        f"```json\n{strategy.model_dump_json(indent=2)}\n```"
    )

    try:
        response_2 = await client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": STAGE_2_SYSTEM_PROMPT},
                {"role": "user", "content": stage_2_message},
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        raw_text_2 = response_2.choices[0].message.content
        if not raw_text_2:
            raise RuntimeError("Empty response from Groq in Stage 2.")
        campaign = MarketingCampaign.model_validate_json(raw_text_2)
    except Exception as exc:
        logger.error("Stage 2 (Copywriting) failed: %s", exc)
        raise RuntimeError(f"Stage 2 (Copywriting) failed: {exc}") from exc

    # Enforce strict tweet character limits as a safety net
    for i, tweet in enumerate(campaign.twitter.tweets):
        if len(tweet) > 280:
            logger.warning("Tweet %d exceeded 280 chars (%d). Truncating.", i + 1, len(tweet))
            campaign.twitter.tweets[i] = tweet[:277] + "..."

    logger.info("✅ Campaign '%s' fully generated.", campaign.campaign_name)
    return campaign
