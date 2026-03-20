"""
nodes.py (PRODUCTION FINAL - NO FALLBACK VERSION)

DESIGN:
✔ NO fallback models (deterministic system)
✔ Detect Gemini quota exhaustion instantly
✔ No retry storm (max_retries=0)
✔ Never crash graph
✔ Clear signal: "FREE_TIER_EXHAUSTED"
✔ Strong logging + observability
✔ Async-safe + timeout-safe

BEHAVIOR:
Gemini OK         → Normal response
Quota exceeded    → "FREE_TIER_EXHAUSTED"
Other error       → "LLM_ERROR"
"""

import os
import sys
import asyncio
from typing import List, Any, Dict

from langsmith import traceable
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from app.graph.state import ReviewState
from app.mcp.github_client import GitHubClient
from app.core.exceptions import CustomException
from app.core.logger import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

LLM_TIMEOUT = 15
LLM_SEMAPHORE = asyncio.Semaphore(2)

FREE_TIER_MSG = "FREE_TIER_EXHAUSTED"
LLM_ERROR_MSG = "LLM_ERROR"


# ─────────────────────────────────────────────
# LLM INITIALIZATION
# ─────────────────────────────────────────────

def get_llm():
    """
    Initialize Gemini model with NO retry.
    Prevents internal retry storms.
    """
    if not os.environ.get("GOOGLE_API_KEY"):
        raise CustomException("GOOGLE_API_KEY missing", sys)

    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        max_retries=0,  # 🔥 CRITICAL FIX
    )


# ─────────────────────────────────────────────
# SAFE LLM CALL
# ─────────────────────────────────────────────

async def safe_llm_invoke(messages: List[Any]) -> str:
    """
    PRODUCTION SAFE LLM CALL

    Guarantees:
    ✔ No crash
    ✔ No retry storm
    ✔ Fast failure on quota
    ✔ Deterministic output

    Returns:
        str:
            - Normal response
            - FREE_TIER_EXHAUSTED
            - LLM_ERROR
    """

    async with LLM_SEMAPHORE:
        try:
            logger.info("[LLM] Invoking Gemini")

            llm = get_llm()

            response = await asyncio.wait_for(
                llm.ainvoke(messages),
                timeout=LLM_TIMEOUT
            )

            logger.info("[LLM] Success")
            return response.content

        except asyncio.TimeoutError:
            logger.error("[LLM] Timeout")
            return LLM_ERROR_MSG

        except Exception as e:
            msg = str(e)

            # 🔥 QUOTA DETECTION (MAIN FIX)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                logger.error("[LLM] FREE TIER EXHAUSTED")

                return FREE_TIER_MSG

            # 🔥 UNKNOWN ERROR
            logger.exception("[LLM] Unexpected failure")
            return LLM_ERROR_MSG


# ─────────────────────────────────────────────
# NODES
# ─────────────────────────────────────────────

@traceable(name="fetch_diff_node")
async def fetch_diff_node(state: ReviewState) -> Dict:
    try:
        logger.info(f"[NODE] Fetching PR #{state['pr_number']}")

        client = GitHubClient()

        diff = client.get_pr_diff(
            state["owner"], state["repo"], state["pr_number"]
        )

        return {"diff": diff, "error": False}

    except Exception:
        logger.exception("[NODE] Fetch failed")

        return {
            "error": True,
            "error_reason": "github_fetch_failed",
        }


# ─────────────────────────────────────────────

@traceable(name="analyze_code_node")
async def analyze_code_node(state: ReviewState) -> Dict:
    if state.get("error"):
        return {}

    logger.info("[NODE] Analyze Code")

    try:
        prompt = [
            SystemMessage(content="You are a senior code reviewer."),
            HumanMessage(content=state["diff"][:4000]),
        ]

        result = await safe_llm_invoke(prompt)

        # 🔥 HANDLE LLM FAILURE STATES
        if result == FREE_TIER_MSG:
            logger.warning("[ANALYZE] Skipped due to quota")

            return {
                "issues": ["LLM unavailable: FREE TIER EXHAUSTED"],
                "error": False,
            }

        if result == LLM_ERROR_MSG:
            logger.error("[ANALYZE] LLM internal error")

            return {
                "issues": ["LLM error during analysis"],
                "error": False,
            }

        return {
            "issues": [result],
            "error": False,
        }

    except Exception:
        logger.exception("[NODE] Analyze failed")

        return {
            "issues": [],
            "error": True,
            "error_reason": "analysis_failed",
        }


# ─────────────────────────────────────────────

@traceable(name="reflect_node")
async def reflect_node(state: ReviewState) -> Dict:
    logger.info("[NODE] Reflect")

    return {
        "reflection_count": state.get("reflection_count", 0) + 1
    }


# ─────────────────────────────────────────────

@traceable(name="lint_node")
async def lint_node(state: ReviewState) -> Dict:
    logger.info("[NODE] Lint")

    code = state.get("diff", "").lower()
    passed = "todo" not in code

    if not passed:
        logger.warning("[LINT] Failed")

    return {"lint_passed": passed}


# ─────────────────────────────────────────────

@traceable(name="refactor_node")
async def refactor_node(state: ReviewState) -> Dict:
    if state.get("lint_passed", True):
        logger.info("[REFACTOR] Skipped")
        return {}

    logger.info("[NODE] Refactor")

    try:
        prompt = [
            SystemMessage(content="Fix code issues"),
            HumanMessage(content=state["diff"][:2000]),
        ]

        result = await safe_llm_invoke(prompt)

        if result in (FREE_TIER_MSG, LLM_ERROR_MSG):
            logger.warning("[REFACTOR] Skipped due to LLM issue")

            return {
                "suggestions": ["Refactor skipped (LLM unavailable)"]
            }

        return {
            "refactor_count": state.get("refactor_count", 0) + 1,
            "suggestions": [result],
        }

    except Exception:
        logger.exception("[NODE] Refactor failed")

        return {"error": True, "error_reason": "refactor_failed"}


# ─────────────────────────────────────────────

@traceable(name="validator_node")
async def validator_node(state: ReviewState) -> Dict:
    logger.info("[NODE] Validate")

    return {
        "validation_passed": state.get("refactor_count", 0) >= 1
    }


# ─────────────────────────────────────────────

@traceable(name="verdict_node")
async def verdict_node(state: ReviewState) -> Dict:
    logger.info("[NODE] Verdict")

    if state.get("error"):
        return {
            "verdict": "FAILED",
            "summary": state.get("error_reason", "Unknown error"),
        }

    if state.get("human_decision") == "rejected":
        return {"verdict": "REJECTED"}

    issues = state.get("issues", [])

    return {
        "verdict": "REQUEST_CHANGES" if issues else "APPROVE",
        "summary": f"{len(issues)} issues found",
    }