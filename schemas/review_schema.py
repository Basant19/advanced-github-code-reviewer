"""
schemas/review_schema.py

Pydantic v2 schemas for the review domain.

Used by:
  - app/api/routes/review.py      — request validation + response serialization
  - app/services/review_service.py — constructing ReviewStep audit records
  - test/                          — response validation in route tests

P1 schemas (unchanged):
  TriggerReviewRequest   — POST /reviews/trigger request body
  ReviewStepResponse     — single node execution record
  ReviewResponse         — full review with all steps

P2 schema (new):
  SandboxResultSchema    — serializes SandboxResult dataclass for API
                           responses and PostgreSQL review_steps storage.
                           Used when review_service stores lint_result and
                           validation_result in the review_steps audit trail.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# P2 — NEW
# ---------------------------------------------------------------------------

class SandboxResultSchema(BaseModel):
    """
    Pydantic representation of app.sandbox.docker_runner.SandboxResult.

    Used to serialize sandbox results for:
      1. review_steps.output_data column (JSON string in PostgreSQL)
      2. GET /reviews/{review_id} API response — shows lint/test outcome
      3. Streamlit dashboard — displays ruff/pytest output to human reviewer

    Note: SandboxResult is a dataclass (not a Pydantic model) because it
    lives in app/sandbox/ which must not import from schemas/. This schema
    is the Pydantic twin used at the API and service layers.
    """

    passed      : bool = Field(
        description="True if exit_code == 0 — ruff/pytest found no errors"
    )
    output      : str = Field(
        description="Full stdout from the tool run inside the container"
    )
    errors      : str = Field(
        default="",
        description="Stderr output — empty string if none"
    )
    exit_code   : int = Field(
        description="Raw Docker container exit code (0=pass, non-zero=findings)"
    )
    duration_ms : int = Field(
        description="Wall-clock milliseconds for the full container lifecycle"
    )
    tool        : str = Field(
        description="Which tool ran: 'lint' (ruff) or 'test' (ruff + pytest)"
    )
    image       : str = Field(
        default="code-reviewer-sandbox:latest",
        description="Docker image used — included for audit trail"
    )

    @classmethod
    def from_dataclass(cls, result) -> "SandboxResultSchema":
        """
        Convert a SandboxResult dataclass instance into this Pydantic schema.

        Usage in review_service.py:
            schema = SandboxResultSchema.from_dataclass(lint_result)
            step.output_data = schema.model_dump_json()
        """
        return cls(
            passed      = result.passed,
            output      = result.output,
            errors      = result.errors,
            exit_code   = result.exit_code,
            duration_ms = result.duration_ms,
            tool        = result.tool,
            image       = result.image,
        )

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# P1 — UNCHANGED
# ---------------------------------------------------------------------------

class TriggerReviewRequest(BaseModel):
    """
    Request body for POST /reviews/trigger.
    Manually triggers a review without a GitHub webhook.
    """
    owner      : str = Field(description="GitHub repository owner login")
    repo       : str = Field(description="GitHub repository name")
    pr_number  : int = Field(description="Pull request number to review")


class ReviewStepResponse(BaseModel):
    """
    Single node execution record from the review_steps table.
    One record per node: fetch_diff, analyze_code, lint, refactor,
    validate, verdict.
    """
    id          : int
    review_id   : int
    step_name   : str
    status      : str
    input_data  : Optional[str] = None
    output_data : Optional[str] = None

    model_config = {"from_attributes": True}


class ReviewResponse(BaseModel):
    """
    Full review record with all node steps.
    Returned by GET /reviews/{review_id}.
    """
    id             : int
    pull_request_id: int
    reviewer       : str
    status         : str
    verdict        : Optional[str] = None
    summary        : Optional[str] = None
    steps          : list[ReviewStepResponse] = []

    model_config = {"from_attributes": True}