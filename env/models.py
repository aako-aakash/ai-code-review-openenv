"""
Pydantic models for the AI Code Review OpenEnv environment.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
#                                     Action
# ---------------------------------------------------------------------------

VALID_ACTION_TYPES = {
    "analyze_code",
    "flag_bug",
    "suggest_fix",
    "approve_pr",
    "request_changes",
    "suggest_improvement",
}


class Action(BaseModel):
    """An action submitted by the agent."""

    action_type: str = Field(
        ...,
        description=(
            "One of: analyze_code | flag_bug | suggest_fix | "
            "approve_pr | request_changes | suggest_improvement"
        ),
    )
    target: Optional[str] = Field(
        None,
        description="The bug/issue label the action targets, if applicable.",
    )
    description: Optional[str] = Field(
        None,
        description="Free-text reasoning or suggestion text.",
    )
    line_number: Optional[int] = Field(
        None,
        description="Optional line number the action refers to.",
    )
    extra: Dict[str, Any] = Field(
        default_factory=dict,
        description="Any additional metadata.",
    )

    def is_valid(self) -> bool:
        return self.action_type in VALID_ACTION_TYPES


# ---------------------------------------------------------------------------
#                                Observation
# ---------------------------------------------------------------------------


class Observation(BaseModel):
    """What the agent sees after every step (and on reset)."""

    task_id: str = Field(..., description="Identifier of the current task.")
    task_goal: str = Field(..., description="Natural-language description of the task goal.")
    code_snippet: str = Field(..., description="The code under review.")
    detected_issues: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Issues flagged so far in the current episode.",
    )
    suggested_fixes: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Fixes suggested so far in the current episode.",
    )
    suggested_improvements: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Improvements suggested so far.",
    )
    review_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Chronological list of (action, reward) pairs.",
    )
    remaining_steps: int = Field(..., description="Steps left before episode ends.")
    pr_approved: bool = Field(False, description="Whether the PR was approved.")
    pr_changes_requested: bool = Field(
        False, description="Whether changes were requested."
    )
    cumulative_reward: float = Field(0.0, description="Total reward accumulated so far.")
    message: str = Field("", description="Human-readable feedback on the last action.")


# ---------------------------------------------------------------------------
#                 Internal bug / fix / improvement records
# ---------------------------------------------------------------------------


class BugRecord(BaseModel):
    bug_id: str
    description: str
    line_number: Optional[int] = None
    bug_type: str  # e.g. "syntax", "logic", "security", "performance"


class FixRecord(BaseModel):
    bug_id: str
    description: str
    correct_fix: str  # canonical correct fix text (lower-cased for comparison)


class ImprovementRecord(BaseModel):
    improvement_id: str
    description: str
    keywords: List[str]  # keywords that must appear in agent's suggestion


# ---------------------------------------------------------------------------
#                                    Task definition
# ---------------------------------------------------------------------------


class TaskDefinition(BaseModel):
    task_id: str
    title: str
    difficulty: str  # easy | medium | hard
    goal: str
    code_snippet: str
    bugs: List[BugRecord]
    fixes: List[FixRecord]
    improvements: List[ImprovementRecord]
    max_steps: int = 20
