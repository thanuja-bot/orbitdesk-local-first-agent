from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field


Classification = Literal[
    "answerable",
    "requires_clarification",
    "requires_escalation",
    "out_of_scope",
    "safe_failure",
]


class SourceReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(min_length=1)
    passage: str = Field(min_length=1)


class SupportResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    classification: Classification
    answer: str = Field(min_length=1)
    sources: list[SourceReference]
    confidence: float = Field(ge=0.0, le=1.0)
    requires_human: bool
    reason: str = Field(min_length=1)
    clarification_question: str | None = None
    warnings: list[str] = Field(default_factory=list)


class Passage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    title: str
    text: str
    source_type: Literal["knowledge_base", "resolved_case"]
    status: Literal["current", "resolved", "superseded"]
    score: float = 0.0
    priority: int = 0


class TraceEvent(BaseModel):
    node: str
    event: Literal["enter", "exit", "route", "error"]
    detail: str = ""


class AgentState(TypedDict, total=False):
    question: str
    classification: Classification
    triage_reason: str
    passages: list[Passage]
    draft: SupportResponse
    response: SupportResponse
    verification_errors: list[str]
    revision_count: int
    max_revisions: int
    trace: list[dict[str, Any]]
    force_bad_first_attempt: bool
    model_latency_ms: float