"""Pydantic state and structured-output models for the Phase 9 agent workflow.

This module defines data only -- no LangGraph nodes, no LLM calls, no tool
implementations. `AgentState` is the single object threaded through the
LangGraph workflow (Router -> Retrieval/Research -> Answer -> Validation);
`RouterDecision` and `ValidationResult` are the structured outputs those
stages parse LLM responses into, so an agent's output is validated against a
schema rather than trusted as free text.

`retrieved_evidence` intentionally reuses the same shape as
`RAG.retrieve()`'s `final_evidence` (document_id, filename, page_number,
chunk_id, rerank_score, document) rather than a new schema, so the existing
evidence/citation architecture from Phases 1-8 carries into the agent
workflow unchanged.
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class RouteType(str, Enum):
    """The request types the Router agent can select between."""

    DOCUMENT_QA = "document_qa"
    RESEARCH = "research"
    SUMMARY = "summary"
    COMPARISON = "comparison"


class RouterDecision(BaseModel):
    """Structured output of the Router agent.

    `extra="forbid"` ensures a malformed or hallucinated field from the LLM's
    structured output surfaces as a validation error instead of being
    silently ignored.
    """

    model_config = ConfigDict(extra="forbid")

    route: RouteType
    reasoning: str = ""


class ResearchFindings(BaseModel):
    """Structured result produced by the Research agent."""

    model_config = ConfigDict(extra="forbid")

    subquestions: list[str] = Field(default_factory=list)
    evidence: list[dict] = Field(default_factory=list)
    sufficient: bool = False
    reasoning: str = ""


class ValidationResult(BaseModel):
    """Structured output of the Validation agent.

    Individual check fields (`claims_supported`, `citation_correct`,
    `addresses_question`) are kept separate from the overall `is_valid`
    verdict so a failure can be attributed to a specific check rather than
    only a pass/fail bit.
    """

    model_config = ConfigDict(extra="forbid")

    is_valid: bool
    claims_supported: bool
    citation_correct: bool
    addresses_question: bool
    unsupported_claims: list[str] = Field(default_factory=list)
    citation_issues: list[str] = Field(default_factory=list)
    reasoning: str = ""


class AgentTraceEntry(BaseModel):
    """One recorded step of the agent workflow, for the Streamlit Agent Trace view."""

    model_config = ConfigDict(extra="forbid")

    stage: str
    status: str
    duration_seconds: float | None = None
    decision: str | None = None
    tools_used: list[str] = Field(default_factory=list)


class AgentState(BaseModel):
    """The state threaded through the LangGraph agent workflow.

    Field types intentionally mirror shapes already used by `app.rag`
    (`conversation_context` and `retrieved_evidence` are plain `list[dict]`,
    matching `conversation_history` and `final_evidence`) rather than
    introducing parallel schemas for the same data.
    """

    model_config = ConfigDict(extra="forbid")

    user_query: str
    conversation_context: list[dict] = Field(default_factory=list)
    selected_route: RouteType | None = None
    rewritten_query: str | None = None
    research_subquestions: list[str] = Field(default_factory=list)
    retrieved_evidence: list[dict] = Field(default_factory=list)
    research_findings: ResearchFindings | None = None
    draft_answer: str | None = None
    validation_result: ValidationResult | None = None
    retry_count: int = Field(default=0, ge=0)
    final_answer: str | None = None
    trace_id: str | None = None
    agent_trace: list[AgentTraceEntry] = Field(default_factory=list)
