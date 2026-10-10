"""Thin service wrapper around the existing LangGraph agent workflow.

No PostgreSQL access here -- same boundary as RagService.
"""

from app.agents.graph import run_agent_workflow
from app.agents.state import AgentState
from app.agents.tools import RetrievalTools
from app.rag import RAG


class AgentService:
    def __init__(self, rag: RAG):
        self.rag = rag
        self.tools = RetrievalTools(rag)

    def run(
        self,
        user_query: str,
        conversation_context: list[dict] | None = None,
        trace_id: str | None = None,
    ) -> AgentState:
        return run_agent_workflow(
            user_query=user_query,
            generator=self.rag.generator,
            tools=self.tools,
            conversation_context=conversation_context or [],
            trace_id=trace_id,
        )
