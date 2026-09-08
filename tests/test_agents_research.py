"""Unit tests for the Research agent (app/agents/research.py).

Generator and RetrievalTools are always mocked -- no real Gemini call, no
real retrieval. These tests check subquestion decomposition, the bounded
sufficiency-check loop, evidence deduplication, fallback behavior, and hard
caps on subquestion/iteration counts.
"""

import json

from unittest.mock import Mock

from app.agents.research import (
    MAX_ADDITIONAL_SUBQUESTIONS,
    MAX_INITIAL_SUBQUESTIONS,
    run_research,
)
from app.agents.state import AgentState
from app.agents.tools import RetrievalTools
from app.models.generator import GenerationUnavailableError


def make_generator(*responses) -> Mock:
    """A Generator mock whose .generate() returns `responses` in order across
    successive calls. Pass a GenerationUnavailableError instance instead of a
    string to make that call raise."""
    generator = Mock()
    generator.generate.side_effect = list(responses)
    return generator


def make_tools(search_results=None) -> Mock:
    tools = Mock(spec=RetrievalTools)
    if search_results is not None:
        tools.search_documents.side_effect = search_results
    else:
        tools.search_documents.return_value = []
    return tools


SUFFICIENT_OK = '{"sufficient": true, "additional_subquestions": [], "reasoning": "Enough evidence."}'


# ---------------------------------------------------------------------------
# Decomposition and retrieval
# ---------------------------------------------------------------------------

def test_run_research_decomposes_and_retrieves_for_each_subquestion():
    state = AgentState(user_query="Compare Evo 1 and Evo 2.")
    generator = make_generator('{"subquestions": ["sub1", "sub2"]}', SUFFICIENT_OK)
    tools = make_tools(
        search_results=[
            [{"document_id": "docA", "chunk_id": 0, "document": "evidence 1"}],
            [{"document_id": "docB", "chunk_id": 0, "document": "evidence 2"}],
        ]
    )

    result = run_research(state, generator, tools)

    assert tools.search_documents.call_count == 2
    called_queries = [call.args[0] for call in tools.search_documents.call_args_list]
    assert called_queries == ["sub1", "sub2"]
    assert result.research_subquestions == ["sub1", "sub2"]
    assert len(result.retrieved_evidence) == 2


def test_run_research_uses_rewritten_query_when_present():
    state = AgentState(
        user_query="What about it?",
        rewritten_query="What is Evo 2's context window?",
    )
    generator = make_generator('{"subquestions": ["sub"]}', SUFFICIENT_OK)
    tools = make_tools()

    run_research(state, generator, tools)

    decompose_prompt = generator.generate.call_args_list[0].args[0]
    assert "Evo 2's context window" in decompose_prompt


# ---------------------------------------------------------------------------
# Decomposition fallback
# ---------------------------------------------------------------------------

def test_run_research_falls_back_to_single_subquestion_on_malformed_decomposition():
    state = AgentState(user_query="a question")
    generator = make_generator("not json at all", SUFFICIENT_OK)
    tools = make_tools()

    result = run_research(state, generator, tools)

    assert result.research_subquestions == ["a question"]
    assert result.agent_trace[0].stage == "research_decompose"
    assert result.agent_trace[0].status == "fallback"


def test_run_research_falls_back_on_decomposition_generation_unavailable():
    state = AgentState(user_query="a question")
    generator = make_generator(GenerationUnavailableError("down"), SUFFICIENT_OK)
    tools = make_tools()

    result = run_research(state, generator, tools)

    assert result.research_subquestions == ["a question"]


def test_run_research_caps_initial_subquestions_to_max():
    state = AgentState(user_query="q")
    many_subs = [f"sub{i}" for i in range(10)]
    generator = make_generator(json.dumps({"subquestions": many_subs}), SUFFICIENT_OK)
    tools = make_tools()

    result = run_research(state, generator, tools)

    assert len(result.research_subquestions) == MAX_INITIAL_SUBQUESTIONS
    assert tools.search_documents.call_count == MAX_INITIAL_SUBQUESTIONS


# ---------------------------------------------------------------------------
# Sufficiency loop
# ---------------------------------------------------------------------------

def test_run_research_stops_after_one_iteration_when_sufficient():
    state = AgentState(user_query="q")
    generator = make_generator('{"subquestions": ["sub1"]}', SUFFICIENT_OK)
    tools = make_tools(search_results=[[{"document_id": "docA", "chunk_id": 0, "document": "e"}]])

    result = run_research(state, generator, tools)

    assert generator.generate.call_count == 2
    assert tools.search_documents.call_count == 1
    assert result.research_findings.sufficient is True


def test_run_research_iterates_once_more_when_insufficient_then_stops_at_cap():
    state = AgentState(user_query="q")
    generator = make_generator(
        '{"subquestions": ["sub1"]}',
        '{"sufficient": false, "additional_subquestions": ["sub2"], "reasoning": "need more"}',
    )
    tools = make_tools(
        search_results=[
            [{"document_id": "docA", "chunk_id": 0, "document": "e1"}],
            [{"document_id": "docB", "chunk_id": 0, "document": "e2"}],
        ]
    )

    result = run_research(state, generator, tools)

    # Decompose + exactly one sufficiency check = 2 LLM calls total, even
    # though a second retrieval round happens -- the iteration cap prevents
    # a third LLM call (a second sufficiency check).
    assert generator.generate.call_count == 2
    assert tools.search_documents.call_count == 2
    assert result.research_subquestions == ["sub1", "sub2"]
    assert result.research_findings.sufficient is False
    assert "maximum research iterations" in result.research_findings.reasoning
    assert len(result.retrieved_evidence) == 2


def test_run_research_stops_when_insufficient_but_no_additional_subquestions_given():
    state = AgentState(user_query="q")
    generator = make_generator(
        '{"subquestions": ["sub1"]}',
        '{"sufficient": false, "additional_subquestions": [], "reasoning": "need more but none suggested"}',
    )
    tools = make_tools()

    result = run_research(state, generator, tools)

    assert result.research_findings.sufficient is True
    assert tools.search_documents.call_count == 1


def test_run_research_falls_back_to_sufficient_on_sufficiency_check_failure():
    state = AgentState(user_query="q")
    generator = make_generator('{"subquestions": ["sub1"]}', "garbage, not json")
    tools = make_tools()

    result = run_research(state, generator, tools)

    assert result.research_findings.sufficient is True
    assert generator.generate.call_count == 2
    assert tools.search_documents.call_count == 1


def test_run_research_caps_additional_subquestions_to_max():
    state = AgentState(user_query="q")
    many_additional = [f"extra{i}" for i in range(10)]
    generator = make_generator(
        '{"subquestions": ["sub1"]}',
        json.dumps(
            {"sufficient": False, "additional_subquestions": many_additional, "reasoning": "need more"}
        ),
    )
    tools = make_tools()

    result = run_research(state, generator, tools)

    assert len(result.research_subquestions) == 1 + MAX_ADDITIONAL_SUBQUESTIONS
    assert tools.search_documents.call_count == 1 + MAX_ADDITIONAL_SUBQUESTIONS


# ---------------------------------------------------------------------------
# Evidence deduplication
# ---------------------------------------------------------------------------

def test_run_research_dedupes_evidence_across_subquestions():
    state = AgentState(user_query="q")
    generator = make_generator('{"subquestions": ["sub1", "sub2"]}', SUFFICIENT_OK)
    tools = make_tools(
        search_results=[
            [{"document_id": "docA", "chunk_id": 0, "document": "shared"}],
            [
                {"document_id": "docA", "chunk_id": 0, "document": "shared"},
                {"document_id": "docB", "chunk_id": 1, "document": "unique"},
            ],
        ]
    )

    result = run_research(state, generator, tools)

    assert len(result.retrieved_evidence) == 2
    ids = {(item["document_id"], item["chunk_id"]) for item in result.retrieved_evidence}
    assert ids == {("docA", 0), ("docB", 1)}


# ---------------------------------------------------------------------------
# Trace, state immutability, trace_id propagation
# ---------------------------------------------------------------------------

def test_run_research_records_trace_entries_for_each_stage():
    state = AgentState(user_query="q")
    generator = make_generator('{"subquestions": ["sub1"]}', SUFFICIENT_OK)
    tools = make_tools()

    result = run_research(state, generator, tools)

    stages = [entry.stage for entry in result.agent_trace]
    assert stages == ["research_decompose", "research_retrieve", "research_sufficiency"]
    assert result.agent_trace[1].tools_used == ["search_documents"]


def test_run_research_does_not_mutate_input_state():
    state = AgentState(user_query="q")
    generator = make_generator('{"subquestions": ["sub1"]}', SUFFICIENT_OK)
    tools = make_tools()

    run_research(state, generator, tools)

    assert state.research_subquestions == []
    assert state.retrieved_evidence == []
    assert state.research_findings is None
    assert state.agent_trace == []


def test_run_research_uses_state_trace_id_when_not_passed_explicitly():
    state = AgentState(user_query="q", trace_id="state-trace")
    generator = make_generator('{"subquestions": ["sub1"]}', SUFFICIENT_OK)
    tools = make_tools()

    run_research(state, generator, tools)

    assert tools.search_documents.call_args.kwargs["trace_id"] == "state-trace"
    assert generator.generate.call_args_list[0].kwargs["trace_id"] == "state-trace"
