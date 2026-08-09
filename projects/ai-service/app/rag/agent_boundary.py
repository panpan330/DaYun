from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field

from app.rag.knowledge_routing import RagKnowledgeRouteDecision
from app.rag.query_intent import (
    QueryIntent,
    QueryIntentClassification,
    classify_query_intent,
)
from app.schemas.tool import ToolAccessLevel
from app.tools.tool_registry import get_tool_definition


RagAgentResponsibility = Literal[
    "rag",
    "agent",
    "tool",
    "direct_answer",
    "safety",
    "clarification",
]
RagAgentBoundaryAction = Literal[
    "retrieve_knowledge",
    "use_rag_as_agent_context",
    "call_read_tool",
    "run_agent_workflow",
    "request_user_confirmation",
    "answer_directly",
    "block_for_safety",
    "ask_clarifying_question",
    "reject_tool_execution",
]
RagAgentBoundaryWarning = Literal[
    "RAG_AGENT_BOUNDARY_RAG_ONLY",
    "RAG_AGENT_BOUNDARY_AGENT_ORCHESTRATES_RAG",
    "RAG_AGENT_BOUNDARY_TOOL_READ_ONLY",
    "RAG_AGENT_BOUNDARY_WRITE_REQUIRES_CONFIRMATION",
    "RAG_AGENT_BOUNDARY_TOOL_NOT_ALLOWED",
    "RAG_AGENT_BOUNDARY_MISSING_TOOL_ARGUMENT",
    "RAG_AGENT_BOUNDARY_NOT_RAG_TASK",
    "RAG_AGENT_BOUNDARY_SAFETY_BLOCK",
]


class RagAgentBoundaryDecision(BaseModel):
    normalized_query: str
    intent: QueryIntent
    primary_owner: RagAgentResponsibility
    should_use_rag: bool
    should_use_agent: bool
    should_call_tool: bool
    should_require_confirmation: bool
    selected_tool_name: str | None = None
    selected_tool_access_level: ToolAccessLevel | None = None
    selected_knowledge_base_ids: list[str] = Field(default_factory=list)
    actions: list[RagAgentBoundaryAction] = Field(default_factory=list)
    warnings: list[RagAgentBoundaryWarning] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    debug_lines: list[str] = Field(default_factory=list)


def build_rag_agent_boundary_decision(
    query: str,
    *,
    classification: QueryIntentClassification | None = None,
    route_decision: RagKnowledgeRouteDecision | None = None,
    requested_tool_name: str | None = None,
    agent_needs_policy_context: bool = False,
) -> RagAgentBoundaryDecision:
    selected_classification = classification or classify_query_intent(query)
    normalized_query = selected_classification.normalized_query

    if requested_tool_name is not None:
        tool_decision = _build_tool_boundary_decision(
            normalized_query=normalized_query,
            intent=selected_classification.intent,
            requested_tool_name=requested_tool_name,
        )
        return _with_debug_lines(tool_decision)

    decision = _build_intent_boundary_decision(
        normalized_query=normalized_query,
        classification=selected_classification,
        route_decision=route_decision,
        agent_needs_policy_context=agent_needs_policy_context,
    )
    return _with_debug_lines(decision)


def format_rag_agent_boundary_decision(
    decision: RagAgentBoundaryDecision,
) -> list[str]:
    lines = [
        (
            f"intent={decision.intent} owner={decision.primary_owner} "
            f"use_rag={decision.should_use_rag} "
            f"use_agent={decision.should_use_agent} "
            f"call_tool={decision.should_call_tool} "
            f"confirm={decision.should_require_confirmation}"
        ),
        (
            f"tool={decision.selected_tool_name or '-'} "
            f"tool_access={decision.selected_tool_access_level or '-'} "
            f"kbs={','.join(decision.selected_knowledge_base_ids) or '-'}"
        ),
        (
            f"actions={','.join(decision.actions) or '-'} "
            f"warnings={','.join(decision.warnings) or '-'} "
            f"reasons={','.join(decision.reasons) or '-'}"
        ),
    ]
    return lines


def _build_intent_boundary_decision(
    *,
    normalized_query: str,
    classification: QueryIntentClassification,
    route_decision: RagKnowledgeRouteDecision | None,
    agent_needs_policy_context: bool,
) -> RagAgentBoundaryDecision:
    if classification.intent == "unsafe":
        return RagAgentBoundaryDecision(
            normalized_query=normalized_query,
            intent=classification.intent,
            primary_owner="safety",
            should_use_rag=False,
            should_use_agent=False,
            should_call_tool=False,
            should_require_confirmation=False,
            actions=["block_for_safety"],
            warnings=["RAG_AGENT_BOUNDARY_SAFETY_BLOCK"],
            reasons=["unsafe queries should be blocked before RAG, Agent, or tools"],
        )

    if classification.intent == "unclear":
        return RagAgentBoundaryDecision(
            normalized_query=normalized_query,
            intent=classification.intent,
            primary_owner="clarification",
            should_use_rag=False,
            should_use_agent=False,
            should_call_tool=False,
            should_require_confirmation=False,
            actions=["ask_clarifying_question"],
            warnings=["RAG_AGENT_BOUNDARY_NOT_RAG_TASK"],
            reasons=["unclear queries need more user information before routing"],
        )

    if classification.intent == "smalltalk":
        return RagAgentBoundaryDecision(
            normalized_query=normalized_query,
            intent=classification.intent,
            primary_owner="direct_answer",
            should_use_rag=False,
            should_use_agent=False,
            should_call_tool=False,
            should_require_confirmation=False,
            actions=["answer_directly"],
            warnings=["RAG_AGENT_BOUNDARY_NOT_RAG_TASK"],
            reasons=["smalltalk does not need knowledge retrieval or tool execution"],
        )

    if classification.intent == "order_lookup":
        has_business_entity = bool(classification.preserved_entities)
        return RagAgentBoundaryDecision(
            normalized_query=normalized_query,
            intent=classification.intent,
            primary_owner="tool" if has_business_entity else "clarification",
            should_use_rag=False,
            should_use_agent=False,
            should_call_tool=has_business_entity,
            should_require_confirmation=False,
            selected_tool_name="query_order" if has_business_entity else None,
            selected_tool_access_level=(
                ToolAccessLevel.READ if has_business_entity else None
            ),
            actions=["call_read_tool"] if has_business_entity else ["ask_clarifying_question"],
            warnings=(
                ["RAG_AGENT_BOUNDARY_TOOL_READ_ONLY"]
                if has_business_entity
                else ["RAG_AGENT_BOUNDARY_MISSING_TOOL_ARGUMENT"]
            ),
            reasons=(
                ["order status is live business data and should be read through a tool"]
                if has_business_entity
                else ["order lookup needs an explicit order id before tool execution"]
            ),
        )

    if classification.intent == "ticket_creation":
        return RagAgentBoundaryDecision(
            normalized_query=normalized_query,
            intent=classification.intent,
            primary_owner="agent",
            should_use_rag=agent_needs_policy_context,
            should_use_agent=True,
            should_call_tool=False,
            should_require_confirmation=True,
            actions=(
                ["use_rag_as_agent_context", "run_agent_workflow", "request_user_confirmation"]
                if agent_needs_policy_context
                else ["run_agent_workflow", "request_user_confirmation"]
            ),
            warnings=(
                [
                    "RAG_AGENT_BOUNDARY_AGENT_ORCHESTRATES_RAG",
                    "RAG_AGENT_BOUNDARY_WRITE_REQUIRES_CONFIRMATION",
                ]
                if agent_needs_policy_context
                else ["RAG_AGENT_BOUNDARY_WRITE_REQUIRES_CONFIRMATION"]
            ),
            reasons=[
                "ticket creation is a multi-step workflow owned by the Agent",
                "write operations must be confirmed before execution",
            ],
        )

    if classification.should_use_rag:
        selected_knowledge_base_ids = _knowledge_base_ids(route_decision)
        return RagAgentBoundaryDecision(
            normalized_query=normalized_query,
            intent=classification.intent,
            primary_owner="rag",
            should_use_rag=True,
            should_use_agent=False,
            should_call_tool=False,
            should_require_confirmation=False,
            selected_knowledge_base_ids=selected_knowledge_base_ids,
            actions=["retrieve_knowledge"],
            warnings=["RAG_AGENT_BOUNDARY_RAG_ONLY"],
            reasons=[
                "policy and process questions should retrieve grounded knowledge before answering"
            ],
        )

    return RagAgentBoundaryDecision(
        normalized_query=normalized_query,
        intent=classification.intent,
        primary_owner="clarification",
        should_use_rag=False,
        should_use_agent=False,
        should_call_tool=False,
        should_require_confirmation=False,
        actions=["ask_clarifying_question"],
        warnings=["RAG_AGENT_BOUNDARY_NOT_RAG_TASK"],
        reasons=["query does not match a supported RAG, Agent, or tool boundary"],
    )


def _build_tool_boundary_decision(
    *,
    normalized_query: str,
    intent: QueryIntent,
    requested_tool_name: str,
) -> RagAgentBoundaryDecision:
    normalized_tool_name = requested_tool_name.strip()
    if not normalized_tool_name:
        raise ValueError("requested_tool_name must not be blank")

    definition = get_tool_definition(normalized_tool_name)
    if definition is None or not definition.enabled:
        return RagAgentBoundaryDecision(
            normalized_query=normalized_query,
            intent=intent,
            primary_owner="safety",
            should_use_rag=False,
            should_use_agent=False,
            should_call_tool=False,
            should_require_confirmation=False,
            selected_tool_name=normalized_tool_name,
            actions=["reject_tool_execution"],
            warnings=["RAG_AGENT_BOUNDARY_TOOL_NOT_ALLOWED"],
            reasons=["tool is missing or disabled and must not be executed"],
        )

    if definition.access_level == ToolAccessLevel.READ and not definition.requires_confirmation:
        return RagAgentBoundaryDecision(
            normalized_query=normalized_query,
            intent=intent,
            primary_owner="tool",
            should_use_rag=False,
            should_use_agent=False,
            should_call_tool=True,
            should_require_confirmation=False,
            selected_tool_name=definition.name,
            selected_tool_access_level=definition.access_level,
            actions=["call_read_tool"],
            warnings=["RAG_AGENT_BOUNDARY_TOOL_READ_ONLY"],
            reasons=["read-only tools can execute after backend argument validation"],
        )

    return RagAgentBoundaryDecision(
        normalized_query=normalized_query,
        intent=intent,
        primary_owner="agent",
        should_use_rag=False,
        should_use_agent=True,
        should_call_tool=False,
        should_require_confirmation=True,
        selected_tool_name=definition.name,
        selected_tool_access_level=definition.access_level,
        actions=["run_agent_workflow", "request_user_confirmation"],
        warnings=["RAG_AGENT_BOUNDARY_WRITE_REQUIRES_CONFIRMATION"],
        reasons=[
            "write or sensitive tools must be orchestrated by an Agent and confirmed by the user"
        ],
    )


def _knowledge_base_ids(
    route_decision: RagKnowledgeRouteDecision | None,
) -> list[str]:
    if route_decision is None:
        return []
    return [route.knowledge_base_id for route in route_decision.routes]


def _with_debug_lines(
    decision: RagAgentBoundaryDecision,
) -> RagAgentBoundaryDecision:
    return decision.model_copy(
        update={"debug_lines": format_rag_agent_boundary_decision(decision)}
    )
