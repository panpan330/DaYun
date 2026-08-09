import pytest

from app.schemas.multiturn_eval import (
    MultiturnEvalCase,
    MultiturnEvalDataset,
    load_multiturn_cases,
)


def test_multiturn_case_roundtrip() -> None:
    case = MultiturnEvalCase(
        id="mt_order_refund_001",
        name="跨轮：查订单后问退款",
        conversation=[
            {
                "message": "我的订单 202501010001 发货了吗",
                "expected_intent": "order_query",
                "expected_route": "query_order",
                "requires_history": False,
            },
            {
                "message": "那我要退款",
                "expected_intent": "refund_request",
                "expected_route": "refund_order",
                "requires_history": True,
            },
        ],
        final_answer={"expected_intent": "refund_request"},
    )
    assert len(case.conversation) == 2
    assert case.conversation[1].requires_history is True
    assert case.final_answer is not None and case.final_answer.expected_intent == "refund_request"


def test_multiturn_case_requires_at_least_two_turns() -> None:
    with pytest.raises(ValueError):
        MultiturnEvalCase(
            id="single",
            name="single turn",
            conversation=[
                {
                    "message": "hi",
                    "expected_intent": "neutral",
                    "expected_route": "unsupported",
                    "requires_history": False,
                },
            ],
        )


def test_multiturn_dataset_requires_unique_ids() -> None:
    with pytest.raises(ValueError):
        MultiturnEvalDataset(
            schema_version="1.0",
            cases=[
                MultiturnEvalCase(
                    id="a",
                    name="a",
                    conversation=[
                        {
                            "message": "hi",
                            "expected_intent": "neutral",
                            "expected_route": "unsupported",
                            "requires_history": False,
                        },
                        {
                            "message": "hi2",
                            "expected_intent": "neutral",
                            "expected_route": "unsupported",
                            "requires_history": False,
                        },
                    ],
                ),
                MultiturnEvalCase(
                    id="a",
                    name="b",
                    conversation=[
                        {
                            "message": "hi",
                            "expected_intent": "neutral",
                            "expected_route": "unsupported",
                            "requires_history": False,
                        },
                        {
                            "message": "hi2",
                            "expected_intent": "neutral",
                            "expected_route": "unsupported",
                            "requires_history": False,
                        },
                    ],
                ),
            ],
        )


def test_load_multiturn_cases_from_json() -> None:
    cases = load_multiturn_cases("data/agent_eval/multiturn_cases.json")
    assert len(cases) >= 4
    assert all(len(case.conversation) >= 2 for case in cases)
    assert any(turn.requires_history for case in cases for turn in case.conversation)
