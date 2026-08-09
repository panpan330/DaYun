from app.rag.generator import RagAnswer, RagAnswerStatus


def test_permission_denied_status_available():
    assert RagAnswerStatus.PERMISSION_DENIED.value == "permission_denied"


def test_rag_answer_can_carry_permission_denied_flag():
    answer = RagAnswer(
        answer="该问题涉及的内容需要相应权限才能查看。",
        status=RagAnswerStatus.PERMISSION_DENIED,
        permission_denied=True,
    )
    assert answer.permission_denied is True
    assert answer.status is RagAnswerStatus.PERMISSION_DENIED


def test_rag_answer_permission_denied_defaults_false():
    answer = RagAnswer(answer="普通回答", status=RagAnswerStatus.ANSWERED)
    assert answer.permission_denied is False
