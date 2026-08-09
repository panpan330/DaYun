import pytest

from app.services.prompt_builder import PromptParts, build_clear_user_prompt


def test_build_clear_user_prompt_with_task_only() -> None:
    prompt = build_clear_user_prompt(PromptParts(task="解释 FastAPI 是什么"))

    assert prompt == "## 任务\n解释 FastAPI 是什么"


def test_build_clear_user_prompt_with_all_sections() -> None:
    prompt = build_clear_user_prompt(
        PromptParts(
            task="解释 token 是什么",
            context="用户刚学完 LLM API 基础。",
            constraints=[
                "用中文回答",
                "适合零基础学习者",
                "不要写真实价格",
            ],
            output_format="先给一句话解释，再给 3 个要点。",
            failure_policy="如果不确定，请明确说不确定，并说明需要查官方文档。",
        )
    )

    assert prompt == "\n\n".join(
        [
            "## 任务\n解释 token 是什么",
            "## 上下文\n用户刚学完 LLM API 基础。",
            "## 要求\n- 用中文回答\n- 适合零基础学习者\n- 不要写真实价格",
            "## 输出格式\n先给一句话解释，再给 3 个要点。",
            "## 无法完成时\n如果不确定，请明确说不确定，并说明需要查官方文档。",
        ]
    )


def test_build_clear_user_prompt_strips_optional_text() -> None:
    prompt = build_clear_user_prompt(
        PromptParts(
            task="  总结上下文  ",
            context="  这里是上下文。  ",
            output_format="  使用列表。  ",
            failure_policy="  不知道就说不知道。  ",
        )
    )

    assert "## 任务\n总结上下文" in prompt
    assert "## 上下文\n这里是上下文。" in prompt
    assert "## 输出格式\n使用列表。" in prompt
    assert "## 无法完成时\n不知道就说不知道。" in prompt


def test_build_clear_user_prompt_ignores_blank_constraints() -> None:
    prompt = build_clear_user_prompt(
        PromptParts(
            task="解释列表",
            constraints=["  用中文回答  ", "", "   "],
        )
    )

    assert prompt == "## 任务\n解释列表\n\n## 要求\n- 用中文回答"


def test_build_clear_user_prompt_rejects_blank_task() -> None:
    with pytest.raises(ValueError, match="task"):
        build_clear_user_prompt(PromptParts(task="   "))
