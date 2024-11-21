import json
from typing import Any, Dict, List
import re

from langchain_core.messages import BaseMessage, HumanMessage

from prompts.prompt_template_registry import PromptTemplateRegistry
from states.agent_state import VerseAnalysisState

from .base import Operation
from .utils import print_operation_output


SYSTEM_PROMPT = """قم بتصحيح الأبيات الشعرية العربية وتعديلها بدقة، بناءً على ملاحظات المدقق، ووفقاً للمعايير التالية دون أي تعليقات أو مقدمات:

١. تنفيذ التعديلات المقترحة من المدقق بدقة
٢. الحفاظ على المعنى الأساسي للبيت قدر الإمكان
٣. الالتزام بالوزن العروضي المطلوب
٤. مراعاة القافية المحددة
٥. استخدام الكلمات البديلة المقترحة إن وجدت
"""


class VerseReviser(Operation[VerseAnalysisState]):
    system_prompt: str = SYSTEM_PROMPT

    def get_messages(self, state: VerseAnalysisState) -> List[BaseMessage]:
        prompt_template = PromptTemplateRegistry.get("VerseReviser")
        broken_verse = f"""\
الشطر الاول: {state['verse']['first_hemistich']}
الشطر الثاني: {state['verse']['second_hemistich']}
"""
        prompt = prompt_template.format(
            broken_verse=broken_verse,
            meter=state["user_preferences"]["meter"],
            rhyme=state["user_preferences"]["rhyme"],
            reviewer_feedback=state["reviewer_feedback"],
        )

        messages = HumanMessage(content=prompt)
        return messages

    async def ainvoke(self, messages: List[BaseMessage]) -> BaseMessage:
        llm = self._get_llm()
        response = await llm.ainvoke(messages)
        return response

    def invoke(self, messages: List[BaseMessage]) -> BaseMessage:
        llm = self._get_llm()
        response = llm.invoke(messages)
        return response

    def process_response(self, response: str, state: VerseAnalysisState) -> Dict[str, Any]:
        print_operation_output(output=f"ID: {state.get('verse_id')}\n" + response.content, operation="VerseReviser")
        return {"reviser_feedback": response.content}


if __name__ == "__main__":
    agent = VerseReviser()
    input = {}
    agent.execute(input)
