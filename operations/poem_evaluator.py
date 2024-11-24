import json
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, Field

from prompts.prompt_template_registry import PromptTemplateRegistry
from states.agent_state import AgentState

from .base import Operation
from .utils import print_operation_output


SYSTEM_PROMPT = """أنت خبير في تقييم الشعر العربي الفصيح واختيار أفضل القصائد. مهمتك تحليل مجموعة من القصائد المولدة واختيار أفضلها بناءً على المعايير التالية:

١. الدقة العروضية (٣٠٪):
- مطابقة البحر الشعري المطلوب: {meter}
- التزام القافية وحرف الروي: {rhyme}
- سلامة التفعيلات والأوزان

٢. جودة المحتوى (٣٠٪):
- مناسبة المحتوى للموضوع المطلوب: {theme}
- تناسق الأفكار وترابطها
- عمق المعاني وأصالتها
- مناسبة الأسلوب للعصر المطلوب: {era}

٣. البناء الفني (٢٠٪):
- قوة المطلع وجمال الختام
- التسلسل المنطقي للأفكار
- جودة الصور البلاغية
- فصاحة اللغة وجزالة الألفاظ

٤. الوحدة العضوية (٢٠٪):
- ترابط الأبيات
- تكامل المعنى العام
- التدرج المنطقي في عرض الأفكار

# تعليمات التقييم:
١. قيّم كل قصيدة من القصائد المقدمة وفق المعايير السابقة
٢. اختر القصيدة التي تحقق أعلى درجات التقييم الإجمالي
"""


class VerseFormat(BaseModel):
    first_hemistich: str = Field(description="الشطر الأول من البيت الشعري، يجب أن يكون نصاً عربياً فصيحاً")
    second_hemistich: str = Field(description="الشطر الأول من البيت الشعري، يجب أن يكون نصاً عربياً فصيحاً")


class PoemResult(BaseModel):
    best_poem: List[VerseFormat] = Field(
        description="القصيدة الأفضل بعد التقييم، تتكون من قائمة من الأبيات، مع تقسيم كل بيت إلى شطرين."
    )


class PoemEvaluator(Operation[AgentState]):
    system_prompt: str = SYSTEM_PROMPT

    def get_messages(self, state: AgentState) -> List[BaseMessage]:
        prompt_template = PromptTemplateRegistry.get("PoemEvaluator")
        poems_list = state["poems"]  # List[str]
        arabic_ordinals = {
            1: "الأولى",
            2: "الثانية",
            3: "الثالثة",
            4: "الرابعة",
            5: "الخامسة",
            6: "السادسة",
            7: "السابعة",
            8: "الثامنة",
            9: "التاسعة",
            10: "العاشرة",
            # Extend as needed for more poems
        }

        # Format each poem in the required structure
        formatted_poems = []
        for i, poem_content in enumerate(poems_list, start=1):
            poem_title = f"القصيدة {arabic_ordinals.get(i, f'رقم {i}')}:"
            # poem_content = "\n".join(
            #     f"{verse['first_hemistich']} | {verse['second_hemistich']}" for verse in poem
            # )
            formatted_poems.append(f"{poem_title}\n{poem_content}")

        poems_text = "\n\n".join(formatted_poems)

        data = state["user_preferences"]
        prompt = prompt_template.format(
            poems=poems_text,
            meter=data.get("meter", "غير محدد"),
            rhyme=data.get("rhyme", "غير محدد"),
            theme=data.get("theme", "غير محدد"),
            era=data.get("era", "غير محدد"),
            num_verses=data.get("num_verses", "غير محدد"),
        )

        messages = HumanMessage(content=prompt)
        return messages

    def format_system_prompt(self, state: AgentState) -> str:
        prompt = self.system_prompt.format(
            meter=state["user_preferences"]["meter"],
            rhyme=state["user_preferences"]["rhyme"],
            theme=state["user_preferences"]["theme"],
            era=state["user_preferences"]["era"],
        )

        return prompt

    async def ainvoke(self, messages: List[BaseMessage]) -> BaseMessage:
        llm = self._get_llm()
        llm = llm.with_structured_output(PoemResult)
        response = await llm.ainvoke(messages)
        return response

    def invoke(self, messages: List[BaseMessage]) -> BaseMessage:
        llm = self._get_llm()
        if self.llm_provider != "custom":
            llm = llm.with_structured_output(PoemResult)
        response = llm.invoke(messages)
        return response

    def process_response(self, response: str, state: AgentState) -> Dict[str, Any]:
        print_operation_output(output=response.best_poem, operation="PoemEvaluator")
        return {"selected_poem": response.best_poem}