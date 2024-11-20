import json
import re
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, Field

from prompts.prompt_template_registry import PromptTemplateRegistry
from states.agent_state import AgentState

from .base import Operation
from .utils import print_operation_output


SYSTEM_PROMPT = """أنت مساعد متخصص في تحويل طلبات الشعر العربي إلى عبارات موجزة وجيدة بالفصحى تناسب مولد الشعر، دون اختزال الطلب إلى كلمات مفتاحية. مهمتك هي:
١. التحقق من أن الطلب يتعلق بموضوع شعري محدد.
٢. إعادة صياغة الطلب بعبارة موجزة وواضحة بالفصحى تحافظ على كل تفاصيل الطلب قدر الإمكان.

قواعد إعادة الصياغة:
- إعادة صياغة الطلب باختصار مع الحفاظ على كامل معناه الأصلي
- استخدام الفصحى فقط
- تجنب الشرح الإضافي والتفسير
- الحفاظ على جميع جوانب الطلب الشعري الأساسية، مع التلخيص فقط عند الحاجة

أمثلة لإعادة الصياغة:
- "أريد شعراً عن الحنين إلى بلدي كلما سافرت بعيداً، وأشتاق إلى أرضي وناسي" -> "قصيدة عن الشوق إلى الوطن والأحبة في الغربة"
- "اكتب لي قصيدة عن الحب الأول، وكيف يكون بريئاً ومختلفاً عن أي تجربة أخرى." -> "قصيدة عن الحب الأول الصادق والمميز"
- "أريد شعراً عن جمال الطبيعة وسحر الجبال، وأريد أن تعبر عن السلام الذي أشعر به عند رؤيتها" -> "قصيدة عن سحر الطبيعة وهدوء الجبال والسلام الذي تمنحه"

عند تحليل الطلب، عليك إرجاع:
- is_valid: صحيح إذا كان الطلب متعلقاً بموضوع شعري
- transformed_theme: إعادة الصياغة الجديدة بالفصحى (فقط إذا كان الطلب صالحاً)
- error: رسالة الخطأ (فقط إذا كان الطلب غير صالح)

أمثلة للخطأ:
- "كيف حالك؟" -> {is_valid: false, error: "الطلب لا يتعلق بموضوع شعري"}
- "أريد نصائح للطبخ" -> {is_valid: false, error: "الطلب لا يتعلق بالشعر"}
"""


class ThemeValidator(BaseModel):
    """Validator for Arabic poem theme requests."""

    is_valid: bool = Field(description="Whether the query is a valid poem theme request or not")
    transformed_theme: Optional[str] = Field(
        description="The transformed theme in Classical Arabic (if valid)", default=None
    )
    error: Optional[str] = Field(description="Error message if the query is invalid", default=None)


class QueryTransform(Operation[AgentState]):
    llm_name: str = "gpt-4o"
    system_prompt: str = SYSTEM_PROMPT

    def get_messages(self, state: AgentState) -> List[BaseMessage]:
        prompt_template = PromptTemplateRegistry.get("QueryTransform")
        prompt = prompt_template.format(query=state["user_preferences"]["theme"])
        messages = HumanMessage(content=prompt)
        return messages

    def _get_llm(self) -> Any:
        llm = ChatOpenAI(
            model=self.llm_name,
            temperature=0,
            max_tokens=200,
            timeout=None,
            max_retries=2,
        )
        return llm.with_structured_output(ThemeValidator)

    async def ainvoke(self, messages: List[BaseMessage]) -> BaseMessage:
        llm = self._get_llm()
        response = await llm.ainvoke(messages)
        return response

    def invoke(self, messages: List[BaseMessage]) -> BaseMessage:
        llm = self._get_llm()
        response = llm.invoke(messages)
        return response

    def process_response(self, response: ThemeValidator, state: AgentState) -> AgentState:
        print_operation_output(output=response, operation="QueryTransform")

        if not response.is_valid:
            # If the query is invalid, update state with error and return early
            return {"error": {"operation": "QueryTransform", "message": response.error, "status": "failed"}}

        preferences = state.get("user_preferences", {})
        preferences.update({"theme": response.transformed_theme})

        return {"user_preferences": preferences}
