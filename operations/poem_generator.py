
from typing import Any, Dict, List

from langchain_core.messages import BaseMessage, HumanMessage

from prompts.prompt_template_registry import PromptTemplateRegistry
from states.agent_state import AgentState

from .base import Operation
from .utils import print_operation_output

SYSTEM_PROMPT = """نظم الشعر العربي الفصيح بمهارة ووفق القواعد العروضية، مع اتباع المنهجية التالية في كل قصيدة:

١. البناء العروضي:
- التزم بالوزن المطلوب في كل شطر بدقة تامة
- حافظ على وحدة القافية وحرف الروي المحدد
- تجنب الضرورات الشعرية إلا في الحالات القصوى
- استخدم رمز "-" للفصل بين كل شطر وشطر في البيت الواحد

٢. البناء الفني:
- ابدأ بصورة حسية قوية تمهد للموضوع وتشد الانتباه
- طوّر الفكرة عبر الأبيات بترابط فكري وتدرج نحو فكرة أو حكمة تجمع المعنى
- اختتم القصيدة بمعنى جامع يلخص الموضوع ويعزز من عمق النص

٣. الصور والأساليب:
- استخدم صوراً بلاغية وتركيبات ترتبط بالموضوع بشكل عضوي ومتماسك
- اختر الألفاظ من معجم لغوي متناسب مع الجو الشعري العام
- حافظ على توازن بين جزالة الألفاظ وسهولة الفهم ووضوح المعنى
- استخدم دائماً اللغة العربية الفصحى في جميع الألفاظ والتعابير

٤. الوحدة العضوية:
- حافظ على وحدة الموضوع وشعور نفسي متسق خلال الأبيات
- اربط بين الأبيات بتسلسل منطقي بحيث يكمل كل بيت ما قبله
- اجعل كل بيت يضيف جديداً ويثري المعنى العام مع الحفاظ على الترابط

القصائد المرجعية التالية توضح نماذج للوزن والقافية المطلوبين:
{reference_poems}
"""

verse_mapping = {
    1: "بيت واحد",
    2: "بيتين",
    3: "ثلاثة أبيات",
    4: "أربعة أبيات",
    5: "خمسة أبيات",
    6: "ستة أبيات",
    7: "سبعة أبيات",
    8: "ثمانية أبيات",
    9: "تسعة أبيات",
    10: "عشرة أبيات",
    11: "أحد عشر بيتًا",
    12: "اثنا عشر بيتًا",
    13: "ثلاثة عشر بيتًا",
    14: "أربعة عشر بيتًا",
    15: "خمسة عشر بيتًا",
    16: "ستة عشر بيتًا",
    17: "سبعة عشر بيتًا",
    18: "ثمانية عشر بيتًا",
    19: "تسعة عشر بيتًا",
    20: "عشرون بيتًا",
}


class PoemGenerator(Operation[AgentState]):
    system_prompt: str = SYSTEM_PROMPT

    def get_messages(self, state: AgentState) -> List[BaseMessage]:
        data = state["user_preferences"]
        prompt_template = PromptTemplateRegistry.get("PoemGenerator")
        prompt = prompt_template.format(
            meter=data.get("meter", "غير محدد"),
            rhyme=data.get("rhyme", "غير محدد"),
            theme=data.get("theme", "غير محدد"),
            era=data.get("era", "غير محدد"),
            num_verses=data.get("num_verses", 4),
        )

        messages = HumanMessage(content=prompt)
        return messages

    def format_system_prompt(self, state: AgentState) -> str:
        if "reference_poems" not in state:
            raise KeyError("reference_poems not found in state")

        reference_poems_list = state["reference_poems"]
        if not reference_poems_list:
            raise ValueError("reference_poems list is empty")

        reference_poems = "\n".join(poem["llm_prompt"] for poem in reference_poems_list)
        return self.system_prompt.format(reference_poems=reference_poems)

    async def ainvoke(self, messages: List[BaseMessage]) -> BaseMessage:
        llm = self._get_llm()
        response = await llm.ainvoke(messages)
        return response

    def invoke(self, messages: List[BaseMessage]) -> BaseMessage:
        llm = self._get_llm()
        response = llm.invoke(messages)
        return response

    def process_response(self, response: str, state: AgentState) -> Dict[str, Any]:
        print_operation_output(output=response.content, operation="PoemGenerator")
        return {"poems": [response.content]}