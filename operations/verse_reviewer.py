import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, Field

from prompts.prompt_template_registry import PromptTemplateRegistry
from states import VerseAnalysisState

from .base import Operation
from .utils import print_operation_output


SYSTEM_PROMPT = """أنت مدقق متخصص في مراجعة الأبيات الشعرية العربية ومهمتك الأساسية هي ضمان التزام البيت بالوزن والقافية والمعنى المطلوب.

### قواعد صارمة للمراجعة:
1. القافية (الأولوية الأولى):
   - إلزامية تطابق حرف الروي مع المطلوب بشكل تام
   - مراجعة الحركات قبل وبعد حرف الروي
   - رصد عيوب القافية: الإيطاء، السِّناد، الإقواء
   - رفض أي بيت لا يلتزم بالقافية المطلوبة بشكل قاطع

2. الوزن العروضي:
   - تحليل تفعيلات كل شطر ومطابقتها مع البحر
   - لا يُقبل أي انحراف عن النمط الأساسي
   - مراعاة الزحافات والعلل المسموحة فقط

3. المعنى والسياق:
   - تحليل علاقة البيت بموضوع القصيدة
   - تقييم الترابط مع الأبيات المجاورة
   - تقييم جودة الصياغة والأسلوب

### شكل المخرجات المطلوب:
1. عند قبول البيت:
{{
    "first_hemistich": "الشطر الأول من البيت بعد التصحيح",
    "second_hemistich": "الشطر الثاني من البيت بعد التصحيح"
}}

عند الحاجة للتعديل ولم يتم استنفاد المحاولات:
{{
    "first_hemistich": "الشطر الأول من البيت بعد التصحيح",
    "second_hemistich": "الشطر الثاني من البيت بعد التصحيح",
    "feedback": "التعليمات والمثال المرجعي هنا"
}}
"""


@dataclass
class ReviewResult:
    is_final: bool
    first_hemistich: Optional[str] = None
    second_hemistich: Optional[str] = None
    feedback: Optional[str] = None


class VerseReview(BaseModel):
    """Structure for verse review output"""

    first_hemistich: str = Field(description="الشطر الأول من البيت بعد التصحيح")
    second_hemistich: str = Field(description="الشطر الثاني من البيت بعد التصحيح")
    feedback: Optional[str] = Field(
        description="تعليقات المراجعة التي تتضمن المشاكل والأمثلة المرجعية (فقط عند الحاجة للتعديل)",
        default=None,
    )


class VerseReviewer(Operation[VerseAnalysisState]):
    llm_name: str = "claude-3-5-sonnet-latest"
    system_prompt: str = SYSTEM_PROMPT

    def get_messages(self, state: VerseAnalysisState) -> List[BaseMessage]:
        prompt_template = PromptTemplateRegistry.get("VerseReviewer")
        history = f"""\
## المدقق:
{state.get("reviewer_feedback", "")}


## المصحح:

{state.get("reviser_feedback", "")}
"""
        reference_poem_lines = []
        for verse in state["reference_poem"]:
            formatted_verse = f"{verse.first_hemistich} | {verse.second_hemistich}"
            # formatted_verse = f"{verse['first_hemistich']} | {verse['second_hemistich']}"
            reference_poem_lines.append(formatted_verse)

        reference_poem = "\n".join(reference_poem_lines)

        broken_verse = f"{state['verse']['first_hemistich']} | {state['verse']['second_hemistich']}"
        verse_metadata = state["analyzed_verse_metadata"]
        prompt = prompt_template.format(
            conversation_history=history,
            current_iteration=state["current_recursion"],
            iteration_limit=state["recursion_limit"] + 1,
            broken_verse=broken_verse,
            meter=state["user_preferences"]["meter"],
            rhyme=state["user_preferences"]["rhyme"],
            first_prosody_form=verse_metadata["prosody_form"]["first_hemistich"][0],
            second_prosody_form=verse_metadata["prosody_form"]["second_hemistich"][0],
            reference_poem=reference_poem,
        )

        messages = HumanMessage(content=prompt)
        return messages

    def _get_llm(self) -> Any:
        llm = ChatAnthropic(
            model=self.llm_name,
            temperature=0.3,
            max_tokens=1000,
            top_p=0.9,
            timeout=None,
            max_retries=2,
        )
        return llm.with_structured_output(VerseReview)

    async def ainvoke(self, messages: List[BaseMessage]) -> BaseMessage:
        llm = self._get_llm()
        response = await llm.ainvoke(messages)
        return response

    def invoke(self, messages: List[BaseMessage]) -> BaseMessage:
        llm = self._get_llm()
        response = llm.invoke(messages)
        return response

    def process_response(self, response: VerseReview, state: VerseAnalysisState) -> Dict[str, Any]:
        """Process the structured response from the reviewer"""
        response_data = response.model_dump()

        feedback_text = response_data.pop("feedback", "") or ""  
        formatted_feedback = "\n    ".join(feedback_text.splitlines()) 
        formatted_json = json.dumps(response_data, indent=4, ensure_ascii=False)

        output_text = f"ID: {state.get('verse_id')}\n" f"{formatted_json}\n" f"    feedback:\n    {formatted_feedback}"

        print_operation_output(output=output_text, operation="VerseReviewer")
        print(f"\n Iteration: {state['current_recursion']} / {state['recursion_limit']}\n")

        # Always return the hemistiches as they represent the best version
        verse_feedback = {
            "first_hemistich": response.first_hemistich,
            "second_hemistich": response.second_hemistich,
        }

        # If we've reached the limit or there's no feedback, approve the verse
        if state["current_recursion"] >= state["recursion_limit"] or response.feedback is None:
            return {
                "reviewer_feedback": verse_feedback,
                "last_best_verse": verse_feedback,  # Store the best version
                "is_approved": True,
                "current_recursion": state["current_recursion"],
            }
        else:
            # Need more revisions
            return {
                "reviewer_feedback": response.feedback,
                "last_best_verse": verse_feedback,  # Store the best version
                "is_approved": False,
                "current_recursion": state["current_recursion"] + 1,
            }
