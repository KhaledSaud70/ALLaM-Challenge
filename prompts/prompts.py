from .prompt_template_registry import PromptTemplateRegistry
from langchain_core.prompts import PromptTemplate


@PromptTemplateRegistry.register("QueryTransform")
class QueryTransformPrompt(PromptTemplate):
    template: str = "{query}"
    input_variables: list = ["query"]


@PromptTemplateRegistry.register("PoemGenerator")
class PoemGeneratorPrompt(PromptTemplate):
    template: str = """Input:
العصر: {era}
البحر: {meter}
القافية: حرف الروي: {rhyme}
الموضوع: {theme}
عدد الأبيات: {num_verses}
Output:
"""
    input_variables: list = ["meter", "rhyme", "num_verses", "theme", "era"]


@PromptTemplateRegistry.register("PoemEvaluator")
class PoemEvaluatorPrompt(PromptTemplate):
    template: str = """القصائد المقدمة للتقييم:

{poems}

متطلبات القصيدة:
العصر: {era}
البحر: {meter}
القافية: {rhyme}
الموضوع: {theme}
عدد الأبيات: {num_verses}
"""
    input_variables: list = ["poems", "era", "meter", "rhyme", "theme", "num_verses"]


@PromptTemplateRegistry.register("PoemAssistant")
class PoemAssistantPrompt(PromptTemplate):
    template: str = """Input: اكتب قصيدة {theme} تتألف من {num_verses}، على وزن بحر {meter} وبقافية بحرف الروى {rhyme}.
Output:
"""
    input_variables: list = ["meter", "rhyme", "num_verses", "theme"]


@PromptTemplateRegistry.register("VerseReviewer")
class VerseReviewerPrompt(PromptTemplate):
    template: str = """سجل المحادثة:
---
{conversation_history}
---

### المعلومات المقدمة:
- البيت المطلوب تحليله: {broken_verse}
- البحر المطلوب: {meter}
- القافية المطلوبة (حرف الروي): "{rhyme}"
- القصيدة المرجعية للسياق: {reference_poem}

**ملاحظة**: استخدم القصيدة المرجعية كإطار مرجعي لفهم الموضوع والموقع العاطفي للبيت، ولكن لا تقم بتعديل القصيدة الأصلية—يجب أن يقتصر التعديل على البيت المطلوب فقط.

### معلومات التحليل العروضي للبيت:
- الكتابة العروضية: {first_prosody_form} || {second_prosody_form}

# المحاولة رقم: {current_iteration} من أصل {iteration_limit}

### خطوات المراجعة الإلزامية:

1. تحليل القافية (الأولوية القصوى):
   - تحديد حرف الروي في نهاية البيت
   - مقارنته مع حرف الروي المطلوب
   - فحص الحركات المصاحبة للقافية
   - التأكد من خلو القافية من العيوب

2. تحليل البحر:
   - مراجعة الكتابة العروضية المقدمة
   - تحديد مواضع الخلل في التفعيلات

3. تحليل المعنى والسياق:
   - قراءة الأبيات المجاورة في القصيدة المرجعية
   - تقييم مدى انسجام البيت مع السياق
   - تقييم قوة الصياغة وجمال التعبير

### آلية تقديم التعليقات:

عند اكتشاف أي خلل، يجب تقديم:
1. وصف دقيق للمشكلة
2. تقديم مثال محسّن كمرجع بحيث يعالج كافة الجوانب المذكورة ويضمن توافق البيت مع البحر المطلوب والقافية ومعنى البيت في القصيدة الأصلية.
3. اقتراحات محددة للتعديل
4. شرح سبب اختيار التعديلات المقترحة

### المخرجات المطلوبة:

إذا كان البيت سليماً:
{{
    "first_hemistich": "الشطر الأول من البيت بعد التصحيح",
    "second_hemistich": "الشطر الثاني من البيت بعد التصحيح"
}}

إذا كان البيت يحتاج لتعديل:
{{
    "first_hemistich": "الشطر الأول المقترح",
    "second_hemistich": "الشطر الثاني المقترح",
    "feedback": \"""
        المشاكل المكتشفة:
        1. [القافية/الوزن/المعنى]: وصف تفصيلي
        
        مثال مرجعي صحيح:
        الشطر الأول: ...
        الشطر الثاني: ...
    \"""
}}
"""
    input_variables: list = [
        "broken_verse",
        "meter",
        "rhyme",
        "reference_poem",
        "conversation_history",
        "current_iteration",
        "iteration_limit",
    ]


@PromptTemplateRegistry.register("VerseReviser")
class VerseReviserPrompt(PromptTemplate):
    template: str = """\
بيت الشعر الأصلي:
{broken_verse}

البحر: {meter}
حرف الروي: {rhyme}

ملاحظات المدقق:
{reviewer_feedback}

---

المطلوب: قم بتصحيح البيت وفقاً لملاحظات المدقق.
"""
    input_variables: list = ["broken_verse", "meter", "rhyme", "reviewer_feedback"]
