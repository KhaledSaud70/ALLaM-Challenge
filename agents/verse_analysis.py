import uuid
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import Enum
from typing import Any, Dict, List, Literal, NamedTuple, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from operations import VerseReviewer, VerseReviser
from states import VerseAnalysisState
from tools import Diacritizer, MeterClassifier, MeterRegistry, ProsodyWriter, RhymeClassifier

from .utils import print_output
from config import Config


@dataclass
class MeterInfo:
    name_en: str
    name_ar: str


class MeterNames(Enum):
    SAREE = MeterInfo("saree", "السريع")
    KAMEL = MeterInfo("kamel", "الكامل")
    MUTAKAREB = MeterInfo("mutakareb", "المتقارب")
    MUTADARAK = MeterInfo("mutadarak", "المتدارك")
    MUNSAREH = MeterInfo("munsareh", "المنسرح")
    MADEED = MeterInfo("madeed", "المديد")
    MUJTATH = MeterInfo("mujtath", "المجتث")
    RAMAL = MeterInfo("ramal", "الرمل")
    BASEET = MeterInfo("baseet", "البسيط")
    KHAFEEF = MeterInfo("khafeef", "الخفيف")
    TAWEEL = MeterInfo("taweel", "الطويل")
    WAFER = MeterInfo("wafer", "الوافر")
    HAZAJ = MeterInfo("hazaj", "الهزج")
    RAJAZ = MeterInfo("rajaz", "الرجز")
    MUDHARE = MeterInfo("mudhare", "المضارع")
    MUQTADHEB = MeterInfo("muqtadheb", "المقتضب")
    NATHR = MeterInfo("nathr", "نثر")

    @classmethod
    def get_english_name(cls, arabic_name: str) -> str:
        for meter in cls:
            if meter.value.name_ar == arabic_name:
                return meter.value.name_en
        return ""

    @classmethod
    def get_arabic_name(cls, english_name: str) -> str:
        for meter in cls:
            if meter.value.name_en == english_name:
                return meter.value.name_ar
        return ""


@dataclass
class SimilarityResult:
    pattern: str
    probability: float
    foot: str

    def __lt__(self, other):
        return self.probability < other.probability


@dataclass
class ValidationCriteria:
    required_meter: str
    required_rhyme: str


class VerseMetadata(NamedTuple):
    meter: str
    rhyme: str


@dataclass
class HemistichProsody:
    text: str
    pattern: str


@dataclass
class VerseProsodyForm:
    first_hemistich: HemistichProsody
    second_hemistich: HemistichProsody

    @classmethod
    def from_dict(cls, data: Dict) -> "VerseProsodyForm":
        return cls(
            first_hemistich=HemistichProsody(*data["first_hemistich"]),
            second_hemistich=HemistichProsody(*data["second_hemistich"]),
        )


class VerseAnalysis:
    def __init__(self, config: Config, memory: Optional[MemorySaver] = None, as_subgraph = False):
        self.config = config
        self.memory = memory
        self.as_subgraph = as_subgraph
        
        # Initialize components with configuration
        self.meter_classifier = MeterClassifier(
            weights_path=str(config.model_paths.meter_weights_path)
        )
        
        self.rhyme_classifier = RhymeClassifier()
        self.prosody_writer = ProsodyWriter()
        
        self.diacritizer = Diacritizer(
            weights_path=str(config.model_paths.diacritizer_weights_path)
        )
        
        self.verse_reviewer = VerseReviewer(
            provider=config.operations.verse_reviewer.provider,
            llm_name=config.operations.verse_reviewer.name,
            llm_params=config.operations.verse_reviewer.params
        )
        
        self.verse_reviser = VerseReviser(
            provider=config.operations.verse_reviser.provider,
            llm_name=config.operations.verse_reviser.name,
            llm_params=config.operations.verse_reviser.params
        )
        
        self._graph = self._create_workflow()

    def _create_workflow(self) -> StateGraph:
        workflow = StateGraph(VerseAnalysisState)

        workflow.add_node("diacritizer", self.diacritizer)
        workflow.add_node("meter_classifier", self.meter_classifier)
        workflow.add_node("rhyme_classifier", self.rhyme_classifier)
        workflow.add_node("prosody_writer", self.prosody_writer)
        workflow.add_node("validate_verse", self.validate_verse)
        workflow.add_node("reviewer", self.verse_reviewer.execute)
        workflow.add_node("reviser", self.verse_reviser.execute)
        workflow.add_node("writer", self._writer)

        workflow.set_entry_point("diacritizer")

        workflow.add_edge("diacritizer", "meter_classifier")
        workflow.add_edge("diacritizer", "rhyme_classifier")
        workflow.add_edge("diacritizer", "prosody_writer")
        workflow.add_edge("meter_classifier", "validate_verse")
        workflow.add_edge("rhyme_classifier", "validate_verse")
        workflow.add_edge("prosody_writer", "validate_verse")

        workflow.add_conditional_edges(
            "validate_verse", self._validation_condition, {"approve": END, "review": "reviewer"}
        )

        workflow.add_conditional_edges("reviewer", self._review_condition, {"revise": "reviser", "approve": "writer"})
        workflow.add_edge("reviser", "reviewer")
        workflow.add_edge("writer", END)

        return workflow

    @staticmethod
    def _validation_condition(state: VerseAnalysisState) -> Literal["approve", "review"]:
        if state["analyzed_verse_metadata"]["is_approved"]:
            return "approve"
        return "review"

    @staticmethod
    def _review_condition(state: VerseAnalysisState) -> Literal["approve", "revise"]:
        if state["is_approved"] or state["current_recursion"] > state["recursion_limit"]:
            return "approve"
        return "revise"

    def _writer(self, state: VerseAnalysisState):
        metadata = state.get("analyzed_verse_metadata", {})
        metadata.update(
            {
                "validated_verse": state["last_best_verse"],
                "is_approved": state.get("is_approved", False),  # Default if `is_approved` not present
            }
        )

        return {"analyzed_verse_metadata": metadata}

    def calculate_similarity(self, pred_pattern: str, meter_name: str) -> SimilarityResult:
        """
        Calculate similarity for a single hemistich pattern.

        Args:
            pred_pattern: The input pattern to check
            meter_name: Name of the meter (in Arabic)

        Returns:
            The best match SimilarityResult object
        """

        meters_patterns = {
            name: meter().all_shatr_combinations_patterns for name, meter in MeterRegistry.get_meters().items()
        }
        meters_foots = {
            name: meter().get_all_shatr_combinations(as_str_list=True)
            for name, meter in MeterRegistry.get_meters().items()
        }

        meter_en = MeterNames.get_english_name(meter_name)
        if not meter_en or meter_en not in meters_patterns:
            raise ValueError(f"Unknown meter: {meter_name}")

        best_match = SimilarityResult("", 0.0, "")
        for target_pattern, target_foot in zip(meters_patterns[meter_en], meters_foots[meter_en]):
            target_pattern = target_pattern.replace("1", "v").replace("0", "-")
            probability = self._similarity_score(pred_pattern, target_pattern)
            current_result = SimilarityResult(target_pattern, probability, target_foot)
            if current_result.probability > best_match.probability:
                best_match = current_result

        return best_match

    @staticmethod
    def _similarity_score(pattern1: str, pattern2: str) -> float:
        """
        Calculate similarity score between two patterns using SequenceMatcher.

        Returns:
            float: Similarity ratio between 0 and 1
        """
        return SequenceMatcher(None, pattern1, pattern2).ratio()

    def check_hemistich(self, pred_pattern: str, meter_name: str) -> SimilarityResult:
        """
        Public method to check verse similarity.

        Args:
            pred_pattern: The input pattern to check
            meter_name: Name of the meter in Arabic

        Returns:
            The best match SimilarityResult object
        """
        return self.calculate_similarity(pred_pattern, meter_name)

    def validate_verse(self, state: VerseAnalysisState) -> Dict[str, Any]:
        """
        Validate verse against user preferences and prosody patterns.
        """

        # if not state["verse_prosody_form"]:
        #     return {"validated_verse_metadata": {"validated_verse": state["verse"], "is_approved": False}}

        metadata = VerseMetadata(
            meter=state["analyzed_verse_metadata"]["meter"],
            rhyme=state["analyzed_verse_metadata"]["rhyme"]["rawwy"],
        )

        criteria = ValidationCriteria(
            required_meter=state["user_preferences"]["meter"],
            required_rhyme=state["user_preferences"]["rhyme"],
        )

        # Parse prosody form
        verse_prosody = VerseProsodyForm.from_dict(state["analyzed_verse_metadata"]["prosody_form"])

        # Check both hemistichs
        first_match = self.check_hemistich(
            pred_pattern=verse_prosody.first_hemistich.pattern,
            meter_name=criteria.required_meter,
        )

        second_match = self.check_hemistich(
            pred_pattern=verse_prosody.second_hemistich.pattern,
            meter_name=criteria.required_meter,
        )

        # Verse is valid only if both hemistichs match and other criteria are met
        is_valid = (
            metadata.meter == criteria.required_meter
            and metadata.rhyme == criteria.required_rhyme
            and first_match.probability == 1.0
            and second_match.probability == 1.0
        )

        print_output(output={"is_valid": is_valid, "ID": state["verse_id"]}, operation="VerseValidation")

        return {
            "analyzed_verse_metadata": {
                "validated_verse": state["verse"],
                "first_suggested_pattern": first_match.pattern,
                "first_pattern_confidence": first_match.probability,
                "first_pattern_foot": first_match.foot,
                "second_suggested_pattern": second_match.pattern,
                "second_pattern_confidence": second_match.probability,
                "second_pattern_foot": second_match.foot,
                "is_approved": is_valid,
            }
        }

    def execute(self, inputs: Dict[str, str], thread_id: int = None) -> Dict[str, Any]:
        workflow = self._graph.compile(debug=False)
        config = {"configurable": {"thread_id": thread_id}}
        response = workflow.invoke(inputs, config=config)

        if self.as_subgraph:
            # Return only the keys that should be propagated to the main graph
            return {
                "analyzed_verses_metadata": [response["analyzed_verse_metadata"] | {"id": inputs["verse_id"]}],
            }

        return response

    async def aexecute(self, inputs: Dict[str, str], thread_id: int = None) -> Dict[str, Any]:
        workflow = self._graph.compile()
        config = {"configurable": {"thread_id": thread_id}}
        response = await workflow.ainvoke(inputs, config=config)

        if self.as_subgraph:
            # Return only the keys that should be propagated to the main graph
            return {
                "analyzed_verses_metadata": [response["analyzed_verse_metadata"] | {"id": inputs["verse_id"]}],
            }

        return response

    def get_graph(self) -> StateGraph:
        return self._create_workflow()


if __name__ == "__main__":
    inputs1 = {
        "user_preferences": {"rhyme": "د", "meter": "الطويل"},
        "verse": {"first_hemistich": "إذا عزّ نفس اللئيم بذلها", "second_hemistich": "وَنفس الكريم الحرّ بالمجدِ تُعلَمُ"},
        "current_recursion": 0,
        "recursion_limit": 2,
        "reviewer_feedback": "",
        "reviser_feedback": "",
    }

    models_configs = {
        "meter_weights_path": "/home/khaled/workspace/projects/allam/models/weights/classic_meters_classifierTF_10L12H.h5",
        "diacritizer_weights_path": "/home/khaled/workspace/projects/allam/models/weights/diacritizer_model_weights.pt",
    }

    agent = VerseAnalysis(models_configs=models_configs)
    response = agent.execute(inputs1, thread_id=1)

    print("FINAL STATE:\n")
    for k, v in response.items():
        print(f"{k}: {v}")
