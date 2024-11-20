from typing import Annotated, Dict, TypedDict, List, Any, Optional
import operator
# from langgraph.graph.message import AnyMessage, add_messages


class AgentState(TypedDict):
    user_preferences: Annotated[Dict[str, Any], operator.or_]
    selected_poem: List[Dict[str, str]]
    poems: Annotated[List[str], operator.add]
    reference_poems: List[Dict[str, Any]]
    num_reference_poems: int
    num_poems_to_evaluate: int
    analyzed_verses_metadata: Annotated[List[dict], operator.add]
    error: Optional[Dict[str, str]]
    final_poem: List[Dict[str, str]]

    # messages: Annotated[list[AnyMessage], add_messages]


class VerseAnalysisState(TypedDict):
    user_preferences: Annotated[Dict[str, Any], operator.or_]
    reference_poem: List[Dict[str, str]]
    verse: Dict[str, str]
    verse_id: int
    verse_prosody_form: Dict[str, tuple]
    reviewer_feedback: str
    reviser_feedback: str
    last_best_verse: str
    analyzed_verse_metadata: Annotated[Dict[str, Any], operator.or_]
    recursion_limit: int = 2
    current_recursion: int = 0
    is_approved: bool
