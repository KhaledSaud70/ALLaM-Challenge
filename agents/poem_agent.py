import uuid
from typing import Any, Dict, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Send

from operations import PoemEvaluator, PoemGenerator, QueryTransform
from states import AgentState
from tools import PoemRetrieval

from .verse_analysis import VerseAnalysis


class PoemGeneratorAgent:
    def __init__(self, models_configs: dict, db_path: str, memory=MemorySaver()):
        self.models_configs = models_configs
        self.memory = memory
        self.poem_retrieval = PoemRetrieval(db_path=db_path if db_path else "data/arabic_poems.db")
        self.query_transform = QueryTransform()
        self.poem_generator = PoemGenerator()
        self.poem_evaluator = PoemEvaluator()
        self.verse_analysis = VerseAnalysis(models_configs=models_configs, as_subgraph=True)
        self._graph = self._create_workflow()

    def _create_workflow(self) -> StateGraph:
        workflow = StateGraph(AgentState)

        workflow.add_node("poem_retrieval", self.poem_retrieval.search_poems)
        workflow.add_node("query_transform", self.query_transform.execute)
        workflow.add_node("poem_generator", self.poem_generator.execute)
        workflow.add_node("poem_evaluator", self.poem_evaluator.execute)
        workflow.add_node("verse_analysis", self.verse_analysis.execute)
        workflow.add_node("writer", self._writer)

        workflow.set_entry_point("query_transform")
        workflow.add_conditional_edges(
            "query_transform", self._query_validation, {"accept": "poem_retrieval", "reject": END}
        )
        workflow.add_conditional_edges("poem_retrieval", self._generate_poems, ["poem_generator"])
        workflow.add_edge("poem_generator", "poem_evaluator")
        workflow.add_conditional_edges("poem_evaluator", self._continue_to_analysis, ["verse_analysis"])
        workflow.add_edge("verse_analysis", "writer")
        workflow.add_edge("writer", END)

        return workflow

    @staticmethod
    def _writer(state: AgentState) -> dict:
        final_poem = []
        for verse_metadata in state["analyzed_verses_metadata"]:
            first_hemistich = verse_metadata["validated_verse"]["first_hemistich"]
            second_hemistich = verse_metadata["validated_verse"]["second_hemistich"]
            final_poem.append({"first_hemistich": first_hemistich, "second_hemistich": second_hemistich})
        return {"final_poem": final_poem}

    @staticmethod
    def _query_validation(state: AgentState) -> Literal["accept", "reject"]:
        if not state["error"]:
            return "accept"
        return "reject"

    @staticmethod
    def _generate_poems(state: AgentState):
        return [
            Send(
                "poem_generator",
                {"user_preferences": state["user_preferences"], "reference_poems": state["reference_poems"]},
            )
            for _ in range(state["num_poems_to_evaluate"])
        ]

    @staticmethod
    def _continue_to_analysis(state: AgentState):
        return [
            Send(
                "verse_analysis",
                {
                    "user_preferences": state["user_preferences"],
                    "reference_poem": state["selected_poem"],
                    "verse": v,
                    "verse_id": id,
                    "reviewer_feedback": "",
                    "reviser_feedback": "",
                    "analyzed_verse_metadata": {},
                    "recursion_limit": 2,
                    "current_recursion": 0,
                },
            )
            for id, v in enumerate(state["selected_poem"])
        ]

    async def aexecute(self, inputs: Dict[str, str], thread_id: int = None) -> Dict[str, Any]:
        workflow = self._graph.compile()

        config = {"configurable": {"thread_id": thread_id}}  # "thread_ts": datetime.now(timezone.utc)
        response = await workflow.ainvoke(inputs, config=config)

        return response

    def execute(self, inputs: Dict[str, str], thread_id: int = None) -> Dict[str, Any]:
        workflow = self._graph.compile(debug=False)

        config = {"configurable": {"thread_id": thread_id}}  # "thread_ts": datetime.now(timezone.utc)
        response = workflow.invoke(inputs, config=config)

        return response

    def get_graph(self):
        return self._create_workflow()


if __name__ == "__main__":
    inputs1 = {
        "user_preferences": {
            "poet_name": "",
            "meter": "الخفيف",
            "rhyme": "د",
            "era": "العصر المملوكي",
            "theme": "",
            "num_verses": 4,
        },
        "num_reference_poems": 3,
        "num_poems_to_evaluate": 3,
        "reference_poems": "",
        "error": {},
    }

    models_configs = {
        "meter_weights_path": "models/weights/classic_meters_classifierTF_10L12H.h5",
        "diacritizer_weights_path": "models/weights/diacritizer_model_weights.pt",
    }

    agent = PoemGeneratorAgent(models_configs=models_configs)
    response = agent.execute(inputs1, thread_id=str(uuid.uuid4()))

    print("FINAL STATE:\n")
    for k, v in response.items():
        print(f"{k}: {v}")
