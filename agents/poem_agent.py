import uuid
from typing import Any, Dict, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Send

from operations import PoemEvaluator, PoemGenerator, QueryTransform
from states import AgentState
from tools import PoemRetrieval

from .verse_analysis import VerseAnalysis
from config import Config


class PoemGeneratorAgent:
    def __init__(self, config: Config, memory=None):
        self.config = config
        self.memory = memory or MemorySaver()
        
        self.poem_retrieval = PoemRetrieval(
            db_path=str(config.db_path),
            diacritizer_weights_path=str(config.model_paths.diacritizer_weights_path)
        )
        
        self.query_transform = QueryTransform(
            llm_provider=config.operations.query_transform.provider,
            llm_name=config.operations.query_transform.name,
            llm_params=config.operations.query_transform.params
        )
        
        self.poem_generator = PoemGenerator(
            llm_provider=config.operations.poem_generator.provider,
            llm_name=config.operations.poem_generator.name,
            llm_params=config.operations.poem_generator.params
        )
        
        self.poem_evaluator = PoemEvaluator(
            llm_provider=config.operations.poem_evaluator.provider,
            llm_name=config.operations.poem_evaluator.name,
            llm_params=config.operations.poem_evaluator.params
        )
        
        self.verse_analysis = VerseAnalysis(
            config=config,
            memory=self.memory,
            as_subgraph=True
        )
        
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

    async def aexecute(self, thread_id: int = None) -> Dict[str, Any]:
        workflow = self._graph.compile()

        config = {"configurable": {"thread_id": thread_id}}
        response = await workflow.ainvoke(self.config.task, config=config)

        return response

    def execute(self, thread_id: int = None) -> Dict[str, Any]:
        workflow = self._graph.compile(debug=False)

        config = {"configurable": {"thread_id": thread_id}}
        response = workflow.invoke(self.config.task, config=config)

        return response

    def get_graph(self):
        return self._create_workflow()