from states.agent_state import VerseAnalysisState
from models import LLMRegistry
from typing import Dict, Any
from .utils import print_tool_output


class MeterClassifier:
    def __init__(self, weights_path: str = None):
        self.model = LLMRegistry().get(llm_provider="custom", model_name="meter_classifier")
        self.model.load_weights(weights_path=weights_path)

    def __call__(self, state: VerseAnalysisState) -> Dict[str, Any]:
        verse = state["verse"]
        verse = f"{verse['first_hemistich']} S {verse['second_hemistich'] or 'E'}"
        prediction = self.model.predict(verse)
        meter, confidence = prediction["meter"], prediction["confidence"]
        print_tool_output(output=meter + f", ID: {state.get('verse_id')}", tool="MeterClassifier")
        return {"analyzed_verse_metadata": {"meter": meter, "meter_confidence": confidence}}


if __name__ == "__main__":
    model = MeterClassifier(
        weights_path="/home/khaled/workspace/projects/allam/models/weights/classic_meters_classifierTF_10L12H.h5"
    )
    state = {"diacritized_verse": "ألا ليت شعري هل أبيتن ليلة S بجنب الغضى أزجي القلاص النواجيا"}
    print(model(state))
