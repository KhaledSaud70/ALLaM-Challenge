from states.agent_state import VerseAnalysisState
from models import LLMRegistry
from typing import Dict, Any
from .utils import merge_diacritization, print_tool_output


class Diacritizer:
    def __init__(self, weights_path: str = None):
        self.model = LLMRegistry().get(model_name="diacritizer_model")
        self.model.load_model(weights_path=weights_path)

    def __call__(self, state: VerseAnalysisState) -> Dict[str, Any]:
        diacritized_verse = {}

        verse = state["verse"] if isinstance(state, dict) else state.verse

        for hemistich_key in ["first_hemistich", "second_hemistich"]:
            hemistich = verse[hemistich_key] if isinstance(verse, dict) else getattr(verse, hemistich_key)
            processed_hemistich = self.model.text_encoder.clean(hemistich).strip()

            # Get diacritization
            diacritized_hemistich = self.model.infer(processed_hemistich)
            diacritized_hemistich = self.model.text_encoder.clean(diacritized_hemistich).strip()

            # Merge diacritization with original text
            merged_diacritization = merge_diacritization(diacritized_hemistich, processed_hemistich)

            diacritized_verse[hemistich_key] = merged_diacritization

        # print_tool_output(output=diacritized_verse, tool="Diacritizer")

        return {"verse": diacritized_verse}


if __name__ == "__main__":
    model = Diacritizer(model_path="/home/khaled/workspace/projects/allam/models/weights/diacritizer_model_weights.pt")
    state = {
        "verse": {
            "first_hemistich": "ألا ليت شعري هل أبيتن ليلة",
            "second_hemistich": "بجنب الغضى أزجي القلاص النواجيا",
        }
    }
    print(model(state))
