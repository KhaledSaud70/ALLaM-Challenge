from states.agent_state import VerseAnalysisState
from typing import Dict, Any
from .utils import get_verse_prosody


class ProsodyWriter:
    def _convert_pattern(self, pattern: str) -> str:
        if not pattern:
            return ""
        return pattern.replace("1", "v").replace("0", "-")

    def __call__(self, state: VerseAnalysisState) -> Dict[str, Any]:
        prosody_form_verse = {}
        for hemistich_key in ["first_hemistich", "second_hemistich"]:
            hemistich = state["verse"][hemistich_key]
            result = get_verse_prosody(hemistich)
            prosody_form, pattern = result[0], result[1]

            if not prosody_form or not pattern:
                return {"analyzed_verse_metadata": {"prosody_form": ""}}

            converted_pattern = self._convert_pattern(pattern)
            prosody_form_verse[hemistich_key] = (prosody_form, converted_pattern)

        return {"analyzed_verse_metadata": {"prosody_form": prosody_form_verse}}


if __name__ == "__main__":
    model = ProsodyWriter()
    state = {
        "diacritized_verse": {
            "first_hemistich": "أَلَالَيْتُ شِعْرِي هَلْ أَبِيتَنَّ لَيْلَةً",
            "second_hemistich": "الْغَضَى أَزْجِي الْقِلَاصَ النَّوَاجِيَا",
        }
    }
    print(model(state))
