from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

import pyarabic.araby as araby
from .utils import print_tool_output

from states.agent_state import VerseAnalysisState


@dataclass
class RhymeConfig:
    """Configuration for rhyme classification."""

    accepted_chars: str = "إةابتثجحخدذرزسشصضطظعغفقكلمنهويىأءئؤ#آ َِْ "
    fatha: str = "َ"
    sukun: str = "ْ"
    special_endings: Dict[str, str] = None

    def __post_init__(self):
        self.accepted_chars = list(self.accepted_chars)
        # Define special character mappings
        self.special_endings = {
            "ؤ": "ء",
            "ئ": "ء",
            "أ": "ء",
            "ء": "ء",  # Hamza forms
            "ه": "ه",
            "ة": "ه",  # Ha forms
        }


class TextNormalizer:
    """Handles Arabic text normalization operations."""

    def __init__(self, config: RhymeConfig):
        self.config = config

    def remove_tashkeel(self, text: str) -> str:
        """Remove diacritical marks."""
        return araby.strip_tashkeel(text)

    def remove_tatweel(self, text: str) -> str:
        """Remove tatweel character."""
        return araby.strip_tatweel(text)

    def remove_unaccepted_chars(self, text: str) -> str:
        """Remove characters not in accepted list."""
        return "".join(c for c in text if c in self.config.accepted_chars)

    def normalize(self, text: str) -> str:
        """Apply all normalization steps."""
        return self.remove_tashkeel(self.remove_tatweel(self.remove_unaccepted_chars(text))).strip()


class RhymeAnalyzer:
    """Analyzes rhyme patterns in Arabic poetry."""

    def __init__(self, config: RhymeConfig = None):
        self.config = config or RhymeConfig()
        self.normalizer = TextNormalizer(self.config)

    def _get_rawwy_char(self, bait: str, with_index: bool = False) -> Union[str, Tuple[str, int]]:
        """
        Extract the rhyme character (rawwy) from the verse.

        Args:
            bait: The verse text
            with_index: Whether to return the position of the rawwy

        Returns:
            Either the rawwy character or a tuple of (rawwy, index)
        """
        clean_bait = self.normalizer.normalize(bait)
        i = -1

        while True:
            last_char = clean_bait[i]
            second_last_char = clean_bait[i - 1]
            if second_last_char == "آ":
                second_last_char = "ا"
            last_two_chars = "".join(clean_bait[i - 1 :])
            i -= 1

            # Handle special cases
            if last_two_chars in ("يا", "يه", "ية"):
                return ("ي", len(clean_bait) + i) if with_index else "ي"

            # Skip certain characters
            if last_char in "ايىو":
                continue

            # Handle ha/ta marbuta cases
            if last_char in "هة" and second_last_char not in "اوي":
                continue

            # Handle kaf cases
            if last_char == "ك" and second_last_char not in "اوي":
                continue

            # Normalize special characters
            if last_char in self.config.special_endings:
                last_char = self.config.special_endings[last_char]

            return (last_char, len(clean_bait) + i + 1) if with_index else last_char

    def _analyze_qafiah_type(self, bait: str, short: bool = False) -> str:
        """
        Analyze the type of rhyme in the verse.

        Args:
            bait: The verse text
            short: Whether to return a short description

        Returns:
            Description of the rhyme type
        """
        clean_bait = self.normalizer.normalize(bait)
        rawwy, rawwy_index = self._get_rawwy_char(bait, with_index=True)

        features = []
        raw_text = self.normalizer.remove_unaccepted_chars(bait).strip()

        # Check for wasl (continuation)
        if self._has_wasl(clean_bait, rawwy_index, raw_text):
            features.append("الوصل" if short else "زاد لها الوصل بإشباع رويها")

        # Check for special endings
        if self._has_special_ending(clean_bait, rawwy_index, raw_text):
            char = clean_bait[-2] if len(clean_bait) > 1 else ""
            features.append("الوصل" if short else f"زاد لها الوصل بـ: {char}")

        # Check for khurooj
        if self._has_khurooj(clean_bait, rawwy_index):
            features.append("الوصل والخَروج" if short else "زاد لها الوصل و الخَروج")

        # Check for ridf
        if self._has_ridf(clean_bait, rawwy_index):
            features.append("والردف" if short else "زاد لها الردف")

        # Check for ta'sees
        if self._has_tasees(clean_bait, rawwy_index):
            features.append("والتأسيس" if short else "زاد لها التأسيس")

        prefix = f"قافية بحرف الروي: {rawwy} ، " if not short else ""
        return prefix + " ".join(features)

    def _has_wasl(self, clean_bait: str, rawwy_index: int, raw_text: str) -> bool:
        """Check if the verse has wasl feature."""
        return (
            rawwy_index == len(clean_bait) - 1 or (rawwy_index == len(clean_bait) - 2 and clean_bait[-1] in "ىاوي")
        ) and raw_text[-1] != self.config.sukun

    def _has_special_ending(self, clean_bait: str, rawwy_index: int, raw_text: str) -> bool:
        """Check if the verse has special ending feature."""
        return rawwy_index == len(clean_bait) - 2 and clean_bait[-1] in "كهة" and raw_text[-1] == self.config.sukun

    def _has_khurooj(self, clean_bait: str, rawwy_index: int) -> bool:
        """Check if the verse has khurooj feature."""
        return rawwy_index == len(clean_bait) - 3 and clean_bait[-2] in "كهة" and clean_bait[-1] in "اوي"

    def _has_ridf(self, clean_bait: str, rawwy_index: int) -> bool:
        """Check if the verse has ridf feature."""
        return clean_bait[rawwy_index - 1] in "اويآى"

    def _has_tasees(self, clean_bait: str, rawwy_index: int) -> bool:
        """Check if the verse has ta'sees feature."""
        return clean_bait[rawwy_index - 2] in "اآ"

    def analyze_verses(self, verses: List[str], short: bool = False) -> List[Tuple[str, str]]:
        """
        Analyze rhyme patterns in multiple verses.

        Args:
            verses: List of verses to analyze
            short: Whether to return short descriptions

        Returns:
            List of tuples containing (rawwy_char, rhyme_description)
        """
        return [(self._get_rawwy_char(verse), self._analyze_qafiah_type(verse, short)) for verse in verses]


class RhymeClassifier:
    """Main class for rhyme classification in Arabic poetry."""

    def __init__(self, config: Dict[str, Any] = None):
        rhyme_config = RhymeConfig(**config) if config else RhymeConfig()
        self.analyzer = RhymeAnalyzer(rhyme_config)

    def __call__(self, state: VerseAnalysisState) -> Dict[str, Any]:
        """
        Classify the rhyme of a given verse.

        Args:
            state: Dictionary containing the verse under 'verse' key

        Returns:
            Dictionary containing rhyme classification results
        """
        verse = state["verse"]
        verse = f"{verse['first_hemistich']} # {verse['second_hemistich']}"
        rawwy, rhyme_type = self.analyzer.analyze_verses([verse])[0]

        print_tool_output(output={"rawwy": rawwy, "type": rhyme_type, "ID": state["verse_id"]}, tool="RhymeClassifier")
        return {"analyzed_verse_metadata": {"rhyme": {"rawwy": rawwy, "type": rhyme_type}}}

    def classify_poem(self, verses: List[str]) -> Tuple[str, str]:
        """
        Classify the rhyme of an entire poem using majority voting.

        Args:
            verses: List of verses (strings) from the poem

        Returns:
            Tuple of (rawwy, rhyme_type) representing the dominant rhyme
        """
        verse_rhymes = []
        verse_types = []

        # Process each pair of hemistiches
        for i in range(0, len(verses), 2):
            try:
                state = {
                    "verse": {
                        "first_hemistich": verses[i],
                        "second_hemistich": verses[i + 1] if i + 1 < len(verses) else "",
                    },
                    "verse_id": None,
                }
                result = self(state)

                rhyme_info = result["analyzed_verse_metadata"]["rhyme"]
                verse_rhymes.append(rhyme_info["rawwy"])
                verse_types.append(rhyme_info["type"])
            except Exception as e:
                print(f"Error processing verse {i//2}: {str(e)}")
                continue

        # Perform majority voting
        if not verse_rhymes:
            raise ValueError("No valid rhymes found in the poem")

        dominant_rawwy = Counter(verse_rhymes).most_common(1)[0][0]

        # Get the most common type associated with the dominant rawwy
        dominant_type = Counter(
            type_ for rhyme, type_ in zip(verse_rhymes, verse_types) if rhyme == dominant_rawwy
        ).most_common(1)[0][0]

        return dominant_rawwy, dominant_type


# Example usage
if __name__ == "__main__":
    classifier = RhymeClassifier()
    state = {
        "verse": {
            "first_hemistich": "ألا ليت شعري هل أبيتن ليلة",
            "second_hemistich": "بجنب الغضى أزجي القلاص النواجيا",
        }
    }
    result = classifier(state)
    print(result)
