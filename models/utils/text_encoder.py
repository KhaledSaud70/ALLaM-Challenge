from dataclasses import dataclass
from enum import Enum
from typing import List

from .text_cleaner import valid_arabic_cleaners


class ArabicDiacritics(Enum):
    """Enumeration of Arabic diacritical marks (harakat)."""

    NONE = ""
    FATHA = "َ"
    FATHATAH = "ً"
    DAMMA = "ُ"
    DAMMATAN = "ٌ"
    KASRA = "ِ"
    KASRATAN = "ٍ"
    SUKUN = "ْ"
    SHADDAH = "ّ"
    SHADDAH_FATHA = "َّ"
    SHADDAH_FATHATAH = "ًّ"
    SHADDAH_DAMMA = "ُّ"
    SHADDAH_DAMMATAN = "ٌّ"
    SHADDAH_KASRA = "ِّ"
    SHADDAH_KASRATAN = "ٍّ"

    @classmethod
    def get_description(cls, diacritic: str) -> str:
        """Get a human-readable description of the diacritic."""
        descriptions = {
            cls.NONE.value: "No Diacritic",
            cls.FATHA.value: "Fatha",
            cls.FATHATAH.value: "Fathatah",
            cls.DAMMA.value: "Damma",
            cls.DAMMATAN.value: "Dammatan",
            cls.KASRA.value: "Kasra",
            cls.KASRATAN.value: "Kasratan",
            cls.SUKUN.value: "Sukun",
            cls.SHADDAH.value: "Shaddah",
            cls.SHADDAH_FATHA.value: "Shaddah + Fatha",
            cls.SHADDAH_FATHATAH.value: "Shaddah + Fathatah",
            cls.SHADDAH_DAMMA.value: "Shaddah + Damma",
            cls.SHADDAH_DAMMATAN.value: "Shaddah + Dammatan",
            cls.SHADDAH_KASRA.value: "Shaddah + Kasra",
            cls.SHADDAH_KASRATAN.value: "Shaddah + Kasratan",
        }
        return descriptions.get(diacritic, "Unknown")


@dataclass
class EncoderConfig:
    """Configuration for the text encoder."""

    reverse_input: bool = False
    reverse_target: bool = False
    pad_token: str = "P"
    start_symbol: str = "s"


class ArabicTextEncoder:
    """A text encoder for Arabic text with diacritics."""

    def __init__(self, config: EncoderConfig = EncoderConfig()):
        self.config = config
        self._initialize_chars()
        self._initialize_mappings()

    def _initialize_chars(self):
        """Initialize input and target character sets."""
        self.input_chars = list("بض.غىهظخة؟:طس،؛فندؤلوئآك-يذاصشحزءمأجإ ترقعث")
        self.target_chars = [d.value for d in ArabicDiacritics] + [self.config.start_symbol]

        # Add padding token
        self.input_chars = [self.config.pad_token] + self.input_chars
        self.target_chars = [self.config.pad_token] + self.target_chars

    def _initialize_mappings(self):
        """Initialize character-to-id and id-to-character mappings."""
        self.input_char_to_id = {char: idx for idx, char in enumerate(self.input_chars)}
        self.input_id_to_char = {idx: char for idx, char in enumerate(self.input_chars)}

        self.target_char_to_id = {char: idx for idx, char in enumerate(self.target_chars)}
        self.target_id_to_char = {idx: char for idx, char in enumerate(self.target_chars)}

        self.pad_id = self.input_char_to_id[self.config.pad_token]
        self.start_symbol_id = self.target_char_to_id[self.config.start_symbol]

    def encode_input(self, text: str) -> List[int]:
        """Convert input text to sequence of ids."""
        if self.config.reverse_input:
            text = text[::-1]
        return [
            self.input_char_to_id[char]
            for char in text
            if char in self.input_char_to_id and char != self.config.pad_token
        ]

    def encode_target(self, text: str) -> List[int]:
        """Convert target text to sequence of ids."""
        if self.config.reverse_target:
            text = text[::-1]
        return [
            self.target_char_to_id[char]
            for char in text
            if char in self.target_char_to_id and char != self.config.pad_token
        ]

    def decode_input(self, sequence: List[int]) -> str:
        """Convert sequence of ids back to input text."""
        return "".join(
            self.input_id_to_char[idx] for idx in sequence if idx in self.input_id_to_char and idx != self.pad_id
        )

    def decode_target(self, sequence: List[int]) -> str:
        """Convert sequence of ids back to target text."""
        return "".join(
            self.target_id_to_char[idx] for idx in sequence if idx in self.target_id_to_char and idx != self.pad_id
        )

    def combine_text_and_diacritics(self, input_ids: List[int], output_ids: List[int]) -> str:
        """Combine input text with its corresponding diacritics."""
        result = ""
        for i, input_id in enumerate(input_ids):
            if input_id == self.pad_id:
                break
            result += self.input_id_to_char[input_id]
            result += self.target_id_to_char[output_ids[i]]
        return result

    def clean(self, text: str) -> str:
        return valid_arabic_cleaners(text)
