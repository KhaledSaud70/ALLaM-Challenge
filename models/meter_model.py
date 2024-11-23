from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pyarabic.araby as araby
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from transformers import AutoTokenizer, TFBertModel

from .llm_registry import LLMRegistry


@dataclass
class ModelConfig:
    """Configuration class for the meter model model."""

    max_sequence_length: int = 32
    classes_num: int = 16
    bert_model_name: str = "faisalq/bert-base-arapoembert"
    tokenizer_name: str = "faisalq/bert-medium-arapoembert"
    learning_rate: float = 5e-05


class MeterClassifier:
    """Arabic poetry meter model using AraPoemBERT."""

    # Class-level constants
    METERS = [
        "الطويل",
        "الكامل",
        "البسيط",
        "الخفيف",
        "الوافر",
        "الرجز",
        "الرمل",
        "المتقارب",
        "السريع",
        "المنسرح",
        "المجتث",
        "الهزج",
        "المديد",
        "المتدارك",
        "المقتضب",
        "المضارع",
    ]

    # Mapping for model output conversion
    MODEL_OUTPUT_IDS = [14, 2, 0, 3, 15, 11, 12, 10, 13, 7, 6, 1, 4, 9, 8, 5]

    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.model = None
        self.tokenizer = None
        self._initialize_model()
        self._initialize_tokenizer()
        self._setup_id_mapping()

    def _setup_id_mapping(self):
        """Set up the mapping between model outputs and meter indices."""
        self.ids_mapping = {k: v for v, k in enumerate(self.MODEL_OUTPUT_IDS)}

    def _initialize_model(self):
        input_ids = Input(shape=(self.config.max_sequence_length,), dtype=tf.int32, name="input_ids")
        input_mask = Input(shape=(self.config.max_sequence_length,), dtype=tf.int32, name="attention_mask")

        bert = TFBertModel.from_pretrained(self.config.bert_model_name, from_pt=True)

        output = bert([input_ids, input_mask])[1]  # pooled_output
        output = Dense(self.config.classes_num, activation="softmax", name="output")(output)

        self.model = tf.keras.Model(inputs=[input_ids, input_mask], outputs=output)

        self.model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss=CategoricalCrossentropy(from_logits=True),
            metrics=[CategoricalAccuracy("balanced_accuracy")],
        )

    def _initialize_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)

    def load_weights(self, weights_path: str):
        self.model.load_weights(weights_path)

    @staticmethod
    def preprocess_text(text: str) -> str:
        """Preprocess input text by removing diacritics."""
        return araby.strip_diacritics(text)

    def _tokenize(self, text: str) -> Dict[str, tf.Tensor]:
        return self.tokenizer(
            text=text,
            add_special_tokens=True,
            max_length=self.config.max_sequence_length,
            truncation=True,
            padding="max_length",
            return_tensors="tf",
        )

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict the meter of given Arabic poetry text.

        Args:
            text: Input Arabic text

        Returns:
            Dictionary containing prediction results:
            - meter: Predicted meter name
            - confidence: Confidence score
        """
        # Preprocess and tokenize
        processed_text = self.preprocess_text(text)
        inputs = self._tokenize(processed_text)

        # Model inference
        predictions = self.model.predict({"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]})

        # Process predictions
        predicted_class = tf.argmax(predictions, axis=1).numpy()[0]
        confidence = float(np.max(predictions))

        # Convert to meter name
        meter_index = self.ids_mapping[predicted_class]
        predicted_meter = self.METERS[meter_index]

        return {
            "meter": predicted_meter,
            "confidence": confidence,
        }


@LLMRegistry.register("meter_classifier")
def get_meter_model() -> MeterClassifier:
    """Factory function for class registry integration."""
    return MeterClassifier(ModelConfig())
