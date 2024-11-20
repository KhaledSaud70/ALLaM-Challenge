import torch
from diacritization_evaluation import util
from .utils import ArabicTextEncoder

from models import CBHGModel

from .llm_registry import LLMRegistry


class Diacritizer:
    def __init__(self, model: torch.nn.Module):
        self.text_encoder = ArabicTextEncoder()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.start_symbol_id = self.text_encoder.start_symbol_id
        self.model = model

    def diacritize_text(self, text: str):
        seq = self.text_encoder.input_to_sequence(text)
        return self.diacritize_batch(torch.LongTensor([seq]).to(self.device))

    def diacritize_batch(self, batch):
        self.model.eval()
        inputs = batch["src"]
        lengths = batch["lengths"]
        outputs = self.model(inputs.to(self.device), lengths.to("cpu"))
        diacritics = outputs["diacritics"]
        predictions = torch.max(diacritics, 2).indices
        sentences = []

        for src, prediction in zip(inputs, predictions):
            sentence = self.text_encoder.combine_text_and_diacritics(
                list(src.detach().cpu().numpy()),
                list(prediction.detach().cpu().numpy()),
            )
            sentences.append(sentence)

        return sentences


@LLMRegistry.register("diacritizer_model")
class DiacritizerModel:
    def __init__(self) -> None:
        self.pad_idx = 0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.text_encoder = ArabicTextEncoder()
        self.start_symbol_id = self.text_encoder.start_symbol_id

        self.model = CBHGModel(
            embedding_dim=256,
            inp_vocab_size=44,
            targ_vocab_size=17,
            use_prenet=False,
            prenet_sizes=[512, 256],
            cbhg_gru_units=256,
            cbhg_filters=16,
            cbhg_projections=[128, 256],
            post_cbhg_layers_units=[256, 256],
            post_cbhg_use_batch_norm=True,
        )

        self.model = self.model.to(self.device)
        self.diacritizer = Diacritizer(model=self.model)

    def load_model(self, weights_path: str = None):
        saved_model = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(saved_model["model_state_dict"])

    def collate_fn(self, data):
        """
        Padding the input and output sequences
        """

        def merge(sequences):
            lengths = [len(seq) for seq in sequences]
            padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i, :end] = seq[:end]
            return padded_seqs, lengths

        data.sort(key=lambda x: len(x[0]), reverse=True)

        # separate source and target sequences
        src_seqs, trg_seqs, original = zip(*data)

        # merge sequences (from tuple of 1D tensor to 2D tensor)
        src_seqs, src_lengths = merge(src_seqs)
        trg_seqs, trg_lengths = merge(trg_seqs)

        batch = {
            "original": original,
            "src": src_seqs,
            "target": trg_seqs,
            "lengths": torch.LongTensor(src_lengths),  # src_lengths = trg_lengths
        }
        return batch

    def get_batch(self, sentence):
        data = self.text_encoder.clean(sentence)
        text, inputs, diacritics = util.extract_haraqat(data)
        inputs = torch.Tensor(self.text_encoder.encode_input("".join(inputs)))
        diacritics = torch.Tensor(self.text_encoder.encode_target(diacritics))
        batch = self.collate_fn([(inputs, diacritics, text)])
        return batch

    def infer(self, sentence):
        self.model.eval()
        batch = self.get_batch(sentence)
        predicted = self.diacritizer.diacritize_batch(batch)
        return predicted[0]


if __name__ == "__main__":
    model = DiacritizerModel()
    model.load_model(model_path="weights/diacritizer_model_weights.pt")
    print(model.infer(sentence="يا من إليه أشتكي من هجره # هل أنت تدري لوعةَ المهجُور"))
