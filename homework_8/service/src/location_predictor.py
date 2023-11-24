import re
import emoji
from typing import List
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForTokenClassification
from optimum.pipelines import pipeline

tokenizer_name = 'xlm-roberta-base'
labels = ['S', 'O', 'B-LOC', 'I-LOC']
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for i, l in enumerate(labels)}


class LocationPredictor:
    """
    Wrapper for Location Prediction NER model.
    """

    def __init__(self, chkp_path: str, device: str, tresh: float):
        self.tresh = tresh
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.base_model = ORTModelForTokenClassification.from_pretrained(chkp_path, file='model.onnx')

        self.model = pipeline(
                'token-classification',
                model=self.base_model,
                tokenizer=self.tokenizer,
                aggregation_strategy='simple',
                device=device
                )

    def filter_text(text: str) -> str:
        text = emoji.replace_emoji(text)
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r' +', ' ', text)
        return text

    def post_process(locations: List[str]) -> List[str]:
        locations = [l for l in locations if (len(l) > 3 and any(c.isupper() for c in l)) or l == 'РФ' or l == 'США' or l == 'РБ']
        return locations

    def predict(self, texts: List[str]) -> List[List[str]]:
        """
        Process text input.

        Args:
            texts (List[str]): input texts.
        Returns:
            (List[List[str]]): list of list of locations.
        """

        filtered_texts = [
                LocationPredictor.filter_text(text)
                for text in texts
                ]

        all_locations = self.model(filtered_texts)

        filtered_locations = [[pred['word'] for pred in preds if pred['score'] >= self.tresh]
                              for preds in all_locations] 

        return [LocationPredictor.post_process(locations)
                for locations in filtered_locations]

