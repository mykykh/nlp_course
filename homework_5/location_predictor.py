import re
from typing import List
from transformers import (
        AutoTokenizer,
        AutoModelForTokenClassification,
        pipeline
        )
from peft import PeftModel

base_model_name = 'xlm-roberta-base'
labels = ['S', 'O', 'B-LOC', 'I-LOC']
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for i, l in enumerate(labels)}


class LocationPredictor:
    """
    Wrapper for Location Prediction NER model.
    """

    def __init__(self, chkp_path: str, device: str, tresh: float):
        self.tresh = tresh
        
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        self.base_model = AutoModelForTokenClassification.from_pretrained(
                base_model_name,
                label2id=label2id,
                id2label=id2label
                )

        lora_model = PeftModel.from_pretrained(
                self.base_model,
                chkp_path
                ).merge_and_unload()

        self.model = pipeline(
                'token-classification',
                model=lora_model,
                tokenizer=self.tokenizer,
                aggregation_strategy='simple',
                device=device
                )

    def filter_text(text: str) -> str:
        pattern = re.compile('['
            '\U0001F600-\U0001F64F'
            '\U0001F300-\U0001F5FF'
            '\U0001F680-\U0001F6FF'
            '\U00010000-\U00010FFF'
            '\U000024C2-\U0001F251'
            '\u2600-\u2B55'
            ']+')
        text = pattern.sub('', text)
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

