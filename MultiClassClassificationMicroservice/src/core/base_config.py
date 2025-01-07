import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer
)
import fasttext
from ftlangdetect import detect


class BaseConfig:
    """Construction level of country configs classes"""
    def __init__(self):
        fasttext.FastText.eprint = lambda x: None
        _ = detect("").get("lang")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.tokenizers = {}

    def _load_model(self, model_path) -> AutoModelForSequenceClassification:
        """Load model pretrained."""
        return AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True).to(self.device)

    def _load_tokenizer(self, model_path) -> AutoTokenizer:
        """Load pretrained tokenizer."""
        return AutoTokenizer.from_pretrained(model_path)
