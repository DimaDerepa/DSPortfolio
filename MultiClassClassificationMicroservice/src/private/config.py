import json
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.core.base_config import BaseConfig
from src.constants import (
    digital_columns_pr,
    keywords_recheck_pr,
    new2old_pr,
    textual_columns_pr,
    threshold_private
)
from src.definitions import (
    ENCODER_PATH,
    private_subtypes,
    MODEL_PATH
)


class PrivateConfig(BaseConfig):
    """
    Private Config. Here specified all objects which need to be loaded when Private part of service start.
    """
    def __init__(self):
        super().__init__()
        self.models = {
            'main': self._load_model(MODEL_PATH)
        }
        self.tokenizers = {
            'main': self._load_tokenizer(MODEL_PATH)
        }
        self.sbert_model = SentenceTransformer(ENCODER_PATH).to(self.device)
        self.private_subtypes_dict = json.load(open(private_subtypes, 'r'))
        self.keywords_recheck_pr = keywords_recheck_pr
        self.threshold_private = threshold_private
        self.map_to_casafari = new2old_pr
        self.digital_columns = digital_columns_pr
        self.textual_columns = textual_columns_pr

