import pandas as pd

from src.constants import types_map_pr, force_list_pr
from src.core.data_processor import compose_data_object_classification, get_language
from src.core.interfaces import Preprocessor


class PrivatePreprocessor(Preprocessor):
    """Preprocessor class for Private."""
    def preprocess(self, data, config) -> pd.DataFrame:

        # Compose `text` and `language` fields
        data = data.assign(text=compose_data_object_classification(data, 'PR'))
        data = data.assign(language=get_language(data["title"]))

        return data
