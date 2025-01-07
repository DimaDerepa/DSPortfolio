import pandas as pd
from src.constants import textual_columns, digital_columns

from src.core.interfaces import Preprocessor
from src.core.data_processor import (
    compose_data_object_classification,
    get_language
)


class CorporativePreprocessor(Preprocessor):
    """Preprocessor class for Corporative flow."""
    def preprocess(self, df: pd.DataFrame, config) -> pd.DataFrame:
        # get composed objects and languages for input data
        df[textual_columns] = df[textual_columns].fillna('')
        df[digital_columns] = df[digital_columns].fillna(0)
        df = df.assign(text=compose_data_object_classification(df, 'CO'))
        df = df.assign(language=get_language(df["title"]))

        return df

