import os

from src.core.base_processor import BaseProcessor
from src.germany.preprocess import GermanyPreprocessor
from src.germany.process import GermanyProcessorCore
from src.germany.postprocess import GermanyPostprocessor
from src.greece.preprocess import GreecePreprocessor
from src.greece.process import GreeceProcessorCore
from src.greece.postprocess import GreecePostprocessor
from src.germany.config import GermanyConfig
from src.greece.config import GreeceConfig


class ProcessorFactory:
    """Return class and specifics methods for each state based on MODEL_NAME"""
    @staticmethod
    def get_processor() -> BaseProcessor:
        model_name = os.getenv('MODEL_NAME')
        if model_name == "e5-base-co":
            config = GermanyConfig()
            return BaseProcessor(CorporativePreprocessor(), CorporativeProcessorCore(), CorporativePostprocessor(), config, 'CO')
        elif model_name == "e5-base-pr":
            config = GreeceConfig()
            return BaseProcessor(PrivatePreprocessor(), PrivateProcessorCore(), PrivatePostprocessor(), config, 'PR')
        raise ValueError(f"Processor for state is not implemented - {model_name}.")
