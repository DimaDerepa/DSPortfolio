from abc import (
    ABC,
    abstractmethod
)
"""
Interfaces of main pipelines functions for each state for organize 
abstract level between base processor and state processors.
"""


class Preprocessor(ABC):
    @abstractmethod
    def preprocess(self, data, config):
        pass


class ProcessorCore(ABC):
    @abstractmethod
    def process(self, data, config):
        pass


class Postprocessor(ABC):
    @abstractmethod
    def postprocess(self, data, config):
        pass