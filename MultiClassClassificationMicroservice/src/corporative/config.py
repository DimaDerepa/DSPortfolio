from src.core.base_config import BaseConfig
from src.constants import (
    digital_columns_co,
    groups_encodes,
    new2old_subgroup_co,
    textual_columns_co,
    types_group_map_co
)
from src.definitions import (
    CO_MODEL_PATH_GROUP,
    CO_MODEL_PATH_SUBGROUP
)


class CorporativeConfig(BaseConfig):
    """
    Corporative Config. Here specified all objects which need to be loaded when Corporative part of service start.
    """
    def __init__(self):
        super().__init__()

        self.encoded_groups = groups_encodes
        self.models = {
            'group': self._load_model(CO_MODEL_PATH_GROUP),
            'subgroup': self._load_model(CO_MODEL_PATH_SUBGROUP)
        }

        self.tokenizers = {
            'group': self._load_tokenizer(CO_MODEL_PATH_GROUP)
        }
        self.types_group_map_co = types_group_map_co
        self.map_to_casafari = new2old_subgroup_co
        self.digital_columns = digital_columns_co
        self.textual_columns = textual_columns_co
