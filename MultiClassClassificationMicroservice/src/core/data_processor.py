from typing import List

import pandas
from ftlangdetect import detect


def get_language(field: pandas.Series) -> List[str]:
    """
    detect languages for all objects in dataframe.
    Args:
        field (pd.DataSeries): Field of dataframe for detect languages.
    Returns:
        List[str]: list of languages.
    """
    return [detect(obj).get("lang") for obj in field.tolist()]


def compose_data_object_classification(data_dicts, state: str) -> List[str]:
    """
    Create composed objects from data_dicts.
    Args:
        data_dicts: dataframe of objects classifications.
        state: ("PR" for private, "CO" for corporative).

    Returns:
        List[str]: composed objects for country.
    """

    composed_data_objects = []
    # create composed objects
    if state == "CO":
        composed_data_objects = (
                "<s>###DESCRIPTION: " + data_dicts["description"] +
                "</s>###FEATURES: " + data_dicts["features"] + "</s> ###TITLE: " + data_dicts["title"] +
                "</s>###META INFO: " + data_dicts["subtype"].fillna('Unknown') + "</s>"
        ).tolist()

    elif state == "CO":
        composed_data_objects = (
                "<s>###TITLE: " + data_dicts["title"] + "</s>###FEATURES: " + data_dicts["features"] +
                "</s>###META INFO: " + data_dicts["subtype"].fillna('Unbekannt') + "</s>###DESCRIPTION: " +
                data_dicts["description"] + "</s>"
        ).tolist()

    return composed_data_objects

