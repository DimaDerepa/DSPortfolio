from src.constants import MAX_OBJECTS_BATCH
from src.utils.utils import format_response
from src.utils.versioning import get_version
from src.core.interfaces import *
from src.core.models import *

import pandas as pd
from typing import (
    Dict,
    List
)


class BaseProcessor:
    """
    Main point of processing, work as compositor and facade
    compose config, preprocessor, processor, and postprocessor
    execute results, validate inputs, map output for casafari types
    """
    def __init__(self, preprocessor: Preprocessor, processor_core: ProcessorCore, postprocessor: Postprocessor, config, state):
        self.config = config
        self.version = get_version()
        self.state = state
        self.preprocessor = preprocessor
        self.processor_core = processor_core
        self.postprocessor = postprocessor
        self.dynamic_schema = DynamicDatasetSchema(
            state=self.state,
            text_columns=self.config.textual_columns,
            numeric_columns=self.config.digital_columns
        )

    def execute(self, data: List[Dict], batch_size: int = MAX_OBJECTS_BATCH) -> dict:
        """
        Executes the data processing flow in batches:
        - Preprocess -> Process -> Postprocess -> Map Results

        Args:
            data: List of input objects for prediction.
            batch_size: Number of objects in each batch.

        Returns:
            dict: Contains the processed results or error information.
        """
        try:
            # Convert input list to DataFrame
            data_df = pd.DataFrame(data)
            validated_data_df = self.dynamic_schema.validate(data_df)
        except Exception as e:
            return {
                "error": "Validation failed",
                "details": str(e),
                "uids": [obj.get("uid", "unknown") for obj in data]
            }

        # Split data into batches
        batches = [validated_data_df[i:i + batch_size] for i in range(0, len(validated_data_df), batch_size)]
        results = []

        for batch in batches:
            uids = batch["uid"].tolist()
            try:
                # Preprocess
                preprocessed_data = self.preprocessor.preprocess(batch, config=self.config)
            except Exception as e:
                return {
                    "error": "Preprocessing failed",
                    "details": str(e),
                    "uids": uids
                }

            try:
                # Process
                processed_data = self.processor_core.process(preprocessed_data, config=self.config)
            except Exception as e:
                return {
                    "error": "Processing failed",
                    "details": str(e),
                    "uids": uids
                }

            try:
                if self.state == 'CO':
                    processed_data = self.map_results(processed_data)
                # Postprocess
                postprocessed_data = self.postprocessor.postprocess(processed_data, config=self.config)
            except Exception as e:
                return {
                    "error": "Postprocessing failed",
                    "details": str(e),
                    "uids": uids
                }

            try:
                # temporary structure remove when fix private postprocessing
                # Map Results
                if self.state == 'PR':
                    postprocessed_data = self.map_results(postprocessed_data)
                results.append(postprocessed_data)
            except Exception as e:
                return {
                    "error": "Mapping results failed",
                    "details": str(e),
                    "uids": uids
                }

        try:
            # Concatenate all batches into a single DataFrame
            final_result_df = pd.concat(results)
        except Exception as e:
            return {
                "error": "Final concatenation failed",
                "details": str(e),
                "uids": [uid for batch in batches for uid in batch["uid"].tolist()]
            }

        # Format the final result as a response
        return format_response(
            final_result_df[["uid", "type_id", "max_probability", "group",
                             "group_probability", "language"]],
            self.version
        )

    def map_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Maps the results' type IDs using the `map_to_casafari` dictionary.
        Args:
            df: DataFrame of results with type IDs to map.

        Returns:
            pd.DataFrame: Updated DataFrame with mapped type IDs.
        """
        df["type_id"] = df["type_id"].map(self.config.map_to_casafari).fillna(df["type_id"])
        return df