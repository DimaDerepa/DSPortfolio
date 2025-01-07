import pandera as pa
from pandera import (
    Check,
    Column,
    DataFrameSchema
)
import pandas as pd

from src.constants import (
    MAX_LEN_DESCRIPTION,
    MAX_LEN_FEATURES,
    MAX_LEN_SUBTYPE,
    MAX_LEN_TITLE
)


class DynamicDatasetSchema:
    """
    Dynamic dataset schema class organized by pandera. Create dynamic dataset schema for
     validation input data depends on state and columns
     """
    def __init__(self, state: str, text_columns: list, numeric_columns: list):
        self.state = state
        self.text_columns = text_columns
        self.numeric_columns = numeric_columns
        self.default_text = 'Unbekannt' if state == 'CO' else 'Unknown'
        self.schema = self.create_schema()

    def create_schema(self):
        """
        Dynamically creates a schema based on the provided text and numeric columns.
        """
        schema_dict = {}

        # Create schema for text columns
        for col in self.text_columns:
            schema_dict[col] = Column(
                str,
                nullable=True,
                coerce=True,
                checks=[Check(lambda x: x.fillna("").notnull())]
            )

        # Create schema for numeric columns
        for col in self.numeric_columns:
            schema_dict[col] = Column(
                float,
                nullable=True,
                coerce=True,
                checks=[Check(lambda x: x.fillna(0).notnull())]
            )

        # Add additional columns
        schema_dict['state'] = Column(str, nullable=False, coerce=True)
        schema_dict['uid'] = Column(str, nullable=False, coerce=True)

        return DataFrameSchema(schema_dict)

    @staticmethod
    def truncate_text_fields(df):
        """Additional method for preprocess input text columns."""
        df['title'] = df['title'].str.slice(0, MAX_LEN_TITLE)
        df['description'] = df['description'].str.slice(0, MAX_LEN_DESCRIPTION)
        df['features'] = df['features'].str.slice(0, MAX_LEN_FEATURES)
        df['subtype'] = df['subtype'].str.slice(0, MAX_LEN_SUBTYPE)
        return df

    def validate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses and validates the data against the generated schema.
        """
        # Replace None or missing values in text and numeric columns
        missing_info = []

        for col in self.text_columns:
            if col not in data.columns:
                missing_info.append((col, 'Missing column'))
                data[col] = ""
            else:
                null_uids = data[data[col].isnull()]['uid'].tolist()
                if null_uids:
                    missing_info.extend([(col, uid) for uid in null_uids])
                data[col].fillna("", inplace=True)

        for col in self.numeric_columns:
            if col not in data.columns:
                missing_info.append((col, 'Missing column'))
                data[col] = 0.0
            else:
                null_uids = data[data[col].isnull()]['uid'].tolist()
                if null_uids:
                    missing_info.extend([(col, uid) for uid in null_uids])
                data[col].fillna(0.0, inplace=True)

        # Log missing information
        if missing_info:
            for col, info in missing_info:
                print(f"Missing or null value in column '{col}' for UID: {info}")

        # Preprocess text columns
        for col in self.text_columns:
            pattern = r"\n+|\t+"
            data.loc[:, col] = data[col].str.replace(pattern, "", regex=True)
            data.loc[:, col] = data[col].str.replace(";", ",")

        # Validate schema
        validated_df = self.create_schema().validate(data)

        # Filter valid columns
        valid_columns = set(self.text_columns + self.numeric_columns + ['country_code', 'uid'])
        filtered_df = validated_df.loc[:, validated_df.columns.intersection(valid_columns)]
        filtered_df = self.truncate_text_fields(filtered_df)
        filtered_df.loc[filtered_df['subtype'] == "", 'subtype'] = self.default_text

        return filtered_df


class ResultModelSchema(pa.DataFrameSchema):
    """
    Pandera schema for the results returned after processing.
    """
    schema = pa.DataFrameSchema({
        "uid": Column(str, nullable=False),
        "type_id": Column(int, checks=pa.Check.ge(0), nullable=False),
        "max_probability": Column(float, checks=pa.Check.in_range(0.0, 1.0), nullable=False),
        "group": Column(str, nullable=False),
        "group_probability": Column(float, checks=pa.Check.in_range(0.0, 1.0), nullable=False),
        "language": Column(str, nullable=False)
    })


class EnvModelSchema(pa.DataFrameSchema):
    """
    Pandera schema for environment-related metadata.
    """
    schema = pa.DataFrameSchema({
        "api_version": Column(int, nullable=False),
        "model_version": Column(str, nullable=False)
    })


class ResponseModel:
    """
    Combines ResultModelSchema and EnvModelSchema.
    """
    @staticmethod
    def validate_response(result: pd.DataFrame, environment: pd.DataFrame) -> dict:
        # Check data by schemas
        validated_result = ResultModelSchema.schema.validate(result)
        validated_env = EnvModelSchema.schema.validate(environment)

        # Return results as dictionary
        return {
            "result": validated_result.to_dict(orient="records"),
            "environment": validated_env.to_dict(orient="records")[0]
        }

    class Config:
        arbitrary_types_allowed = True
