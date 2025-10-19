from pathlib import Path
import polars as pl
from joblib import load


class BaseMLModel:
    def __init__(self):
        self.model = None
        self.imputer = None
        self.scaler = None
        self.sex_label_mapping = {
            "female": 0,
            "male": 1,
        }
        self.threshold = 0.5

    def create_model(self):
        raise NotImplementedError

    def _model_load(self, model_path: Path):
        raise NotImplementedError

    def _preprocessor_load(self, preprocessor_path: Path):
        preprocessor = load(preprocessor_path)
        self.imputer = preprocessor["imputer"]
        self.scaler = preprocessor["scaler"]
        print(f"Load preprocessor from {preprocessor_path}")

    def _pydantic_model_to_df(self, pydantic_model_input) -> pl.DataFrame:
        model_input_dict = pl.from_dict(pydantic_model_input.model_dump())
        return model_input_dict

    def _label_encode(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            pl.col(
                "Sex"
            ).replace_strict(
                self.sex_label_mapping
            ).alias("Sex")
        )

    def preprocess(self, x):
        return self._label_encode(self._pydantic_model_to_df(x))

    def predict(self):
        raise NotImplementedError
