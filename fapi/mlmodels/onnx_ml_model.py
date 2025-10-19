from pathlib import Path
from joblib import load

import numpy as np
import onnxruntime as rt
from fastapi import FastAPI

from .base_ml_model import BaseMLModel


class OnnxMLModel(BaseMLModel):
    def create_model(
        self,
        model_path=Path("models/mlp_torch.onnx"),
        preprocessor_path=Path("models/mlp_torch_preprocess.joblib"),
    ):
        self._model_load(model_path)
        self._preprocessor_load(preprocessor_path)

    def _model_load(self, model_path):
        self.model = rt.InferenceSession(model_path)
        print(f"Load {model_path}")

    def preprocess(self, x):
        # get pydantic model input and return input type for onnx prediction
        label_encoded = self._label_encode(self._pydantic_model_to_df(x)).to_numpy().astype(dtype=np.float32)

        imputer_transformed = self.imputer.transform(label_encoded)
        scaler_transformed = self.scaler.transform(imputer_transformed)
        return scaler_transformed

    def predict(self, x):
        preprocessed_x = self.preprocess(x)

        input_name = self.model.get_inputs()[0].name
        # Make predictions using the ONNX model
        output_name = self.model.get_outputs()[0].name

        return (
            self.model.run(
                [output_name],
                {input_name: preprocessed_x},
            )[0] > self.threshold
        ).ravel().astype(int)

if __name__ == "__main__":
    import sys
    sys.path.append("../")
    from input_model import InputData

    input_data = InputData(
        Pclass=0,
        Sex="male",
        Age=10,
        SibSp=1,
        Parch=0,
        Fare=11.1,
    )
    print(input_data)
    print("="*10, "LightGBM ONNX")

    model_path = Path("../../models/titanic_lgb.onnx")
    preprocessor_path = Path(
        "../../models/mlp_torch_preprocess.joblib",
    )

    model = OnnxMLModel()
    model.create_model(
        model_path=model_path,
        preprocessor_path=preprocessor_path,
    )
    print("="*10)
    print(model.predict(input_data))

    print("*"*20)
    print("="*10, "Torch ONNX")
    model_path = Path("../../models/mlp_torch.onnx")
    preprocessor_path = Path(
        "../../models/mlp_torch_preprocess.joblib",
    )

    model = OnnxMLModel()
    model.create_model(
        model_path=model_path,
        preprocessor_path=preprocessor_path,
    )
    print("="*10)
    print(model.predict(input_data))
