from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import load

from .base_ml_model import BaseMLModel


class Net(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, 1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x


class TorchMLModel(BaseMLModel):
    def __init__(self, in_dim=6, hid_dim=32):
        super().__init__()
        self.model = Net(in_dim, hid_dim)

    def create_model(
        self,
        model_path=Path("models/mlp_torch.pth"),
        preprocessor_path=Path("models/mlp_torch_preprocess.joblib"),
    ):
        self._model_load(model_path)
        self._preprocessor_load(preprocessor_path)

    def _model_load(self, model_path):
        self.model.load_state_dict(
            torch.load(
                model_path,
                weights_only=True,
            )
        )
        print(f"Load model from {model_path}")

    def preprocess(self, x):
        # get pydantic model input and return input type for onnx prediction
        label_encoded = self._label_encode(self._pydantic_model_to_df(x)).to_torch()

        imputer_transformed = self.imputer.transform(label_encoded)
        scaler_transformed = self.scaler.transform(imputer_transformed)
        return scaler_transformed

    def predict(self, x):
        preprocessed_x = torch.Tensor(self.preprocess(x))
        with torch.no_grad():
            preds = self.model(preprocessed_x).detach().ravel().cpu().numpy()
        return (preds > self.threshold).astype(int)


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
    print("="*10)

    model_path = Path("../../models/mlp_torch.pth")
    preprocessor_path = Path(
        "../../models/mlp_torch_preprocess.joblib",
    )

    model = TorchMLModel()
    model.create_model(
        model_path=model_path,
        preprocessor_path=preprocessor_path,
    )
    print("="*10)
    print(model.predict(input_data))
