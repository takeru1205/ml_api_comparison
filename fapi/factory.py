from pathlib import Path
from mlmodels import TorchMLModel
from mlmodels import LgbMLModel
from mlmodels import OnnxMLModel

model_type_weights = {
    "torch": Path("../models/mlp_torch.pth"),
    "lgb": Path("../models/titanic_lgb_model.txt"),
    "torch_onnx": Path("../models/mlp_torch.onnx"),
    "lgb_onnx": Path("../models/titanic_lgb.onnx"),
}
preprocessor_path = Path("../models/mlp_torch_preprocess.joblib")


class ModelFactory():
    def create_torch(self):
        model = TorchMLModel()
        model.create_model(
            model_path=model_type_weights["torch"],
            preprocessor_path=preprocessor_path,
        )
        return model

    def create_lgb(self):
        model = LgbMLModel()
        model.create_model(
            model_path=model_type_weights["lgb"],
            preprocessor_path=preprocessor_path,
        )
        return model
   
    def create_torch_onnx(self):
        model = OnnxMLModel()
        model.create_model(
            model_path=model_type_weights["torch_onnx"],
            preprocessor_path=preprocessor_path,
        )
        return model
 
    def create_lgb_onnx(self):
        model = OnnxMLModel()
        model.create_model(
            model_path=model_type_weights["lgb_onnx"],
            preprocessor_path=preprocessor_path,
        )
        return model
