from pathlib import Path
import torch
import torch.nn as nn


def load_onnx(
    model_path: Path=Path("models/mlp_torch.onnx"),
):
    import onnx
    model = onnx.load(model_path)
    print(f"Load {model_path}")
    return model


def load_lgb(
    model_path: Path=Path("models/titanic_lgb_model.txt"),
):
    import lightgbm as lgb
    model = lgb.Booster(model_file=model_path)
    print(f"Load {model_path}")
    return model


class Net(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hid_dim, 2)  # binary logit
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)  # logits
        return x


def load_torch(
    model_path: Path=Path("models/mlp_torch_best.pth"),
    in_dim=6, hid_dim=32,
):
    model = Net(in_dim, hid_dim)
    model.load_state_dict(
        torch.load(
            model_path,
            weights_only=True,
        )
    )
    print(f"Load from {model_path}")
    return model


if __name__ == "__main__":
    torch_onnx = load_onnx("../models/mlp_torch.onnx")
    lgb_onnx = load_onnx("../models/titanic_lgb.onnx")

    lgb_model = load_lgb("../models/titanic_lgb_model.txt")

    torch_model = load_torch("../models/mlp_torch.pth")
