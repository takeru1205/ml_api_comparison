# ml_api_comparison

機械学習モデルをデプロイする時に、選択しがちなフレームワークごとの実行速度を雑に比較する。


## 比較項目

### モデル

- PyTorch
- LightGBM
- PyTorch(onnx)
- LightGBM(onnx)

### フレームワーク

- Python
  - FastAPI
  - Flask
- Rust(axum, tokio)
  - ort(onnx model only)
  - tract(onnx model only)


## タスク

Titanic

学習コードは `notebooks/` にある。

### 特徴量

- Pclass(Int)
- Sex(String)
- Age(Int)
- SibSp(Int)
- Parch(Int)
- Fare(Float)

### 前処理

- Sex →  Label Encoding
- Age → Fill Null(median)
- Age → Standard Scaler
- Fare → Standard Scaler


## 結果

雑なまとめ

| Framework        | 平均推論時間 (ms)   |
| ---------------- | ------------- |
| **FastAPI**      | 約 **1.75 ms** |
| **Flask**        | 約 **2.0 ms**  |
| **ORT (Rust)**   | 約 **0.47 ms** |
| **Tract (Rust)** | 約 **0.46 ms** |


各フレームワークごとの結果

| Framework        | モデル             | Avg   | Min   | Max   | Total    | Pop Var | Sample Var |
| ---------------- | --------------- | ----- | ----- | ----- | -------- | ------- | ---------- |
| **FastAPI**      | PyTorch         | 1.761 | 1.591 | 7.440 | 1760.629 | 0.043   | 0.043      |
|                  | LightGBM        | 1.758 | 1.548 | 3.142 | 1758.414 | 0.017   | 0.017      |
|                  | PyTorch (onnx)  | 1.542 | 1.400 | 2.534 | 1541.626 | 0.009   | 0.009      |
|                  | LightGBM (onnx) | 1.757 | 1.562 | 2.716 | 1757.161 | 0.010   | 0.010      |
| **Flask**        | PyTorch         | 2.013 | 1.836 | 6.441 | 2013.075 | 0.037   | 0.037      |
|                  | LightGBM        | 2.170 | 2.005 | 3.072 | 2169.753 | 0.009   | 0.009      |
|                  | PyTorch (onnx)  | 1.774 | 1.612 | 2.348 | 1773.535 | 0.008   | 0.008      |
|                  | LightGBM (onnx) | 1.990 | 1.798 | 3.943 | 1989.903 | 0.023   | 0.023      |
| **ORT (Rust)**   | PyTorch (onnx)  | 0.426 | 0.402 | 0.804 | 425.580  | 0.001   | 0.001      |
|                  | LightGBM (onnx) | 0.599 | 0.533 | 0.917 | 598.631  | 0.000   | 0.000      |
| **Tract (Rust)** | PyTorch (onnx)  | 0.428 | 0.403 | 0.693 | 427.609  | 0.000   | 0.000      |
|                  | LightGBM (onnx) | 0.510 | 0.488 | 0.764 | 510.188  | 0.000   | 0.000      |

