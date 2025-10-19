# to run server: uv run python main.py
from flask import Flask, request, jsonify
from factory import ModelFactory
from input_model import InputData

torch_model = ModelFactory().create_torch()
lgb_model = ModelFactory().create_lgb()
torch_onnx_model = ModelFactory().create_torch_onnx()
lgb_onnx_model = ModelFactory().create_lgb_onnx()

app = Flask(__name__)

@app.get("/")
def hello():
    return jsonify({"message": "Hello Flask"})

@app.post("/predict/torch/")
def torch_predict():
    payload = request.get_json(silent=True) or {}
    data = InputData(**payload)
    pred = torch_model.predict(data)
    return jsonify({"prediction": pred[0].item()})

@app.post("/predict/lgb/")
def lgb_predict():
    payload = request.get_json(silent=True) or {}
    data = InputData(**payload)
    pred = lgb_model.predict(data)
    return jsonify({"prediction": pred[0].item()})

@app.post("/predict/torch_onnx/")
def torch_onnx_predict():
    payload = request.get_json(silent=True) or {}
    data = InputData(**payload)
    pred = torch_onnx_model.predict(data)
    return jsonify({"prediction": pred[0].item()})

@app.post("/predict/lgb_onnx/")
def lgb_onnx_predict():
    payload = request.get_json(silent=True) or {}
    data = InputData(**payload)
    pred = lgb_onnx_model.predict(data)
    return jsonify({"prediction": pred[0].item()})

if __name__ == "__main__":
    # FastAPI のデフォルトに近い 8000 番ポートで起動
    app.run(host="0.0.0.0", port=8000)
