# to run server: uv run uvicorn main:app
from fastapi import FastAPI
from factory import ModelFactory
from input_model import InputData

torch_model = ModelFactory().create_torch()
lgb_model = ModelFactory().create_lgb()
torch_onnx_model = ModelFactory().create_torch_onnx()
lgb_onnx_model = ModelFactory().create_lgb_onnx()

app = FastAPI()

@app.get("/")
async def hello():
    return {"message": "Hello FastAPI"}

@app.post("/predict/torch/", status_code=200)
async def torch_predict(data: InputData):
    pred = torch_model.predict(data)
    return {"prediction": pred[0].item()}

@app.post("/predict/lgb/", status_code=200)
async def lgb_predict(data: InputData):
    pred = lgb_model.predict(data)
    return {"prediction": pred[0].item()}

@app.post("/predict/torch_onnx/", status_code=200)
async def torch_onnx_predict(data: InputData):
    pred = torch_onnx_model.predict(data)
    return {"prediction": pred[0].item()}

@app.post("/predict/lgb_onnx/", status_code=200)
async def lgb_onnx_predict(data: InputData):
    pred = lgb_onnx_model.predict(data)
    return {"prediction": pred[0].item()}
