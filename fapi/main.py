from fastapi import FastAPI
from pydantic impot BaseModel

class InputData(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float

app = FastAPI()


@app.get("/")
def hello():
    return {"message": "Hello FastAPI"}


@app.post("/predict/")
def predict(data: InputData):
    return {"prediction": 0}
