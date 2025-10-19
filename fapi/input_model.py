from pydantic import BaseModel


class InputData(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float


if __name__ == "__main__":
    input_data = InputData(
        Pclass=0,
        Sex="male",
        Age=10,
        SibSp=1,
        Parch=0,
        Fare=11.1,
    )

    print(input_data)
