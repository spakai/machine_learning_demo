from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    day_of_week: str = Field(
        ...,
        min_length=3,
        description="Day of the week, e.g. Monday.",
    )
    month: str = Field(
        ...,
        min_length=3,
        description="Month name, e.g. April.",
    )
    temperature: float = Field(
        ...,
        ge=-50,
        le=130,
        description="Ambient temperature in Fahrenheit.",
    )
    rainfall: float = Field(
        ...,
        ge=0,
        description="Total rainfall (inches) for the day.",
    )


class PredictionResponse(BaseModel):
    predicted_sales: float = Field(..., description="Predicted number of ice-creams sold.")
    temperature: float
    rainfall: float
    day_of_week: str
    month: str
    model_version: str = Field(..., description="File name or identifier of the trained model.")
    train_r2: float
    train_rmse: float
    test_r2: float
    test_rmse: float
