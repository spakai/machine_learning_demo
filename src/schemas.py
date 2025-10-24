from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    temperature: float = Field(
        ...,
        ge=-50,
        le=90,
        description="Ambient temperature in Celsius.",
    )


class PredictionResponse(BaseModel):
    predicted_sales: float = Field(..., description="Predicted number of ice-creams sold.")
    temperature: float
    model_version: str = Field(..., description="File name or identifier of the trained model.")
    r2: float
    rmse: float
