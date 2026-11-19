from typing import Literal
from pydantic import BaseModel, Field


class ProductRow(BaseModel):

    Age: float

    SystolicBP: float

    DiastolicBP: float

    BS: float

    BodyTemp: float

    HeartRate: float

    RiskLevel: Literal["high risk", "low risk", "mid risk"]
