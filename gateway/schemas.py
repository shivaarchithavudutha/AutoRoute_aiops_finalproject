"""Request/response models (the API contract documented in the LLD)."""
from typing import List, Literal

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    text: str = Field(..., max_length=5000, description="Complaint text (max 5000 characters)")

    @field_validator("text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be empty")
        return v


class ClassScore(BaseModel):
    category: str
    score: float


class PredictResponse(BaseModel):
    category: str
    confidence: float
    top3: List[ClassScore]


class TriggerRequest(BaseModel):
    action: Literal["retrain", "full"] = "retrain"


class FeedbackRequest(BaseModel):
    predicted: str = Field(..., max_length=100)
    correct: bool


class ExplainRequest(BaseModel):
    text: str = Field(..., max_length=5000)
    category: str = Field(..., max_length=100)
