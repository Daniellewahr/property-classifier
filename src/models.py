from typing import Literal

from pydantic import BaseModel, Field


Category = Literal["Nursery", "SEN School", "Food Store", "None"]
Confidence = Literal["High", "Medium", "Low"]


class PreprocessedListing(BaseModel):
    id: str
    context: str


class ClassificationResult(BaseModel):
    id: str
    category: Category
    confidence: Confidence
    reasoning: str = Field(max_length=500)