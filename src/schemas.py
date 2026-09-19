"""
src/schemas.py
==================

Pydantic schema definitions that act as the strict data contracts enforced
at every boundary of the Multimodal E-Commerce AI Pipeline:

    RawReview       -> output of the ingestion stage (src/ingestion.py)
    ProductRecord   -> product-details companion to RawReview (bonus model;
                       supports the "product details" half of ingestion)
    EnrichedOutput  -> output of the LLM enrichment stage (processing/)
    VideoMetadata   -> output of the Higgsfield video-generation stage

Any payload that fails validation against these models is rejected at the
boundary (raises `pydantic.ValidationError`) rather than silently flowing
downstream with bad data.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator


# --------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------
class VideoStatus(str, Enum):
    """Lifecycle states for a promo-video generation job."""

    QUEUED = "queued"
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


# --------------------------------------------------------------------------
# Product details (supports the review dataset; not explicitly requested
# but required for the "product details" half of the ingestion stage and
# for the review→product foreign key to mean anything).
# --------------------------------------------------------------------------
class ProductRecord(BaseModel):
    """A single product-details record."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    product_id: str = Field(..., min_length=1, max_length=50, description="Unique product identifier, e.g. 'P1001'")
    product_name: str = Field(..., min_length=1, max_length=200)
    category: str = Field(..., min_length=1, max_length=100)
    brand: str = Field(..., min_length=1, max_length=100)
    price_usd: float = Field(..., gt=0, description="Retail price in USD")

    @field_validator("product_id")
    @classmethod
    def product_id_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("product_id cannot be blank")
        return v


# --------------------------------------------------------------------------
# 1. Raw review data
# --------------------------------------------------------------------------
class RawReview(BaseModel):
    """A single raw customer product review, exactly as ingested from source."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    review_id: str = Field(..., min_length=1, max_length=50, description="Unique review identifier, e.g. 'R0001'")
    product_id: str = Field(..., min_length=1, max_length=50, description="Foreign key to ProductRecord.product_id")
    user_review: str = Field(..., min_length=1, max_length=5000, description="Free-text review body")
    rating: int = Field(..., ge=1, le=5, description="Star rating, 1-5 inclusive")
    reviewer_name: Optional[str] = Field(default=None, max_length=120)
    review_date: Optional[datetime] = Field(default=None)

    @field_validator("user_review")
    @classmethod
    def review_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("user_review cannot be blank")
        return v


# --------------------------------------------------------------------------
# 2. Enriched AI output
# --------------------------------------------------------------------------
class EnrichedAIOutput(BaseModel):
    """Structured AI enrichment result produced by the LLM enrichment stage."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    product_id: str = Field(..., min_length=1, max_length=50, description="Foreign key to ProductRecord.product_id")
    sentiment_score: float = Field(
        ..., ge=0.0, le=1.0, description="Aggregate sentiment score, 0.0 (very negative) to 1.0 (very positive)"
    )
    key_themes: List[str] = Field(
        ..., min_length=1, max_length=10, description="Top recurring themes/features mentioned across reviews"
    )
    promo_video_prompt: str = Field(
        ..., min_length=1, max_length=1000, description="Visual promo-video concept derived from sentiment & themes"
    )

    @field_validator("key_themes")
    @classmethod
    def themes_non_empty(cls, v: List[str]) -> List[str]:
        cleaned = [theme.strip() for theme in v if theme.strip()]
        if not cleaned:
            raise ValueError("key_themes must contain at least one non-empty theme")
        return cleaned

    @field_validator("promo_video_prompt")
    @classmethod
    def prompt_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("promo_video_prompt cannot be blank")
        return v


# Backward-compatible alias — earlier drafts of this schema were named
# `EnrichedOutput`; keep both names importable so nothing downstream breaks.
EnrichedOutput = EnrichedAIOutput


# --------------------------------------------------------------------------
# 3. Video metadata
# --------------------------------------------------------------------------
class VideoMetadata(BaseModel):
    """Metadata for a triggered promo-video job (Higgsfield or mock client)."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    video_id: str = Field(..., min_length=1, max_length=50, description="Unique video job identifier")
    product_id: str = Field(..., min_length=1, max_length=50, description="Foreign key to ProductRecord.product_id")
    video_url: Optional[HttpUrl] = Field(default=None, description="Final asset URL, set once status == 'succeeded'")
    status: VideoStatus = Field(default=VideoStatus.QUEUED)

    @model_validator(mode="after")
    def url_required_when_succeeded(self) -> "VideoMetadata":
        if self.status == VideoStatus.SUCCEEDED and self.video_url is None:
            raise ValueError("video_url must be set when status is 'succeeded'")
        return self
