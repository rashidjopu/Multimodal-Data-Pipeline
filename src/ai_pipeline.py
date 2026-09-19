"""
src/ai_pipeline.py
======================

Stage 2 of the Multimodal E-Commerce AI Pipeline: AI Sentiment Extraction.

`AIPipeline` reads raw reviews from `data/raw_warehouse.db` (populated by
`src/ingestion.py`), batches them per product, sends them to an LLM
(Anthropic Claude or OpenAI GPT, selectable via config) asking for strict
JSON matching `EnrichedAIOutput`, validates every response with Pydantic,
and persists the validated rows into the `enriched_reviews` table.

Design principles:
  - Provider-agnostic: same code path drives either Anthropic or OpenAI.
  - Mock-first: with no API key configured, falls back to a deterministic
    MockLLMClient so the whole pipeline is runnable/testable for free.
  - Strict contracts: an LLM response that fails Pydantic validation is
    retried once with a corrective re-prompt, then — if still invalid —
    routed to a dead-letter file instead of corrupting the warehouse.

Run directly:
    python -m src.ai_pipeline
    python -m src.ai_pipeline --provider openai
    python -m src.ai_pipeline --db-path data/raw_warehouse.db --force-mock
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sqlite3
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import ValidationError

from src.schemas import EnrichedAIOutput, ProductRecord, RawReview

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/raw_warehouse.db")
DEAD_LETTER_PATH = Path("data/processed/enrichment_dead_letter.jsonl")

_STOPWORDS = {
    "the", "and", "for", "with", "this", "that", "was", "are", "but", "have",
    "its", "it's", "not", "you", "your", "just", "very", "really", "like",
    "from", "than", "then", "them", "they", "would", "could", "into", "some",
    "after", "once", "even", "while", "when", "still", "also", "been", "were",
}

SYSTEM_PROMPT = (
    "You are a strict JSON-generating API for an e-commerce analytics pipeline. "
    "Given a product name and a set of customer reviews, respond with ONLY a single "
    "JSON object — no markdown fences, no preamble, no commentary — with exactly "
    "these keys:\n"
    '  "product_id": string,\n'
    '  "sentiment_score": float between 0.0 (very negative) and 1.0 (very positive),\n'
    '  "key_themes": array of 3-6 short strings, the most salient recurring '
    "themes/features/complaints across the reviews,\n"
    '  "promo_video_prompt": string, a single vivid sentence describing a visual '
    "concept for a 5-second product promo video that leans into the positive "
    "themes.\n"
    "Output must be valid JSON parseable by `json.loads`. Nothing else."
)


# --------------------------------------------------------------------------
# Provider clients
# --------------------------------------------------------------------------
class LLMCallError(RuntimeError):
    """Raised when an LLM provider call fails after all retries."""


class _AnthropicClient:
    """Thin wrapper around the Anthropic Messages API."""

    def __init__(self, api_key: str, model: str):
        import anthropic  # local import: optional dependency

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def complete(self, user_prompt: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return "".join(block.text for block in response.content if getattr(block, "type", None) == "text")


class _OpenAIClient:
    """Thin wrapper around the OpenAI Chat Completions API (JSON mode)."""

    def __init__(self, api_key: str, model: str):
        import openai  # local import: optional dependency

        self._client = openai.OpenAI(api_key=api_key)
        self._model = model

    def complete(self, user_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or ""


class MockLLMClient:
    """
    Deterministic, dependency-free stand-in for a real LLM.

    Used automatically when no API key is configured (or when --force-mock
    is passed), so the enrichment stage — and everything downstream of it —
    can be run and tested with zero cost and zero network access. Derives a
    plausible sentiment score from star ratings and extracts naive keyword
    themes from the review text instead of hallucinating an LLM response.
    """

    def __init__(self, model: str = "mock-llm-v1"):
        self._model = model

    def complete(self, user_prompt: str) -> str:
        # The prompt embeds the data we need as a JSON blob after a marker —
        # see AIPipeline._build_user_prompt — so we can parse it back out
        # instead of doing real NLP, keeping this fully deterministic.
        match = re.search(r"<<<DATA>>>(.*?)<<<END_DATA>>>", user_prompt, re.DOTALL)
        if not match:
            raise LLMCallError("MockLLMClient could not locate embedded data payload in prompt.")
        payload = json.loads(match.group(1))

        product_id = payload["product_id"]
        product_name = payload["product_name"]
        ratings = payload["ratings"]
        texts: List[str] = payload["review_texts"]

        avg_rating = sum(ratings) / len(ratings) if ratings else 3.0
        sentiment_score = round(max(0.0, min(1.0, (avg_rating - 1) / 4)), 3)

        words = re.findall(r"[a-zA-Z']{5,}", " ".join(texts).lower())
        counted = Counter(w for w in words if w not in _STOPWORDS)
        top_words = [w for w, _ in counted.most_common(5)] or ["quality", "value"]
        key_themes = top_words[:5]

        tone = "energetic, upbeat" if sentiment_score >= 0.6 else "honest, measured"
        promo_video_prompt = (
            f"A {tone} 5-second product shot of the {product_name}, highlighting "
            f"{', '.join(key_themes[:3])} with quick dynamic cuts and bold on-screen text."
        )

        return json.dumps(
            {
                "product_id": product_id,
                "sentiment_score": sentiment_score,
                "key_themes": key_themes,
                "promo_video_prompt": promo_video_prompt,
            }
        )


# --------------------------------------------------------------------------
# Simple retry helper (kept dependency-light; tenacity is a valid drop-in
# swap here if you prefer the decorator-based API listed in requirements.txt)
# --------------------------------------------------------------------------
def _with_retries(fn, *, max_attempts: int, description: str):
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - deliberately broad, we log & retry
            last_exc = exc
            logger.warning("%s failed on attempt %d/%d: %s", description, attempt, max_attempts, exc)
    raise LLMCallError(f"{description} failed after {max_attempts} attempts") from last_exc


@dataclass
class ProductReviewBundle:
    product: ProductRecord
    reviews: List[RawReview]


# --------------------------------------------------------------------------
# AIPipeline
# --------------------------------------------------------------------------
class AIPipeline:
    """
    Orchestrates the AI sentiment-extraction stage: reads raw reviews,
    batches them per product, calls the configured LLM provider, validates
    the structured response, and persists it to `enriched_reviews`.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        db_path: Path = DEFAULT_DB_PATH,
        batch_size: int = 10,
        max_retries: int = 3,
        force_mock: bool = False,
    ):
        self.db_path = Path(db_path)
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.provider = (provider or os.getenv("LLM_PROVIDER", "anthropic")).lower()

        self.client, self.model_name, self.mock_mode = self._build_client(
            provider=self.provider, model=model, api_key=api_key, force_mock=force_mock
        )

    # ---- client construction -------------------------------------------------
    def _build_client(self, provider: str, model: Optional[str], api_key: Optional[str], force_mock: bool):
        if force_mock:
            logger.warning("AIPipeline running in MOCK MODE (--force-mock). No real LLM calls will be made.")
            return MockLLMClient(), "mock-llm-v1", True

        if provider == "anthropic":
            key = api_key or os.getenv("ANTHROPIC_API_KEY")
            model_name = model or os.getenv("LLM_MODEL_ANTHROPIC", "claude-sonnet-4-6")
            if not key:
                logger.warning(
                    "ANTHROPIC_API_KEY not set — AIPipeline falling back to MockLLMClient. "
                    "Set ANTHROPIC_API_KEY (or pass --provider openai with OPENAI_API_KEY) for real enrichment."
                )
                return MockLLMClient(), "mock-llm-v1", True
            try:
                return _AnthropicClient(api_key=key, model=model_name), model_name, False
            except ImportError:
                logger.warning("`anthropic` package not installed — falling back to MockLLMClient.")
                return MockLLMClient(), "mock-llm-v1", True

        elif provider == "openai":
            key = api_key or os.getenv("OPENAI_API_KEY")
            model_name = model or os.getenv("LLM_MODEL_OPENAI", "gpt-4o-mini")
            if not key:
                logger.warning(
                    "OPENAI_API_KEY not set — AIPipeline falling back to MockLLMClient. "
                    "Set OPENAI_API_KEY (or pass --provider anthropic with ANTHROPIC_API_KEY) for real enrichment."
                )
                return MockLLMClient(), "mock-llm-v1", True
            try:
                return _OpenAIClient(api_key=key, model=model_name), model_name, False
            except ImportError:
                logger.warning("`openai` package not installed — falling back to MockLLMClient.")
                return MockLLMClient(), "mock-llm-v1", True

        else:
            raise ValueError(f"Unknown provider '{provider}'. Use 'anthropic' or 'openai'.")

    # ---- database helpers -----------------------------------------------------
    def _get_connection(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _ensure_enriched_table(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS enriched_reviews (
                product_id         TEXT PRIMARY KEY,
                sentiment_score    REAL NOT NULL CHECK (sentiment_score BETWEEN 0.0 AND 1.0),
                key_themes         TEXT NOT NULL,   -- JSON-encoded list[str]
                promo_video_prompt TEXT NOT NULL,
                provider           TEXT NOT NULL,
                model_used         TEXT NOT NULL,
                review_count       INTEGER NOT NULL,
                created_at         TEXT NOT NULL,
                FOREIGN KEY (product_id) REFERENCES products (product_id)
            );
            """
        )
        conn.commit()

    def _load_product_review_bundles(self, conn: sqlite3.Connection) -> List[ProductReviewBundle]:
        products_rows = conn.execute(
            "SELECT product_id, product_name, category, brand, price_usd FROM products ORDER BY product_id;"
        ).fetchall()

        bundles: List[ProductReviewBundle] = []
        for product_id, product_name, category, brand, price_usd in products_rows:
            review_rows = conn.execute(
                """
                SELECT review_id, product_id, user_review, rating, reviewer_name, review_date
                FROM reviews WHERE product_id = ? ORDER BY review_id;
                """,
                (product_id,),
            ).fetchall()

            if not review_rows:
                logger.info("Skipping product %s: no reviews found.", product_id)
                continue

            reviews = [
                RawReview(
                    review_id=r[0],
                    product_id=r[1],
                    user_review=r[2],
                    rating=r[3],
                    reviewer_name=r[4],
                    review_date=r[5],
                )
                for r in review_rows
            ]
            product = ProductRecord(
                product_id=product_id, product_name=product_name, category=category, brand=brand, price_usd=price_usd
            )
            bundles.append(ProductReviewBundle(product=product, reviews=reviews))

        return bundles

    def _save_enriched_output(self, conn: sqlite3.Connection, result: EnrichedAIOutput, review_count: int) -> None:
        conn.execute(
            """
            INSERT INTO enriched_reviews
                (product_id, sentiment_score, key_themes, promo_video_prompt, provider, model_used, review_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(product_id) DO UPDATE SET
                sentiment_score    = excluded.sentiment_score,
                key_themes         = excluded.key_themes,
                promo_video_prompt = excluded.promo_video_prompt,
                provider           = excluded.provider,
                model_used         = excluded.model_used,
                review_count       = excluded.review_count,
                created_at         = excluded.created_at;
            """,
            (
                result.product_id,
                result.sentiment_score,
                json.dumps(result.key_themes),
                result.promo_video_prompt,
                "mock" if self.mock_mode else self.provider,
                self.model_name,
                review_count,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()

    def _write_dead_letter(self, product_id: str, raw_response: str, error: str) -> None:
        DEAD_LETTER_PATH.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "product_id": product_id,
            "raw_response": raw_response,
            "error": error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with DEAD_LETTER_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        logger.error("Dead-lettered enrichment failure for product %s -> %s", product_id, DEAD_LETTER_PATH)

    # ---- prompt construction ---------------------------------------------------
    def _build_user_prompt(self, bundle: ProductReviewBundle) -> str:
        reviews_batch = bundle.reviews[: self.batch_size]
        review_lines = "\n".join(
            f'- ({r.rating}/5) "{r.user_review}"' for r in reviews_batch
        )
        embedded_data = json.dumps(
            {
                "product_id": bundle.product.product_id,
                "product_name": bundle.product.product_name,
                "ratings": [r.rating for r in reviews_batch],
                "review_texts": [r.user_review for r in reviews_batch],
            }
        )
        return (
            f"Product: {bundle.product.product_name} (id: {bundle.product.product_id})\n"
            f"Category: {bundle.product.category} | Brand: {bundle.product.brand}\n\n"
            f"Customer reviews ({len(reviews_batch)} of {len(bundle.reviews)} total):\n"
            f"{review_lines}\n\n"
            "Respond with only the JSON object described in your system instructions.\n"
            f"<<<DATA>>>{embedded_data}<<<END_DATA>>>"
        )

    # ---- response parsing --------------------------------------------------
    @staticmethod
    def _strip_code_fences(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
            text = re.sub(r"```$", "", text).strip()
        return text

    def _parse_and_validate(self, product_id: str, raw_text: str) -> EnrichedAIOutput:
        cleaned = self._strip_code_fences(raw_text)
        data = json.loads(cleaned)  # may raise json.JSONDecodeError
        data.setdefault("product_id", product_id)
        return EnrichedAIOutput(**data)

    # ---- core per-product call with retry + corrective re-prompt --------------
    def _enrich_single_product(self, bundle: ProductReviewBundle) -> Optional[EnrichedAIOutput]:
        prompt = self._build_user_prompt(bundle)
        product_id = bundle.product.product_id
        last_raw = ""
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                last_raw = _with_retries(
                    lambda: self.client.complete(prompt),
                    max_attempts=1,  # network-level retries happen at call-site; this loop handles parse retries
                    description=f"LLM call for {product_id}",
                )
                return self._parse_and_validate(product_id, last_raw)
            except (json.JSONDecodeError, ValidationError, LLMCallError) as exc:
                last_error = exc
                logger.warning(
                    "Enrichment attempt %d/%d for %s failed validation/parsing: %s",
                    attempt,
                    self.max_retries,
                    product_id,
                    exc,
                )
                # Corrective re-prompt: make the JSON-only instruction more forceful.
                prompt = (
                    prompt
                    + "\n\nYour previous response was invalid. Reply with ONLY raw JSON, "
                    "no markdown, no explanation, matching the exact schema."
                )

        self._write_dead_letter(product_id, last_raw, str(last_error))
        return None

    # ---- public API -------------------------------------------------------
    def enrich_product_reviews(self) -> List[EnrichedAIOutput]:
        """
        Read every product's reviews from SQLite, enrich each via the LLM
        (batched per product, capped at `self.batch_size` reviews per call),
        validate the structured output, and persist it to `enriched_reviews`.

        Returns the list of successfully validated EnrichedAIOutput records.
        Products whose LLM response can't be validated after retries are
        skipped and logged to the dead-letter file rather than crashing the
        whole run.
        """
        conn = self._get_connection()
        results: List[EnrichedAIOutput] = []
        try:
            self._ensure_enriched_table(conn)
            bundles = self._load_product_review_bundles(conn)
            logger.info(
                "Enriching %d products (%s mode, provider=%s, model=%s)...",
                len(bundles),
                "MOCK" if self.mock_mode else "LIVE",
                self.provider,
                self.model_name,
            )

            for bundle in bundles:
                enriched = self._enrich_single_product(bundle)
                if enriched is None:
                    continue
                self._save_enriched_output(conn, enriched, review_count=len(bundle.reviews))
                results.append(enriched)
                logger.info(
                    "Enriched %-8s sentiment=%.3f themes=%s",
                    enriched.product_id,
                    enriched.sentiment_score,
                    enriched.key_themes,
                )

            logger.info("Enrichment complete: %d/%d products succeeded.", len(results), len(bundles))
        finally:
            conn.close()

        return results


# --------------------------------------------------------------------------
# Module-level convenience function (mirrors the class method 1:1)
# --------------------------------------------------------------------------
def enrich_product_reviews(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    db_path: Path = DEFAULT_DB_PATH,
    batch_size: int = 10,
    force_mock: bool = False,
) -> List[EnrichedAIOutput]:
    """Convenience wrapper: build an AIPipeline and run enrichment end-to-end."""
    pipeline = AIPipeline(provider=provider, model=model, db_path=db_path, batch_size=batch_size, force_mock=force_mock)
    return pipeline.enrich_product_reviews()


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Run AI sentiment/theme enrichment over ingested reviews.")
    parser.add_argument("--provider", choices=["anthropic", "openai"], default=None, help="LLM provider to use.")
    parser.add_argument("--model", default=None, help="Override the default model for the chosen provider.")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH, help="SQLite DB path.")
    parser.add_argument("--batch-size", type=int, default=10, help="Max reviews per product included per LLM call.")
    parser.add_argument(
        "--force-mock", action="store_true", help="Skip real API calls and use the deterministic MockLLMClient."
    )
    args = parser.parse_args()

    enrich_product_reviews(
        provider=args.provider,
        model=args.model,
        db_path=args.db_path,
        batch_size=args.batch_size,
        force_mock=args.force_mock,
    )


if __name__ == "__main__":
    main()
