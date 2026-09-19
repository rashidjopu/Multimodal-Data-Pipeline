"""
tests/test_pipeline.py
==========================

Unit tests for the Multimodal E-Commerce AI Pipeline, focused on:
  A. Pydantic schema parsing/validation (src/schemas.py) — the contracts
     enforced at every pipeline boundary.
  B. The mocked LLM client used by AIPipeline (src/ai_pipeline.py) —
     deterministic enrichment output + dead-letter behavior on bad responses.
  C. The mocked Higgsfield client used by HiggsfieldVideoClient
     (src/video_generator.py) — job submission, polling, and the guarantee
     that mock mode never touches the network.

Run:
    pytest tests/test_pipeline.py -v
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.ai_pipeline import AIPipeline, LLMCallError, MockLLMClient
from src.schemas import (
    EnrichedAIOutput,
    ProductRecord,
    RawReview,
    VideoMetadata,
    VideoStatus,
)
from src.video_generator import (
    HighSentimentProduct,
    HiggsfieldVideoClient,
    VideoGenerationError,
)


# ==========================================================================
# Fixtures
# ==========================================================================
@pytest.fixture
def seeded_db(tmp_path: Path) -> Path:
    """
    A minimal, isolated SQLite DB with 2 products and 4 reviews (2 each),
    matching the schema produced by src/ingestion.py. Used to test
    AIPipeline and HiggsfieldVideoClient without depending on the full
    ingestion stage.
    """
    db_path = tmp_path / "test_warehouse.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE products (product_id TEXT PRIMARY KEY, product_name TEXT, "
        "category TEXT, brand TEXT, price_usd REAL);"
    )
    conn.execute(
        "CREATE TABLE reviews (review_id TEXT PRIMARY KEY, product_id TEXT, user_review TEXT, "
        "rating INTEGER, reviewer_name TEXT, review_date TEXT);"
    )
    conn.executemany(
        "INSERT INTO products VALUES (?, ?, ?, ?, ?);",
        [
            ("P1001", "AeroFit Pro Wireless Earbuds", "Electronics", "AeroSound", 79.99),
            ("P1002", "Summit Trail 40L Hiking Backpack", "Outdoor & Sports", "Summit Gear", 129.99),
        ],
    )
    conn.executemany(
        "INSERT INTO reviews VALUES (?, ?, ?, ?, ?, ?);",
        [
            ("R0001", "P1001", "Fantastic sound quality and battery life, love these earbuds!", 5, "A.", None),
            ("R0002", "P1001", "Comfortable fit, great value for the price point.", 4, "B.", None),
            ("R0003", "P1002", "Sturdy backpack, held up great on a week-long trek.", 5, "C.", None),
            ("R0004", "P1002", "Zippers feel a bit cheap but overall solid capacity.", 3, "D.", None),
        ],
    )
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def sample_product() -> HighSentimentProduct:
    return HighSentimentProduct(
        product_id="P1001",
        product_name="AeroFit Pro Wireless Earbuds",
        sentiment_score=0.9,
        key_themes=["battery", "comfort", "sound quality"],
        promo_video_prompt="A dynamic 5-second shot of the earbuds glowing on a charging case.",
    )


# ==========================================================================
# A. Pydantic schema validation (src/schemas.py)
# ==========================================================================
class TestProductRecord:
    def test_valid_product_record(self):
        product = ProductRecord(
            product_id="P1001", product_name="Widget", category="Gadgets", brand="Acme", price_usd=19.99
        )
        assert product.product_id == "P1001"
        assert product.price_usd == 19.99

    def test_rejects_non_positive_price(self):
        with pytest.raises(ValidationError):
            ProductRecord(product_id="P1001", product_name="Widget", category="Gadgets", brand="Acme", price_usd=0)

    def test_rejects_blank_product_id(self):
        with pytest.raises(ValidationError):
            ProductRecord(product_id="   ", product_name="Widget", category="Gadgets", brand="Acme", price_usd=9.99)


class TestRawReview:
    def test_valid_review(self):
        review = RawReview(review_id="R0001", product_id="P1001", user_review="Great product!", rating=5)
        assert review.rating == 5

    @pytest.mark.parametrize("bad_rating", [0, 6, -1, 10])
    def test_rejects_rating_out_of_range(self, bad_rating):
        with pytest.raises(ValidationError):
            RawReview(review_id="R0001", product_id="P1001", user_review="Great product!", rating=bad_rating)

    def test_rejects_blank_review_text(self):
        with pytest.raises(ValidationError):
            RawReview(review_id="R0001", product_id="P1001", user_review="   ", rating=5)


class TestEnrichedAIOutput:
    def test_valid_enriched_output(self):
        result = EnrichedAIOutput(
            product_id="P1001",
            sentiment_score=0.85,
            key_themes=["battery life", "comfort"],
            promo_video_prompt="Show the earbuds in a workout setting.",
        )
        assert 0.0 <= result.sentiment_score <= 1.0
        assert len(result.key_themes) == 2

    @pytest.mark.parametrize("bad_score", [-0.1, 1.1, 2.0])
    def test_rejects_sentiment_score_out_of_range(self, bad_score):
        with pytest.raises(ValidationError):
            EnrichedAIOutput(
                product_id="P1001", sentiment_score=bad_score, key_themes=["x"], promo_video_prompt="A prompt."
            )

    def test_rejects_empty_key_themes(self):
        with pytest.raises(ValidationError):
            EnrichedAIOutput(product_id="P1001", sentiment_score=0.5, key_themes=[], promo_video_prompt="A prompt.")

    def test_rejects_blank_promo_prompt(self):
        with pytest.raises(ValidationError):
            EnrichedAIOutput(product_id="P1001", sentiment_score=0.5, key_themes=["x"], promo_video_prompt="   ")


class TestVideoMetadata:
    def test_succeeded_status_requires_video_url(self):
        """Cross-field validator: status='succeeded' without a video_url must fail."""
        with pytest.raises(ValidationError):
            VideoMetadata(video_id="V0001", product_id="P1001", video_url=None, status=VideoStatus.SUCCEEDED)

    def test_succeeded_status_with_video_url_is_valid(self):
        video = VideoMetadata(
            video_id="V0001",
            product_id="P1001",
            video_url="https://mock-cdn.higgsfield.ai/videos/v1.mp4",
            status=VideoStatus.SUCCEEDED,
        )
        assert video.status == VideoStatus.SUCCEEDED
        assert str(video.video_url).startswith("https://")

    def test_queued_status_without_video_url_is_valid(self):
        """A queued/processing job legitimately has no URL yet — must NOT raise."""
        video = VideoMetadata(video_id="V0002", product_id="P1001", video_url=None, status=VideoStatus.QUEUED)
        assert video.video_url is None

    def test_rejects_malformed_url(self):
        with pytest.raises(ValidationError):
            VideoMetadata(
                video_id="V0003", product_id="P1001", video_url="not-a-url", status=VideoStatus.SUCCEEDED
            )


# ==========================================================================
# B. AIPipeline + MockLLMClient (src/ai_pipeline.py)
# ==========================================================================
class TestMockLLMClient:
    def test_returns_valid_json_matching_schema(self):
        client = MockLLMClient()
        embedded = json.dumps(
            {
                "product_id": "P1001",
                "product_name": "AeroFit Pro Wireless Earbuds",
                "ratings": [5, 4, 5],
                "review_texts": ["Amazing battery life", "Great comfort", "Superb sound"],
            }
        )
        prompt = f"irrelevant preamble\n<<<DATA>>>{embedded}<<<END_DATA>>>"

        raw_response = client.complete(prompt)
        data = json.loads(raw_response)  # must be valid JSON
        result = EnrichedAIOutput(**data)  # must satisfy the strict schema

        assert result.product_id == "P1001"
        assert 0.0 <= result.sentiment_score <= 1.0
        assert len(result.key_themes) >= 1

    def test_high_ratings_produce_high_sentiment(self):
        client = MockLLMClient()
        embedded = json.dumps(
            {"product_id": "P1", "product_name": "X", "ratings": [5, 5, 5], "review_texts": ["great great great"]}
        )
        data = json.loads(client.complete(f"<<<DATA>>>{embedded}<<<END_DATA>>>"))
        assert data["sentiment_score"] > 0.9

    def test_low_ratings_produce_low_sentiment(self):
        client = MockLLMClient()
        embedded = json.dumps(
            {"product_id": "P1", "product_name": "X", "ratings": [1, 1, 1], "review_texts": ["terrible terrible"]}
        )
        data = json.loads(client.complete(f"<<<DATA>>>{embedded}<<<END_DATA>>>"))
        assert data["sentiment_score"] < 0.1

    def test_raises_when_data_marker_missing(self):
        client = MockLLMClient()
        with pytest.raises(LLMCallError):
            client.complete("a prompt with no embedded data marker at all")


class TestAIPipeline:
    def test_defaults_to_mock_client_when_no_api_key_configured(self, monkeypatch, seeded_db):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        pipeline = AIPipeline(provider="anthropic", db_path=seeded_db)

        assert pipeline.mock_mode is True
        assert isinstance(pipeline.client, MockLLMClient)

    def test_force_mock_flag_short_circuits_provider_selection(self, seeded_db):
        pipeline = AIPipeline(db_path=seeded_db, force_mock=True)
        assert pipeline.mock_mode is True
        assert isinstance(pipeline.client, MockLLMClient)

    def test_enrich_product_reviews_end_to_end(self, seeded_db):
        pipeline = AIPipeline(db_path=seeded_db, force_mock=True)
        results = pipeline.enrich_product_reviews()

        assert len(results) == 2  # both seeded products should enrich successfully
        product_ids = {r.product_id for r in results}
        assert product_ids == {"P1001", "P1002"}
        for r in results:
            assert isinstance(r, EnrichedAIOutput)
            assert 0.0 <= r.sentiment_score <= 1.0
            assert len(r.key_themes) >= 1

        # Verify it was actually persisted to the enriched_reviews table.
        conn = sqlite3.connect(seeded_db)
        row_count = conn.execute("SELECT COUNT(*) FROM enriched_reviews;").fetchone()[0]
        conn.close()
        assert row_count == 2

    def test_dead_letters_on_persistently_invalid_llm_response(self, seeded_db, monkeypatch, tmp_path):
        dead_letter_path = tmp_path / "dead_letter.jsonl"
        monkeypatch.setattr("src.ai_pipeline.DEAD_LETTER_PATH", dead_letter_path)

        pipeline = AIPipeline(db_path=seeded_db, force_mock=True, max_retries=2)

        class BrokenClient:
            def complete(self, prompt: str) -> str:
                return "this is not valid json at all"

        pipeline.client = BrokenClient()  # override after construction
        results = pipeline.enrich_product_reviews()

        assert results == []
        assert dead_letter_path.exists()
        lines = dead_letter_path.read_text().strip().splitlines()
        assert len(lines) == 2  # one dead-letter entry per seeded product
        for line in lines:
            record = json.loads(line)
            assert record["product_id"] in {"P1001", "P1002"}
            assert "error" in record


# ==========================================================================
# C. HiggsfieldVideoClient + mock mode (src/video_generator.py)
# ==========================================================================
class TestHiggsfieldVideoClientMockMode:
    def test_defaults_to_mock_mode_without_api_key(self, monkeypatch, tmp_path):
        monkeypatch.delenv("HIGGSFIELD_API_KEY", raising=False)
        client = HiggsfieldVideoClient(db_path=tmp_path / "warehouse.db")

        assert client.mock_mode is True
        # In mock mode no HTTP session should even be constructed.
        assert not hasattr(client, "_session")

    def test_trigger_video_job_returns_job_id_in_mock_mode(self, tmp_path, sample_product):
        client = HiggsfieldVideoClient(db_path=tmp_path / "warehouse.db", force_mock=True)
        job_id = client.trigger_video_job(sample_product)

        assert isinstance(job_id, str)
        assert job_id.startswith("mock-job-")

    def test_poll_job_status_reaches_completed_with_valid_url(self, tmp_path, sample_product):
        client = HiggsfieldVideoClient(
            db_path=tmp_path / "warehouse.db", force_mock=True, poll_interval_seconds=0.01
        )
        job_id = client.trigger_video_job(sample_product)
        result = client.poll_job_status(job_id, sample_product.product_id)

        assert result["status"] == "COMPLETED"
        assert result["video_url"].startswith("https://mock-cdn.higgsfield.ai/videos/")
        assert job_id in result["video_url"]

    def test_mock_mode_never_touches_the_network(self, monkeypatch, tmp_path, sample_product):
        """Patch requests.Session.post/get to explode if ever called — mock mode must not call them."""
        import requests

        def _explode(*args, **kwargs):
            raise AssertionError("Network call attempted in mock mode!")

        monkeypatch.setattr(requests.Session, "post", _explode)
        monkeypatch.setattr(requests.Session, "get", _explode)

        client = HiggsfieldVideoClient(
            db_path=tmp_path / "warehouse.db", force_mock=True, poll_interval_seconds=0.01
        )
        job_id = client.trigger_video_job(sample_product)
        result = client.poll_job_status(job_id, sample_product.product_id)

        assert result["status"] == "COMPLETED"  # completed without ever touching requests.Session

    def test_generate_videos_for_high_sentiment_products_end_to_end(self, seeded_db):
        # Seed an enriched_reviews row above the sentiment threshold for one product.
        conn = sqlite3.connect(seeded_db)
        conn.execute(
            "CREATE TABLE enriched_reviews (product_id TEXT PRIMARY KEY, sentiment_score REAL, "
            "key_themes TEXT, promo_video_prompt TEXT, provider TEXT, model_used TEXT, "
            "review_count INTEGER, created_at TEXT);"
        )
        conn.execute(
            "INSERT INTO enriched_reviews VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
            (
                "P1001",
                0.92,
                json.dumps(["battery", "comfort"]),
                "A vibrant shot of the earbuds mid-workout.",
                "mock",
                "mock-llm-v1",
                2,
                "2026-01-01T00:00:00",
            ),
        )
        conn.commit()
        conn.close()

        client = HiggsfieldVideoClient(
            db_path=seeded_db, force_mock=True, sentiment_threshold=0.8, poll_interval_seconds=0.01
        )
        videos = client.generate_videos_for_high_sentiment_products()

        assert len(videos) == 1
        video = videos[0]
        assert isinstance(video, VideoMetadata)
        assert video.product_id == "P1001"
        assert video.status == VideoStatus.SUCCEEDED
        assert str(video.video_url).startswith("https://mock-cdn.higgsfield.ai/videos/")

        conn = sqlite3.connect(seeded_db)
        row = conn.execute("SELECT product_id, status, provider FROM generated_videos;").fetchone()
        conn.close()
        assert row == ("P1001", "succeeded", "mock")

    def test_no_candidates_below_threshold_yields_zero_videos(self, seeded_db):
        conn = sqlite3.connect(seeded_db)
        conn.execute(
            "CREATE TABLE enriched_reviews (product_id TEXT PRIMARY KEY, sentiment_score REAL, "
            "key_themes TEXT, promo_video_prompt TEXT, provider TEXT, model_used TEXT, "
            "review_count INTEGER, created_at TEXT);"
        )
        conn.execute(
            "INSERT INTO enriched_reviews VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
            ("P1001", 0.5, json.dumps(["ok"]), "A prompt.", "mock", "mock-llm-v1", 2, "2026-01-01T00:00:00"),
        )
        conn.commit()
        conn.close()

        client = HiggsfieldVideoClient(db_path=seeded_db, force_mock=True, sentiment_threshold=0.8)
        videos = client.generate_videos_for_high_sentiment_products()

        assert videos == []


class TestVideoGenerationErrorPaths:
    def test_trigger_video_job_raises_on_live_http_failure(self, tmp_path, sample_product, monkeypatch):
        """With a (fake) API key set, a failing HTTP call should surface as VideoGenerationError."""
        import requests

        client = HiggsfieldVideoClient(
            db_path=tmp_path / "warehouse.db", api_key="fake-key-for-test", force_mock=False
        )

        def _raise_request_exception(*args, **kwargs):
            raise requests.RequestException("simulated network failure")

        monkeypatch.setattr(client._session, "post", _raise_request_exception)

        with pytest.raises(VideoGenerationError):
            client.trigger_video_job(sample_product)
