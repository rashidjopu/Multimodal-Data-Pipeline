"""
src/video_generator.py
==========================

Stage 3 of the Multimodal E-Commerce AI Pipeline: Automated Video Generation.

`HiggsfieldVideoClient` reads `enriched_reviews` for products whose
`sentiment_score` clears a configurable threshold (default 0.8), submits a
5-second promo-video generation job to the Higgsfield AI API, polls until
the job reaches a terminal state, and persists the result to the
`generated_videos` table.

If no `HIGGSFIELD_API_KEY` environment variable is set, the client
transparently falls back to a deterministic mock mode: it simulates the
QUEUED -> PROCESSING -> COMPLETED job lifecycle (with realistic polling
delays) and returns mock CDN video URLs, so the pipeline runs end-to-end
with zero API cost and zero network dependency.

Run directly:
    python -m src.video_generator
    python -m src.video_generator --threshold 0.75
    python -m src.video_generator --force-mock
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import requests
from pydantic import ValidationError

from src.schemas import VideoMetadata, VideoStatus

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/raw_warehouse.db")
DEFAULT_BASE_URL = "https://gateway.pixazo.ai/ai-model-api/v1"
DEFAULT_THRESHOLD = 0.8
DEFAULT_VIDEO_DURATION_SECONDS = 5

# Higgsfield's job-status vocabulary -> our internal VideoStatus enum.
_HIGGSFIELD_STATUS_MAP: Dict[str, VideoStatus] = {
    "QUEUED": VideoStatus.QUEUED,
    "PENDING": VideoStatus.QUEUED,
    "PROCESSING": VideoStatus.PROCESSING,
    "RUNNING": VideoStatus.PROCESSING,
    "COMPLETED": VideoStatus.SUCCEEDED,
    "SUCCEEDED": VideoStatus.SUCCEEDED,
    "FAILED": VideoStatus.FAILED,
    "ERROR": VideoStatus.FAILED,
}


class VideoGenerationError(RuntimeError):
    """Raised when a Higgsfield job fails, times out, or returns an unexpected payload."""


@dataclass
class HighSentimentProduct:
    product_id: str
    product_name: str
    sentiment_score: float
    key_themes: List[str]
    promo_video_prompt: str


class HiggsfieldVideoClient:
    """
    Client for triggering and polling Higgsfield AI promo-video generation
    jobs, with a built-in deterministic mock mode for offline/CI use.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        db_path: Path = DEFAULT_DB_PATH,
        sentiment_threshold: float = DEFAULT_THRESHOLD,
        poll_interval_seconds: float = 5.0,
        poll_timeout_seconds: float = 180.0,
        force_mock: bool = False,
    ):
        self.db_path = Path(db_path)
        self.base_url = base_url.rstrip("/")
        self.sentiment_threshold = sentiment_threshold
        self.poll_interval_seconds = poll_interval_seconds
        self.poll_timeout_seconds = poll_timeout_seconds

        self.api_key = api_key or os.getenv("HIGGSFIELD_API_KEY")
        self.mock_mode = force_mock or not bool(self.api_key)

        if self.mock_mode:
            logger.warning(
                "HIGGSFIELD_API_KEY not set (or --force-mock passed) — HiggsfieldVideoClient "
                "running in MOCK MODE. Video jobs will be simulated and no real network calls "
                "will be made to %s.",
                self.base_url,
            )
        else:
            self._session = requests.Session()
            self._session.headers.update(
                {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
            )

    # ---- database helpers -----------------------------------------------------
    def _get_connection(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _ensure_generated_videos_table(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS generated_videos (
                video_id     TEXT PRIMARY KEY,
                product_id   TEXT NOT NULL,
                video_url    TEXT,
                status       TEXT NOT NULL,
                provider     TEXT NOT NULL,     -- 'higgsfield' or 'mock'
                prompt_used  TEXT NOT NULL,
                created_at   TEXT NOT NULL,
                FOREIGN KEY (product_id) REFERENCES products (product_id)
            );
            """
        )
        conn.commit()

    def get_high_sentiment_products(self, conn: sqlite3.Connection) -> List[HighSentimentProduct]:
        """Query `enriched_reviews` joined with `products` for products above the sentiment threshold."""
        import json as _json

        rows = conn.execute(
            """
            SELECT p.product_id, p.product_name, e.sentiment_score, e.key_themes, e.promo_video_prompt
            FROM enriched_reviews e
            JOIN products p ON p.product_id = e.product_id
            WHERE e.sentiment_score > ?
            ORDER BY e.sentiment_score DESC;
            """,
            (self.sentiment_threshold,),
        ).fetchall()

        return [
            HighSentimentProduct(
                product_id=r[0],
                product_name=r[1],
                sentiment_score=r[2],
                key_themes=_json.loads(r[3]),
                promo_video_prompt=r[4],
            )
            for r in rows
        ]

    def _save_video_metadata(self, conn: sqlite3.Connection, video: VideoMetadata, prompt_used: str) -> None:
        conn.execute(
            """
            INSERT INTO generated_videos (video_id, product_id, video_url, status, provider, prompt_used, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(video_id) DO UPDATE SET
                video_url   = excluded.video_url,
                status      = excluded.status,
                provider    = excluded.provider,
                prompt_used = excluded.prompt_used,
                created_at  = excluded.created_at;
            """,
            (
                video.video_id,
                video.product_id,
                str(video.video_url) if video.video_url else None,
                video.status.value,
                "mock" if self.mock_mode else "higgsfield",
                prompt_used,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()

    # ---- payload construction -------------------------------------------------
    def _build_payload(self, product: HighSentimentProduct) -> dict:
        """
        Build the request payload for Higgsfield's text-to-video endpoint.
        (Swap to POST {base_url}/image-to-video and add a `reference_image_url`
        field here if you have product imagery to condition the video on.)
        """
        return {
            "model": "higgsfield-promo-v1",
            "prompt": product.promo_video_prompt,
            "duration_seconds": DEFAULT_VIDEO_DURATION_SECONDS,
            "aspect_ratio": "9:16",
            "metadata": {
                "product_id": product.product_id,
                "product_name": product.product_name,
                "key_themes": product.key_themes,
                "sentiment_score": product.sentiment_score,
            },
        }

    # ---- job submission ---------------------------------------------------
    def trigger_video_job(self, product: HighSentimentProduct) -> str:
        """Submit a video-generation job and return the provider's job id."""
        payload = self._build_payload(product)

        if self.mock_mode:
            job_id = f"mock-job-{uuid.uuid4().hex[:12]}"
            logger.info("[MOCK] Submitted text-to-video job %s for product %s", job_id, product.product_id)
            return job_id

        url = f"{self.base_url}/text-to-video"
        try:
            response = self._session.post(url, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            raise VideoGenerationError(f"Higgsfield job submission failed for {product.product_id}: {exc}") from exc

        job_id = data.get("job_id") or data.get("id")
        if not job_id:
            raise VideoGenerationError(f"Higgsfield response missing job id for {product.product_id}: {data}")

        logger.info("Submitted text-to-video job %s for product %s", job_id, product.product_id)
        return job_id

    # ---- polling ------------------------------------------------------------
    def poll_job_status(self, job_id: str, product_id: str) -> Dict[str, Optional[str]]:
        """
        Poll the job status every `poll_interval_seconds` until it reaches a
        terminal state (COMPLETED/FAILED) or `poll_timeout_seconds` elapses.
        Returns {"status": VideoStatus, "video_url": Optional[str]}.
        """
        if self.mock_mode:
            return self._poll_mock_job(job_id, product_id)
        return self._poll_real_job(job_id, product_id)

    def _poll_mock_job(self, job_id: str, product_id: str) -> Dict[str, Optional[str]]:
        """Simulate a realistic QUEUED -> PROCESSING -> COMPLETED lifecycle."""
        simulated_states = ["QUEUED", "PROCESSING", "PROCESSING", "COMPLETED"]
        # Cap simulated sleeps so a demo run doesn't take minutes per video.
        simulated_delay = min(self.poll_interval_seconds, 2.0)

        for state in simulated_states:
            logger.info("[MOCK] Job %s (%s) status -> %s", job_id, product_id, state)
            time.sleep(simulated_delay)

        video_url = f"https://mock-cdn.higgsfield.ai/videos/{job_id}.mp4"
        return {"status": "COMPLETED", "video_url": video_url}

    def _poll_real_job(self, job_id: str, product_id: str) -> Dict[str, Optional[str]]:
        url = f"{self.base_url}/jobs/{job_id}"
        elapsed = 0.0

        while elapsed <= self.poll_timeout_seconds:
            try:
                response = self._session.get(url, timeout=30)
                response.raise_for_status()
                data = response.json()
            except requests.RequestException as exc:
                raise VideoGenerationError(f"Higgsfield status poll failed for job {job_id}: {exc}") from exc

            status = str(data.get("status", "")).upper()
            logger.info("Job %s (%s) status -> %s", job_id, product_id, status or "UNKNOWN")

            if status in ("COMPLETED", "SUCCEEDED"):
                return {"status": status, "video_url": data.get("video_url") or data.get("output_url")}
            if status in ("FAILED", "ERROR"):
                return {"status": status, "video_url": None}

            time.sleep(self.poll_interval_seconds)
            elapsed += self.poll_interval_seconds

        raise VideoGenerationError(
            f"Higgsfield job {job_id} for product {product_id} did not complete within "
            f"{self.poll_timeout_seconds}s (last status unknown/timeout)."
        )

    # ---- public API -------------------------------------------------------
    def generate_videos_for_high_sentiment_products(self) -> List[VideoMetadata]:
        """
        Query high-sentiment products, trigger + poll a Higgsfield promo-video
        job for each, validate the result against `VideoMetadata`, and persist
        it to `generated_videos`. Products whose job fails or times out are
        logged and skipped rather than crashing the whole run.
        """
        conn = self._get_connection()
        results: List[VideoMetadata] = []
        try:
            self._ensure_generated_videos_table(conn)
            candidates = self.get_high_sentiment_products(conn)
            logger.info(
                "Found %d product(s) above sentiment threshold %.2f (%s mode).",
                len(candidates),
                self.sentiment_threshold,
                "MOCK" if self.mock_mode else "LIVE",
            )

            for product in candidates:
                try:
                    job_id = self.trigger_video_job(product)
                    poll_result = self.poll_job_status(job_id, product.product_id)
                    internal_status = _HIGGSFIELD_STATUS_MAP.get(
                        str(poll_result["status"]).upper(), VideoStatus.FAILED
                    )

                    video = VideoMetadata(
                        video_id=job_id,
                        product_id=product.product_id,
                        video_url=poll_result["video_url"],
                        status=internal_status,
                    )
                    self._save_video_metadata(conn, video, prompt_used=product.promo_video_prompt)
                    results.append(video)
                    logger.info(
                        "Video ready for %-8s status=%s url=%s",
                        product.product_id,
                        video.status.value,
                        video.video_url,
                    )
                except (VideoGenerationError, ValidationError) as exc:
                    logger.error("Video generation failed for product %s: %s", product.product_id, exc)
                    continue

            logger.info("Video generation complete: %d/%d products succeeded.", len(results), len(candidates))
        finally:
            conn.close()

        return results


# --------------------------------------------------------------------------
# Module-level convenience function (mirrors the class method 1:1)
# --------------------------------------------------------------------------
def generate_promo_videos(
    db_path: Path = DEFAULT_DB_PATH,
    sentiment_threshold: float = DEFAULT_THRESHOLD,
    force_mock: bool = False,
) -> List[VideoMetadata]:
    """Convenience wrapper: build a HiggsfieldVideoClient and run video generation end-to-end."""
    client = HiggsfieldVideoClient(db_path=db_path, sentiment_threshold=sentiment_threshold, force_mock=force_mock)
    return client.generate_videos_for_high_sentiment_products()


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Trigger promo videos for high-sentiment products via Higgsfield AI.")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH, help="SQLite DB path.")
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD, help="Minimum sentiment_score to trigger a video."
    )
    parser.add_argument(
        "--force-mock", action="store_true", help="Skip real API calls and use the deterministic mock client."
    )
    args = parser.parse_args()

    generate_promo_videos(db_path=args.db_path, sentiment_threshold=args.threshold, force_mock=args.force_mock)


if __name__ == "__main__":
    main()
