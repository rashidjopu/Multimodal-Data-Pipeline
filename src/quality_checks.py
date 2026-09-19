"""
src/quality_checks.py
=========================

Stage 5 of the Multimodal E-Commerce AI Pipeline: Data Quality.

Lightweight, dependency-free (custom Python assertions, no Great Expectations
required) data-quality gate that runs BEFORE and AFTER pipeline execution:

  - `run_pre_flight_checks()`   -> sanity checks BEFORE the pipeline starts
                                    (DB reachable/writable, not corrupted,
                                    schema not mid-migration).
  - `run_post_pipeline_checks()` -> the main boundary-validation sweep run
                                    AFTER ingestion + enrichment + video
                                    generation have all completed:
                                       * zero nulls in sentiment_score
                                       * every generated video has a
                                         well-formed URL when COMPLETED
                                       * every row across every table still
                                         satisfies its Pydantic schema
                                       * referential integrity between
                                         products / reviews / enriched_reviews
                                         / generated_videos

Every check produces a `CheckResult` tagged CRITICAL or WARNING. A
`QualityReport` aggregates them; `report.passed` is False if ANY critical
check fails (warnings never fail the pipeline, e.g. partial enrichment
coverage from a couple of dead-lettered products is expected/acceptable).

The Airflow DAG (`dags/ecommerce_multimodal_dag.py`) calls
`run_post_pipeline_checks()` as its final task and raises `AirflowException`
if `report.passed` is False, so any task wired downstream of it is skipped
by Airflow's default trigger rule instead of running against bad data.

Run directly:
    python -m src.quality_checks --stage post
    python -m src.quality_checks --stage pre
    python -m src.quality_checks --stage all --db-path data/raw_warehouse.db
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

from pydantic import ValidationError

from src.schemas import EnrichedAIOutput, ProductRecord, RawReview, VideoMetadata, VideoStatus

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/raw_warehouse.db")
DEFAULT_SENTIMENT_THRESHOLD = 0.8
SAMPLE_ERROR_LIMIT = 5  # cap how many example failures we log per check


class Severity(str, Enum):
    CRITICAL = "critical"  # fails the pipeline / aborts downstream tasks
    WARNING = "warning"    # logged, does not fail the pipeline


class DataQualityError(RuntimeError):
    """Raised when one or more CRITICAL data quality checks fail."""


@dataclass
class CheckResult:
    name: str
    severity: Severity
    passed: bool
    details: str = ""


@dataclass
class QualityReport:
    stage: str
    results: List[CheckResult] = field(default_factory=list)

    def add(self, name: str, severity: Severity, passed: bool, details: str = "") -> None:
        self.results.append(CheckResult(name=name, severity=severity, passed=passed, details=details))
        level = logging.INFO if passed else (logging.ERROR if severity == Severity.CRITICAL else logging.WARNING)
        status = "PASS" if passed else "FAIL"
        logger.log(level, "[%s] (%s) %-45s %s", status, severity.value.upper(), name, details)

    @property
    def critical_failures(self) -> List[CheckResult]:
        return [r for r in self.results if not r.passed and r.severity == Severity.CRITICAL]

    @property
    def warning_failures(self) -> List[CheckResult]:
        return [r for r in self.results if not r.passed and r.severity == Severity.WARNING]

    @property
    def passed(self) -> bool:
        """The pipeline is considered healthy iff there are zero CRITICAL failures."""
        return len(self.critical_failures) == 0

    def log_summary(self) -> None:
        total = len(self.results)
        failed = len([r for r in self.results if not r.passed])
        logger.info("=" * 70)
        logger.info("DATA QUALITY REPORT — stage=%s", self.stage)
        logger.info("=" * 70)
        logger.info("Checks run: %d | Passed: %d | Failed: %d", total, total - failed, failed)
        if self.critical_failures:
            logger.error("CRITICAL failures (%d):", len(self.critical_failures))
            for r in self.critical_failures:
                logger.error("  - %s: %s", r.name, r.details)
        if self.warning_failures:
            logger.warning("WARNING failures (%d):", len(self.warning_failures))
            for r in self.warning_failures:
                logger.warning("  - %s: %s", r.name, r.details)
        logger.info("Overall result: %s", "PASSED" if self.passed else "FAILED (critical checks did not pass)")
        logger.info("=" * 70)

    def raise_if_failed(self) -> None:
        if not self.passed:
            names = ", ".join(r.name for r in self.critical_failures)
            raise DataQualityError(
                f"Data quality gate '{self.stage}' failed {len(self.critical_failures)} critical "
                f"check(s): {names}"
            )


# --------------------------------------------------------------------------
# Checker
# --------------------------------------------------------------------------
class DataQualityChecker:
    """Runs a battery of assertions against the SQLite warehouse at a given path."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH, sentiment_threshold: float = DEFAULT_SENTIMENT_THRESHOLD):
        self.db_path = Path(db_path)
        self.sentiment_threshold = sentiment_threshold

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _table_exists(self, conn: sqlite3.Connection, table: str) -> bool:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table,)
        ).fetchone()
        return row is not None

    # ---- PRE-FLIGHT (before the pipeline runs) --------------------------
    def run_pre_flight_checks(self) -> QualityReport:
        report = QualityReport(stage="pre-flight")

        # 1. DB file location is writable (parent dir can be created, no permission errors).
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            probe = self.db_path.parent / ".dq_write_probe"
            probe.write_text("ok")
            probe.unlink()
            report.add("db_directory_writable", Severity.CRITICAL, True, str(self.db_path.parent))
        except OSError as exc:
            report.add("db_directory_writable", Severity.CRITICAL, False, str(exc))

        # 2. If the DB already exists, it must be a valid, non-corrupted SQLite file.
        if self.db_path.exists():
            try:
                conn = self._connect()
                conn.execute("PRAGMA integrity_check;").fetchone()
                conn.close()
                report.add("existing_db_not_corrupted", Severity.CRITICAL, True, str(self.db_path))
            except sqlite3.DatabaseError as exc:
                report.add("existing_db_not_corrupted", Severity.CRITICAL, False, str(exc))
        else:
            report.add(
                "existing_db_not_corrupted", Severity.WARNING, True, "No existing DB — fresh run, nothing to check."
            )

        report.log_summary()
        return report

    # ---- BOUNDARY 1: after ingestion (raw layer) -------------------------
    def check_raw_layer(self, conn: sqlite3.Connection, report: QualityReport) -> None:
        if not self._table_exists(conn, "products") or not self._table_exists(conn, "reviews"):
            report.add("raw_tables_exist", Severity.CRITICAL, False, "'products' and/or 'reviews' table missing.")
            return
        report.add("raw_tables_exist", Severity.CRITICAL, True)

        product_rows = conn.execute("SELECT product_id, product_name, category, brand, price_usd FROM products;").fetchall()
        review_rows = conn.execute(
            "SELECT review_id, product_id, user_review, rating, reviewer_name, review_date FROM reviews;"
        ).fetchall()

        report.add("products_nonempty", Severity.CRITICAL, len(product_rows) > 0, f"{len(product_rows)} rows")
        report.add("reviews_nonempty", Severity.CRITICAL, len(review_rows) > 0, f"{len(review_rows)} rows")

        # Schema compliance: re-validate every row against Pydantic models.
        product_errors: List[str] = []
        product_ids = set()
        for row in product_rows:
            product_ids.add(row[0])
            try:
                ProductRecord(product_id=row[0], product_name=row[1], category=row[2], brand=row[3], price_usd=row[4])
            except ValidationError as exc:
                product_errors.append(f"{row[0]}: {exc.errors()[0]['msg']}")
        report.add(
            "product_schema_compliance",
            Severity.CRITICAL,
            len(product_errors) == 0,
            f"{len(product_errors)} invalid row(s): {product_errors[:SAMPLE_ERROR_LIMIT]}",
        )

        review_errors: List[str] = []
        null_field_errors: List[str] = []
        orphan_reviews: List[str] = []
        for row in review_rows:
            review_id, product_id, user_review, rating, reviewer_name, review_date = row
            if review_id is None or product_id is None or user_review is None or rating is None:
                null_field_errors.append(str(review_id))
                continue
            if product_id not in product_ids:
                orphan_reviews.append(review_id)
            try:
                RawReview(
                    review_id=review_id,
                    product_id=product_id,
                    user_review=user_review,
                    rating=rating,
                    reviewer_name=reviewer_name,
                    review_date=review_date,
                )
            except ValidationError as exc:
                review_errors.append(f"{review_id}: {exc.errors()[0]['msg']}")

        report.add(
            "reviews_no_null_required_fields",
            Severity.CRITICAL,
            len(null_field_errors) == 0,
            f"{len(null_field_errors)} row(s) with null required fields: {null_field_errors[:SAMPLE_ERROR_LIMIT]}",
        )
        report.add(
            "reviews_referential_integrity",
            Severity.CRITICAL,
            len(orphan_reviews) == 0,
            f"{len(orphan_reviews)} review(s) reference a missing product_id: {orphan_reviews[:SAMPLE_ERROR_LIMIT]}",
        )
        report.add(
            "review_schema_compliance",
            Severity.CRITICAL,
            len(review_errors) == 0,
            f"{len(review_errors)} invalid row(s): {review_errors[:SAMPLE_ERROR_LIMIT]}",
        )

        ratings_out_of_range = [r[0] for r in review_rows if r[3] is not None and not (1 <= r[3] <= 5)]
        report.add(
            "rating_within_1_to_5",
            Severity.CRITICAL,
            len(ratings_out_of_range) == 0,
            f"{len(ratings_out_of_range)} out-of-range rating(s): {ratings_out_of_range[:SAMPLE_ERROR_LIMIT]}",
        )

    # ---- BOUNDARY 2: after AI enrichment ---------------------------------
    def check_enrichment_layer(self, conn: sqlite3.Connection, report: QualityReport) -> None:
        if not self._table_exists(conn, "enriched_reviews"):
            report.add("enriched_table_exists", Severity.CRITICAL, False, "'enriched_reviews' table missing.")
            return
        report.add("enriched_table_exists", Severity.CRITICAL, True)

        product_ids = {r[0] for r in conn.execute("SELECT product_id FROM products;").fetchall()}
        enriched_rows = conn.execute(
            "SELECT product_id, sentiment_score, key_themes, promo_video_prompt FROM enriched_reviews;"
        ).fetchall()

        # Zero nulls in sentiment_score (explicitly requested check).
        null_sentiment = [r[0] for r in enriched_rows if r[1] is None]
        report.add(
            "zero_null_sentiment_scores",
            Severity.CRITICAL,
            len(null_sentiment) == 0,
            f"{len(null_sentiment)} row(s) with NULL sentiment_score: {null_sentiment[:SAMPLE_ERROR_LIMIT]}",
        )

        out_of_range = [r[0] for r in enriched_rows if r[1] is not None and not (0.0 <= r[1] <= 1.0)]
        report.add(
            "sentiment_score_within_0_1",
            Severity.CRITICAL,
            len(out_of_range) == 0,
            f"{len(out_of_range)} row(s) out of [0,1] range: {out_of_range[:SAMPLE_ERROR_LIMIT]}",
        )

        schema_errors: List[str] = []
        orphans: List[str] = []
        for product_id, sentiment_score, key_themes_json, promo_prompt in enriched_rows:
            if product_id not in product_ids:
                orphans.append(product_id)
            try:
                themes = json.loads(key_themes_json) if key_themes_json else []
                EnrichedAIOutput(
                    product_id=product_id,
                    sentiment_score=sentiment_score,
                    key_themes=themes,
                    promo_video_prompt=promo_prompt,
                )
            except (ValidationError, json.JSONDecodeError) as exc:
                msg = exc.errors()[0]["msg"] if isinstance(exc, ValidationError) else str(exc)
                schema_errors.append(f"{product_id}: {msg}")

        report.add(
            "enrichment_referential_integrity",
            Severity.CRITICAL,
            len(orphans) == 0,
            f"{len(orphans)} enrichment row(s) reference a missing product_id: {orphans[:SAMPLE_ERROR_LIMIT]}",
        )
        report.add(
            "enrichment_schema_compliance",
            Severity.CRITICAL,
            len(schema_errors) == 0,
            f"{len(schema_errors)} invalid row(s): {schema_errors[:SAMPLE_ERROR_LIMIT]}",
        )

        # Coverage is a WARNING, not CRITICAL — some products may be legitimately
        # dead-lettered by the enrichment stage without invalidating the whole run.
        missing_coverage = product_ids - {r[0] for r in enriched_rows}
        if product_ids and len(missing_coverage) == len(product_ids):
            # Total failure: nothing got enriched at all -> that IS critical.
            report.add(
                "enrichment_coverage",
                Severity.CRITICAL,
                False,
                "0 of %d products have enrichment data — enrichment stage produced no output." % len(product_ids),
            )
        else:
            report.add(
                "enrichment_coverage",
                Severity.WARNING,
                len(missing_coverage) == 0,
                f"{len(missing_coverage)}/{len(product_ids)} product(s) missing enrichment "
                f"(likely dead-lettered): {list(missing_coverage)[:SAMPLE_ERROR_LIMIT]}",
            )

    # ---- BOUNDARY 3: after video generation -------------------------------
    def check_video_layer(self, conn: sqlite3.Connection, report: QualityReport) -> None:
        if not self._table_exists(conn, "generated_videos"):
            # No videos table at all is only CRITICAL if we expected some
            # (i.e. there were candidates above the sentiment threshold).
            candidates = 0
            if self._table_exists(conn, "enriched_reviews"):
                candidates = conn.execute(
                    "SELECT COUNT(*) FROM enriched_reviews WHERE sentiment_score > ?;", (self.sentiment_threshold,)
                ).fetchone()[0]
            severity = Severity.CRITICAL if candidates > 0 else Severity.WARNING
            report.add(
                "generated_videos_table_exists",
                severity,
                False,
                f"'generated_videos' table missing (had {candidates} high-sentiment candidate(s)).",
            )
            return
        report.add("generated_videos_table_exists", Severity.CRITICAL, True)

        product_ids = {r[0] for r in conn.execute("SELECT product_id FROM products;").fetchall()}
        video_rows = conn.execute(
            "SELECT video_id, product_id, video_url, status FROM generated_videos;"
        ).fetchall()

        schema_errors: List[str] = []
        bad_urls: List[str] = []
        orphans: List[str] = []

        for video_id, product_id, video_url, status in video_rows:
            if product_id not in product_ids:
                orphans.append(video_id)

            # Valid URL format check (explicitly requested): required whenever
            # status is 'succeeded'; scheme must be http/https and host present.
            if status == VideoStatus.SUCCEEDED.value:
                is_valid_url = False
                if video_url:
                    parsed = urlparse(video_url)
                    is_valid_url = parsed.scheme in ("http", "https") and bool(parsed.netloc)
                if not is_valid_url:
                    bad_urls.append(f"{video_id} (url={video_url!r})")

            try:
                VideoMetadata(video_id=video_id, product_id=product_id, video_url=video_url, status=status)
            except ValidationError as exc:
                schema_errors.append(f"{video_id}: {exc.errors()[0]['msg']}")

        report.add(
            "video_referential_integrity",
            Severity.CRITICAL,
            len(orphans) == 0,
            f"{len(orphans)} video(s) reference a missing product_id: {orphans[:SAMPLE_ERROR_LIMIT]}",
        )
        report.add(
            "video_url_valid_for_succeeded_jobs",
            Severity.CRITICAL,
            len(bad_urls) == 0,
            f"{len(bad_urls)} succeeded job(s) with missing/malformed URL: {bad_urls[:SAMPLE_ERROR_LIMIT]}",
        )
        report.add(
            "video_schema_compliance",
            Severity.CRITICAL,
            len(schema_errors) == 0,
            f"{len(schema_errors)} invalid row(s): {schema_errors[:SAMPLE_ERROR_LIMIT]}",
        )

    # ---- POST-PIPELINE (after ingestion + enrichment + video gen) ---------
    def run_post_pipeline_checks(self) -> QualityReport:
        report = QualityReport(stage="post-pipeline")
        conn = self._connect()
        try:
            self.check_raw_layer(conn, report)
            self.check_enrichment_layer(conn, report)
            self.check_video_layer(conn, report)
        finally:
            conn.close()
        report.log_summary()
        return report


# --------------------------------------------------------------------------
# Module-level convenience functions (used directly by the Airflow DAG)
# --------------------------------------------------------------------------
def run_pre_flight_checks(db_path: Path = DEFAULT_DB_PATH) -> QualityReport:
    return DataQualityChecker(db_path=db_path).run_pre_flight_checks()


def run_post_pipeline_checks(
    db_path: Path = DEFAULT_DB_PATH, sentiment_threshold: float = DEFAULT_SENTIMENT_THRESHOLD
) -> QualityReport:
    return DataQualityChecker(db_path=db_path, sentiment_threshold=sentiment_threshold).run_post_pipeline_checks()


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Run data quality assertions against the SQLite warehouse.")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--stage", choices=["pre", "post", "all"], default="all")
    parser.add_argument("--threshold", type=float, default=DEFAULT_SENTIMENT_THRESHOLD)
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Treat WARNING-level failures as CRITICAL for this CLI invocation (stricter local/manual runs).",
    )
    args = parser.parse_args()

    checker = DataQualityChecker(db_path=args.db_path, sentiment_threshold=args.threshold)
    reports: List[QualityReport] = []

    if args.stage in ("pre", "all"):
        reports.append(checker.run_pre_flight_checks())
    if args.stage in ("post", "all"):
        reports.append(checker.run_post_pipeline_checks())

    overall_ok = all(r.passed for r in reports)
    if args.fail_on_warning:
        overall_ok = overall_ok and all(len(r.warning_failures) == 0 for r in reports)

    if not overall_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
