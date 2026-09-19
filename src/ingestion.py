"""
src/ingestion.py
====================

Stage 1 of the Multimodal E-Commerce AI Pipeline: Ingestion.

Generates a realistic mock dataset of 4 distinct products and 20+ diverse
customer reviews, validates every record against the Pydantic contracts in
`src/schemas.py`, and loads the validated data into a SQLite database at
`data/raw_warehouse.db`.

Run directly:
    python -m src.ingestion
    python -m src.ingestion --reset            # drop & recreate tables first
    python -m src.ingestion --db-path some.db  # write to a custom location

Every record is validated *before* it touches the database, so a malformed
mock record fails fast with a clear Pydantic error instead of silently
corrupting the warehouse.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

from pydantic import ValidationError

from src.schemas import ProductRecord, RawReview

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/raw_warehouse.db")


# --------------------------------------------------------------------------
# Mock data definitions
# --------------------------------------------------------------------------
def _mock_product_rows() -> List[dict]:
    """Raw dicts for 4 distinct products spanning different categories."""
    return [
        {
            "product_id": "P1001",
            "product_name": "AeroFit Pro Wireless Earbuds",
            "category": "Electronics",
            "brand": "AeroSound",
            "price_usd": 79.99,
        },
        {
            "product_id": "P1002",
            "product_name": "Summit Trail 40L Hiking Backpack",
            "category": "Outdoor & Sports",
            "brand": "Summit Gear",
            "price_usd": 129.99,
        },
        {
            "product_id": "P1003",
            "product_name": "LumaBrew Smart Coffee Maker",
            "category": "Home & Kitchen",
            "brand": "LumaHome",
            "price_usd": 149.99,
        },
        {
            "product_id": "P1004",
            "product_name": "FlexFoam Ergonomic Office Chair",
            "category": "Furniture",
            "brand": "FlexFoam",
            "price_usd": 249.99,
        },
    ]


def _mock_review_rows() -> List[dict]:
    """
    Raw dicts for 24 diverse reviews across the 4 products above (6 each).
    Deliberately mixes rating levels, review lengths, tone, and typos/slang
    to stress-test the downstream LLM sentiment/theme extraction stage.
    """
    base_date = datetime(2026, 6, 1)

    reviews = [
        # ---- P1001: AeroFit Pro Wireless Earbuds ----------------------
        {
            "review_id": "R0001",
            "product_id": "P1001",
            "user_review": (
                "These earbuds are incredible for the price. Battery lasts "
                "all day, the noise cancellation actually works on my commute, "
                "and the case fits perfectly in my pocket."
            ),
            "rating": 5,
            "reviewer_name": "Jamie K.",
            "review_date": base_date + timedelta(days=1),
        },
        {
            "review_id": "R0002",
            "product_id": "P1001",
            "user_review": "Sound quality is great but the right earbud keeps disconnecting randomly. Kind of annoying during calls.",
            "rating": 3,
            "reviewer_name": "Priya S.",
            "review_date": base_date + timedelta(days=3),
        },
        {
            "review_id": "R0003",
            "product_id": "P1001",
            "user_review": "Broke after two weeks. Charging case stopped holding a charge. Very disappointed, expected better quality control.",
            "rating": 1,
            "reviewer_name": "Marcus T.",
            "review_date": base_date + timedelta(days=5),
        },
        {
            "review_id": "R0004",
            "product_id": "P1001",
            "user_review": "Best earbuds I've owned. The bass is punchy without being muddy and they never fall out during workouts.",
            "rating": 5,
            "reviewer_name": "Elena R.",
            "review_date": base_date + timedelta(days=6),
        },
        {
            "review_id": "R0005",
            "product_id": "P1001",
            "user_review": "decent, does the job. nothing special but nothing wrong either. comfortable fit tho",
            "rating": 4,
            "reviewer_name": "dev_ops_dan",
            "review_date": base_date + timedelta(days=9),
        },
        {
            "review_id": "R0006",
            "product_id": "P1001",
            "user_review": "Connectivity range is weak, loses signal if my phone is in another room. Comfortable otherwise.",
            "rating": 2,
            "reviewer_name": "Sofia M.",
            "review_date": base_date + timedelta(days=11),
        },
        # ---- P1002: Summit Trail 40L Hiking Backpack -------------------
        {
            "review_id": "R0007",
            "product_id": "P1002",
            "user_review": (
                "Took this on a 5-day trek through the Rockies and it held up beautifully. "
                "The hip belt distributes weight really well and the rain cover is a nice touch."
            ),
            "rating": 5,
            "reviewer_name": "Outdoor_Hank",
            "review_date": base_date + timedelta(days=2),
        },
        {
            "review_id": "R0008",
            "product_id": "P1002",
            "user_review": "Zippers feel cheap and one already snagged after a month of light use. Storage layout is great though.",
            "rating": 3,
            "reviewer_name": "Grace L.",
            "review_date": base_date + timedelta(days=4),
        },
        {
            "review_id": "R0009",
            "product_id": "P1002",
            "user_review": "Exactly what I needed for backpacking season. Tons of compartments, comfortable straps, love the color.",
            "rating": 5,
            "reviewer_name": "Tomas B.",
            "review_date": base_date + timedelta(days=7),
        },
        {
            "review_id": "R0010",
            "product_id": "P1002",
            "user_review": "Way too bulky for carry-on travel like the listing implied. Good for actual hiking, bad for flights.",
            "rating": 2,
            "reviewer_name": "Nadia F.",
            "review_date": base_date + timedelta(days=8),
        },
        {
            "review_id": "R0011",
            "product_id": "P1002",
            "user_review": "Solid, dependable pack. Nothing fancy but everything works as expected and it's held up over 6 months of weekend trips.",
            "rating": 4,
            "reviewer_name": "Chris P.",
            "review_date": base_date + timedelta(days=12),
        },
        {
            "review_id": "R0012",
            "product_id": "P1002",
            "user_review": "Stitching came undone at the bottom seam within 3 uses. Returning it. Really wanted to like this one.",
            "rating": 1,
            "reviewer_name": "Wendy A.",
            "review_date": base_date + timedelta(days=14),
        },
        # ---- P1003: LumaBrew Smart Coffee Maker -------------------------
        {
            "review_id": "R0013",
            "product_id": "P1003",
            "user_review": (
                "The app scheduling feature is a game changer, wake up to fresh coffee every morning. "
                "Brew is consistently smooth, never bitter."
            ),
            "rating": 5,
            "reviewer_name": "CoffeeAddict92",
            "review_date": base_date + timedelta(days=1),
        },
        {
            "review_id": "R0014",
            "product_id": "P1003",
            "user_review": "Wifi setup was a nightmare, took me an hour and three factory resets. Coffee tastes fine once it's working.",
            "rating": 3,
            "reviewer_name": "Harold V.",
            "review_date": base_date + timedelta(days=3),
        },
        {
            "review_id": "R0015",
            "product_id": "P1003",
            "user_review": "Leaks from the base after every brew cycle. Had to return it. Such a shame because the design is beautiful.",
            "rating": 1,
            "reviewer_name": "Ines D.",
            "review_date": base_date + timedelta(days=6),
        },
        {
            "review_id": "R0016",
            "product_id": "P1003",
            "user_review": "Love love LOVE this machine! Sleek design, quiet operation, and the built-in grinder saves me so much time.",
            "rating": 5,
            "reviewer_name": "Beatrice O.",
            "review_date": base_date + timedelta(days=9),
        },
        {
            "review_id": "R0017",
            "product_id": "P1003",
            "user_review": "It's fine. Makes coffee. The app crashes occasionally but a restart fixes it.",
            "rating": 3,
            "reviewer_name": "quiet_reviewer",
            "review_date": base_date + timedelta(days=10),
        },
        {
            "review_id": "R0018",
            "product_id": "P1003",
            "user_review": "Upgraded from a basic drip machine and the difference is night and day. Worth every penny for the customization options.",
            "rating": 5,
            "reviewer_name": "Marcus T.",
            "review_date": base_date + timedelta(days=13),
        },
        # ---- P1004: FlexFoam Ergonomic Office Chair ---------------------
        {
            "review_id": "R0019",
            "product_id": "P1004",
            "user_review": (
                "My back pain has genuinely improved since switching to this chair. Lumbar support is adjustable "
                "and the mesh keeps me cool during long work sessions."
            ),
            "rating": 5,
            "reviewer_name": "RemoteWorkerRae",
            "review_date": base_date + timedelta(days=2),
        },
        {
            "review_id": "R0020",
            "product_id": "P1004",
            "user_review": "Assembly instructions were confusing and missing a bolt. Chair itself is comfortable once built.",
            "rating": 3,
            "reviewer_name": "Peter Q.",
            "review_date": base_date + timedelta(days=5),
        },
        {
            "review_id": "R0021",
            "product_id": "P1004",
            "user_review": "Armrests wobble constantly and squeak when I lean on them. Returning for a refund.",
            "rating": 2,
            "reviewer_name": "Diane H.",
            "review_date": base_date + timedelta(days=7),
        },
        {
            "review_id": "R0022",
            "product_id": "P1004",
            "user_review": "Best purchase for my home office this year. Reclines smoothly, headrest is a nice bonus, very sturdy build.",
            "rating": 5,
            "reviewer_name": "Oliver N.",
            "review_date": base_date + timedelta(days=10),
        },
        {
            "review_id": "R0023",
            "product_id": "P1004",
            "user_review": "Seat cushion flattened out within a month of daily use. Expected more durability at this price point.",
            "rating": 2,
            "reviewer_name": "Fatima Z.",
            "review_date": base_date + timedelta(days=12),
        },
        {
            "review_id": "R0024",
            "product_id": "P1004",
            "user_review": "great chair, comfy, easy to adjust, exactly as described. wish it came in more colors!",
            "rating": 4,
            "reviewer_name": "lena_writes",
            "review_date": base_date + timedelta(days=15),
        },
    ]
    return reviews


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def build_validated_products() -> List[ProductRecord]:
    """Validate raw product dicts against ProductRecord; fail fast on error."""
    validated: List[ProductRecord] = []
    for row in _mock_product_rows():
        try:
            validated.append(ProductRecord(**row))
        except ValidationError as exc:
            logger.error("Product record failed validation: %s\n%s", row.get("product_id"), exc)
            raise
    logger.info("Validated %d product records against ProductRecord schema.", len(validated))
    return validated


def build_validated_reviews() -> List[RawReview]:
    """Validate raw review dicts against RawReview; fail fast on error."""
    validated: List[RawReview] = []
    for row in _mock_review_rows():
        try:
            validated.append(RawReview(**row))
        except ValidationError as exc:
            logger.error("Review record failed validation: %s\n%s", row.get("review_id"), exc)
            raise
    logger.info("Validated %d review records against RawReview schema.", len(validated))
    return validated


# --------------------------------------------------------------------------
# SQLite persistence
# --------------------------------------------------------------------------
def get_connection(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_schema(conn: sqlite3.Connection, reset: bool = False) -> None:
    """Create the `products` and `reviews` tables (optionally dropping first)."""
    if reset:
        conn.execute("DROP TABLE IF EXISTS reviews;")
        conn.execute("DROP TABLE IF EXISTS products;")
        logger.info("Existing tables dropped (--reset).")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS products (
            product_id   TEXT PRIMARY KEY,
            product_name TEXT NOT NULL,
            category     TEXT NOT NULL,
            brand        TEXT NOT NULL,
            price_usd    REAL NOT NULL CHECK (price_usd > 0)
        );
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS reviews (
            review_id     TEXT PRIMARY KEY,
            product_id    TEXT NOT NULL,
            user_review   TEXT NOT NULL,
            rating        INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
            reviewer_name TEXT,
            review_date   TEXT,
            FOREIGN KEY (product_id) REFERENCES products (product_id)
        );
        """
    )
    conn.commit()
    logger.info("Schema ready: tables 'products' and 'reviews'.")


def load_products(conn: sqlite3.Connection, products: List[ProductRecord]) -> None:
    rows = [(p.product_id, p.product_name, p.category, p.brand, p.price_usd) for p in products]
    conn.executemany(
        """
        INSERT INTO products (product_id, product_name, category, brand, price_usd)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(product_id) DO UPDATE SET
            product_name = excluded.product_name,
            category     = excluded.category,
            brand        = excluded.brand,
            price_usd    = excluded.price_usd;
        """,
        rows,
    )
    conn.commit()
    logger.info("Upserted %d rows into 'products'.", len(rows))


def load_reviews(conn: sqlite3.Connection, reviews: List[RawReview]) -> None:
    rows = [
        (
            r.review_id,
            r.product_id,
            r.user_review,
            r.rating,
            r.reviewer_name,
            r.review_date.isoformat() if r.review_date else None,
        )
        for r in reviews
    ]
    conn.executemany(
        """
        INSERT INTO reviews (review_id, product_id, user_review, rating, reviewer_name, review_date)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(review_id) DO UPDATE SET
            product_id    = excluded.product_id,
            user_review   = excluded.user_review,
            rating        = excluded.rating,
            reviewer_name = excluded.reviewer_name,
            review_date   = excluded.review_date;
        """,
        rows,
    )
    conn.commit()
    logger.info("Upserted %d rows into 'reviews'.", len(rows))


# --------------------------------------------------------------------------
# Summary / verification
# --------------------------------------------------------------------------
def print_summary(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM products;")
    n_products = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM reviews;")
    n_reviews = cur.fetchone()[0]

    cur.execute(
        """
        SELECT p.product_name, COUNT(r.review_id), ROUND(AVG(r.rating), 2)
        FROM products p
        LEFT JOIN reviews r ON r.product_id = p.product_id
        GROUP BY p.product_id
        ORDER BY p.product_id;
        """
    )
    breakdown = cur.fetchall()

    logger.info("=" * 60)
    logger.info("INGESTION SUMMARY")
    logger.info("=" * 60)
    logger.info("Total products : %d", n_products)
    logger.info("Total reviews  : %d", n_reviews)
    logger.info("-" * 60)
    for name, count, avg_rating in breakdown:
        logger.info("%-38s reviews=%-3d avg_rating=%s", name, count, avg_rating)
    logger.info("=" * 60)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def run(db_path: Path = DEFAULT_DB_PATH, reset: bool = False) -> None:
    logger.info("Starting ingestion -> target DB: %s", db_path)

    products = build_validated_products()
    reviews = build_validated_reviews()

    if len(reviews) < 20:
        raise ValueError(f"Expected at least 20 mock reviews, got {len(reviews)}.")
    if len({p.product_id for p in products}) < 4:
        raise ValueError("Expected at least 4 distinct products.")

    conn = get_connection(db_path)
    try:
        init_schema(conn, reset=reset)
        load_products(conn, products)
        load_reviews(conn, reviews)
        print_summary(conn)
    finally:
        conn.close()

    logger.info("Ingestion complete. Data written to %s", db_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest mock e-commerce product & review data into SQLite.")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"Path to the SQLite database file (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate the 'products' and 'reviews' tables before loading.",
    )
    args = parser.parse_args()
    run(db_path=args.db_path, reset=args.reset)


if __name__ == "__main__":
    main()
