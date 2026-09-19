"""
app.py
==========

Streamlit dashboard for the Multimodal E-Commerce AI Pipeline.

Reads directly from the SQLite warehouse (`data/raw_warehouse.db`) populated
by src/ingestion.py -> src/ai_pipeline.py -> src/video_generator.py, and
shows:
  1. Headline metrics: total reviews processed, average sentiment, videos generated.
  2. A data table of enriched product analytics (sentiment, themes, promo prompt).
  3. A video gallery pairing each generated promo video with its source
     product metadata — with a clear "mock" badge for videos produced by
     MockHiggsfieldClient (no HIGGSFIELD_API_KEY configured) since those
     point at a fake CDN and won't actually play.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import streamlit as st

DEFAULT_DB_PATH = Path("data/raw_warehouse.db")

st.set_page_config(
    page_title="E-Commerce AI Pipeline Dashboard",
    page_icon="🛍️",
    layout="wide",
)

# --------------------------------------------------------------------------
# Minimal styling polish (kept subtle — Streamlit's own component styling
# does most of the work; this just tightens spacing and adds a video-card
# frame that Streamlit doesn't provide natively).
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; }
        .video-card {
            border: 1px solid rgba(120, 120, 120, 0.25);
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 1rem;
            height: 100%;
        }
        .mock-badge {
            display: inline-block;
            background: rgba(255, 193, 7, 0.18);
            color: #b8860b;
            border: 1px solid rgba(184, 134, 11, 0.4);
            border-radius: 6px;
            padding: 0.05rem 0.5rem;
            font-size: 0.75rem;
            font-weight: 600;
            margin-left: 0.4rem;
        }
        .live-badge {
            display: inline-block;
            background: rgba(46, 160, 67, 0.15);
            color: #1a7f37;
            border: 1px solid rgba(26, 127, 55, 0.4);
            border-radius: 6px;
            padding: 0.05rem 0.5rem;
            font-size: 0.75rem;
            font-weight: 600;
            margin-left: 0.4rem;
        }
        .theme-chip {
            display: inline-block;
            background: rgba(99, 110, 250, 0.12);
            border-radius: 999px;
            padding: 0.1rem 0.6rem;
            margin: 0.1rem;
            font-size: 0.8rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Data access
# --------------------------------------------------------------------------
def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return (
        conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table,)).fetchone()
        is not None
    )


@st.cache_data(ttl=30, show_spinner=False)
def load_data(db_path_str: str, _cache_bust: int = 0):
    """
    Load all four pipeline tables into DataFrames. `_cache_bust` lets the
    sidebar refresh button force a re-read without changing the DB path.
    Returns a dict of empty-safe DataFrames plus a `db_missing` flag.
    """
    db_path = Path(db_path_str)
    empty = {
        "products": pd.DataFrame(),
        "reviews": pd.DataFrame(),
        "enriched": pd.DataFrame(),
        "videos": pd.DataFrame(),
    }
    if not db_path.exists():
        return {**empty, "db_missing": True}

    conn = sqlite3.connect(db_path)
    try:
        products = (
            pd.read_sql("SELECT * FROM products;", conn) if _table_exists(conn, "products") else pd.DataFrame()
        )
        reviews = pd.read_sql("SELECT * FROM reviews;", conn) if _table_exists(conn, "reviews") else pd.DataFrame()
        enriched = (
            pd.read_sql("SELECT * FROM enriched_reviews;", conn)
            if _table_exists(conn, "enriched_reviews")
            else pd.DataFrame()
        )
        videos = (
            pd.read_sql("SELECT * FROM generated_videos;", conn)
            if _table_exists(conn, "generated_videos")
            else pd.DataFrame()
        )
    finally:
        conn.close()

    if not enriched.empty and "key_themes" in enriched.columns:
        enriched["key_themes"] = enriched["key_themes"].apply(lambda v: json.loads(v) if v else [])

    return {"products": products, "reviews": reviews, "enriched": enriched, "videos": videos, "db_missing": False}


def is_mock_url(url: str) -> bool:
    if not url:
        return True
    return "mock-cdn" in urlparse(url).netloc


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.title("🛍️ Pipeline Dashboard")
db_path_input = st.sidebar.text_input("SQLite DB path", value=str(DEFAULT_DB_PATH))

if "cache_bust" not in st.session_state:
    st.session_state.cache_bust = 0
if st.sidebar.button("🔄 Refresh data", use_container_width=True):
    st.session_state.cache_bust += 1

st.sidebar.caption(
    "Reads directly from the pipeline's SQLite warehouse. Run `bash run.sh` "
    "(or the individual `python -m src.*` stages) to populate or update it."
)

data = load_data(db_path_input, st.session_state.cache_bust)

if data["db_missing"]:
    st.warning(
        f"No database found at `{db_path_input}`. Run the pipeline first, e.g.:\n\n"
        "```bash\npython -m src.ingestion\npython -m src.ai_pipeline --force-mock\n"
        "python -m src.video_generator --force-mock\n```"
    )
    st.stop()

products_df = data["products"]
reviews_df = data["reviews"]
enriched_df = data["enriched"]
videos_df = data["videos"]

# --------------------------------------------------------------------------
# Header + headline metrics
# --------------------------------------------------------------------------
st.title("Multimodal E-Commerce AI Pipeline")
st.caption("Ingestion → AI Enrichment → Video Generation → Data Quality, end to end.")

total_reviews = len(reviews_df)
avg_sentiment = enriched_df["sentiment_score"].mean() if not enriched_df.empty else None
videos_generated = len(videos_df)
videos_succeeded = int((videos_df["status"] == "succeeded").sum()) if not videos_df.empty else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Reviews Processed", f"{total_reviews:,}")
col2.metric("Average Sentiment", f"{avg_sentiment:.2f}" if avg_sentiment is not None else "—")
col3.metric("Videos Generated", f"{videos_generated:,}")
col4.metric("Videos Succeeded", f"{videos_succeeded:,}")

st.divider()

# --------------------------------------------------------------------------
# Enriched product analytics table
# --------------------------------------------------------------------------
st.subheader("Enriched Product Analytics")

if enriched_df.empty or products_df.empty:
    st.info("No enrichment data yet — run `python -m src.ai_pipeline` to populate this table.")
else:
    review_counts = reviews_df.groupby("product_id").size().rename("review_count") if not reviews_df.empty else None

    table = enriched_df.merge(products_df, on="product_id", how="left")
    if review_counts is not None:
        table = table.merge(review_counts, on="product_id", how="left")

    display_cols = {
        "product_id": "Product ID",
        "product_name": "Product",
        "category": "Category",
        "sentiment_score": "Sentiment",
        "key_themes": "Key Themes",
        "review_count": "# Reviews",
        "promo_video_prompt": "Promo Video Concept",
    }
    available_cols = [c for c in display_cols if c in table.columns]
    display_df = table[available_cols].rename(columns=display_cols)
    if "Key Themes" in display_df.columns:
        display_df["Key Themes"] = display_df["Key Themes"].apply(
            lambda themes: ", ".join(themes) if isinstance(themes, list) else themes
        )
    display_df = display_df.sort_values("Sentiment", ascending=False) if "Sentiment" in display_df.columns else display_df

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Sentiment": st.column_config.ProgressColumn(
                "Sentiment", min_value=0.0, max_value=1.0, format="%.2f"
            ),
        },
    )

st.divider()

# --------------------------------------------------------------------------
# Video gallery
# --------------------------------------------------------------------------
st.subheader("Promo Video Gallery")

if videos_df.empty:
    st.info(
        "No promo videos yet — either enrichment hasn't run, or no product cleared the sentiment "
        "threshold. Run `python -m src.video_generator` after enrichment."
    )
else:
    gallery = videos_df.merge(products_df, on="product_id", how="left")
    if "enriched" in data and not enriched_df.empty:
        gallery = gallery.merge(
            enriched_df[["product_id", "sentiment_score", "key_themes"]], on="product_id", how="left"
        )

    cols_per_row = 3
    rows = [gallery.iloc[i : i + cols_per_row] for i in range(0, len(gallery), cols_per_row)]

    for row in rows:
        cols = st.columns(cols_per_row)
        for col, (_, video) in zip(cols, row.iterrows()):
            with col:
                st.markdown('<div class="video-card">', unsafe_allow_html=True)

                mock = is_mock_url(video.get("video_url"))
                badge = '<span class="mock-badge">MOCK</span>' if mock else '<span class="live-badge">LIVE</span>'
                product_name = video.get("product_name") or video.get("product_id")
                st.markdown(f"**{product_name}** {badge}", unsafe_allow_html=True)
                st.caption(f"Status: `{video.get('status')}` · Product ID: `{video.get('product_id')}`")

                if video.get("status") == "succeeded" and video.get("video_url"):
                    if mock:
                        st.info(
                            "🎬 Simulated video (mock mode — no `HIGGSFIELD_API_KEY` configured). "
                            f"Mock asset URL: `{video.get('video_url')}`"
                        )
                    else:
                        st.video(video["video_url"])
                elif video.get("status") == "failed":
                    st.error("Video generation failed for this product.")
                else:
                    st.warning(f"Video not yet completed (status: {video.get('status')}).")

                sentiment = video.get("sentiment_score")
                if pd.notna(sentiment):
                    st.progress(min(max(float(sentiment), 0.0), 1.0), text=f"Sentiment: {sentiment:.2f}")

                themes = video.get("key_themes")
                if isinstance(themes, str):
                    try:
                        themes = json.loads(themes)
                    except json.JSONDecodeError:
                        themes = []
                if isinstance(themes, list) and themes:
                    chips = "".join(f'<span class="theme-chip">{t}</span>' for t in themes)
                    st.markdown(chips, unsafe_allow_html=True)

                with st.expander("Promo concept / prompt used"):
                    st.write(video.get("prompt_used", "—"))

                st.markdown("</div>", unsafe_allow_html=True)

st.divider()
st.caption(
    "Multimodal E-Commerce AI Pipeline — Ingestion · AI Enrichment · Video Generation · Data Quality · Airflow"
)
