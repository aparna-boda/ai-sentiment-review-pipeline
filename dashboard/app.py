import os
import re
from collections import Counter

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import bindparam, create_engine, text
from sqlalchemy.exc import SQLAlchemyError

load_dotenv()

st.set_page_config(page_title="AI Sentiment Review Pipeline", layout="wide", initial_sidebar_state="collapsed")

CACHE_TTL_SECONDS = 300

SENTIMENT_ORDER = ["Positive", "Negative", "Neutral"]
SENTIMENT_COLORS = {
    "Positive": "#0ca30c",
    "Negative": "#d03b3b",
    "Neutral": "#898781",
}

# Chart chrome tokens — validated reference palette (dataviz skill, references/palette.md).
# Status colors (SENTIMENT_COLORS above) are fixed and identical in both modes; only
# chrome (surfaces/ink/grid) swaps between light and dark.
THEME_TOKENS = {
    "light": {
        "chart_surface": "#fcfcfb",
        "page_plane": "#f9f9f7",
        "sidebar_bg": "#f6f5f2",
        "text_primary": "#0b0b0b",
        "text_secondary": "#52514e",
        "gridline": "#e1e0d9",
        "baseline": "#c3c2b7",
        "sequential_hue": "#2a78d6",
    },
    "dark": {
        "chart_surface": "#1a1a19",
        "page_plane": "#0d0d0d",
        "sidebar_bg": "#161615",
        "text_primary": "#ffffff",
        "text_secondary": "#c3c2b7",
        "gridline": "#2c2c2a",
        "baseline": "#383835",
        "sequential_hue": "#3987e5",
    },
}

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "was", "were", "are", "be", "been", "being",
    "to", "of", "in", "on", "for", "with", "this", "that", "it", "its", "as", "at", "by",
    "from", "i", "you", "he", "she", "they", "we", "my", "your", "their", "our", "his", "her",
    "not", "no", "so", "if", "just", "than", "then", "there", "here", "these", "those", "have",
    "has", "had", "do", "does", "did", "will", "would", "could", "should", "can", "one", "all",
    "out", "up", "about", "into", "over", "after", "very", "too", "much", "more", "most", "some",
    "what", "which", "who", "when", "where", "why", "how", "am", "im", "me", "us", "them", "also",
}


def _normalise_database_url(database_url: str) -> str:
    if database_url.startswith("postgres://"):
        return database_url.replace("postgres://", "postgresql://", 1)
    return database_url


@st.cache_resource
def get_engine():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        return None
    return create_engine(_normalise_database_url(database_url))


def _themed(fig, tokens):
    fig.update_layout(
        paper_bgcolor=tokens["chart_surface"],
        plot_bgcolor=tokens["chart_surface"],
        font_color=tokens["text_secondary"],
        legend_title_text="",
        margin=dict(t=10, b=10, l=10, r=10),
    )
    fig.update_xaxes(gridcolor=tokens["gridline"], linecolor=tokens["baseline"], zerolinecolor=tokens["baseline"])
    fig.update_yaxes(gridcolor=tokens["gridline"], linecolor=tokens["baseline"], zerolinecolor=tokens["baseline"])
    return fig


def inject_theme_css(dark: bool) -> None:
    tokens = THEME_TOKENS["dark" if dark else "light"]
    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{ background-color: {tokens["page_plane"]}; }}
        [data-testid="stHeader"] {{ background-color: transparent; }}
        [data-testid="stSidebar"] {{ background-color: {tokens["sidebar_bg"]}; }}
        [data-testid="stVerticalBlockBorderWrapper"] > div {{ background-color: {tokens["chart_surface"]}; }}
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"],
        h1, h2, h3, p, span, label, .stMarkdown {{ color: {tokens["text_primary"]}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Data access — filter-aware queries. Filters (date range, category, sentiment)
# are passed as plain hashable args so st.cache_data can key on them.
# ---------------------------------------------------------------------------

FILTER_WHERE = "review_date BETWEEN :start_date AND :end_date AND product_category IN :categories AND sentiment_label IN :sentiments"


def _filtered_sql(select_clause: str, extra: str = "") -> text:
    return text(f"{select_clause} WHERE {FILTER_WHERE} {extra}").bindparams(
        bindparam("categories", expanding=True),
        bindparam("sentiments", expanding=True),
    )


def _filter_params(start_date, end_date, categories, sentiments) -> dict:
    return {
        "start_date": start_date,
        "end_date": end_date,
        "categories": list(categories),
        "sentiments": list(sentiments),
    }


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_filter_bounds():
    bounds = pd.read_sql(
        text("SELECT MIN(review_date) AS min_date, MAX(review_date) AS max_date FROM reviews"),
        get_engine(),
    )
    categories = pd.read_sql(
        text("SELECT DISTINCT product_category FROM reviews ORDER BY product_category"),
        get_engine(),
    )["product_category"].tolist()
    return bounds["min_date"].iloc[0], bounds["max_date"].iloc[0], categories


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_pipeline_runs(limit: int = 2) -> pd.DataFrame:
    return pd.read_sql(
        text("SELECT * FROM pipeline_runs ORDER BY run_at DESC LIMIT :limit"),
        get_engine(),
        params={"limit": limit},
        parse_dates=["run_at"],
    )


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_run_shift(cutoff) -> dict:
    row = pd.read_sql(
        text(
            "SELECT "
            "COUNT(*) FILTER (WHERE scored_at < :cutoff) AS before_total, "
            "COUNT(*) FILTER (WHERE scored_at < :cutoff AND sentiment_label = 'Positive') AS before_pos, "
            "COUNT(*) FILTER (WHERE scored_at < :cutoff AND sentiment_label = 'Negative') AS before_neg, "
            "COUNT(*) FILTER (WHERE scored_at < :cutoff AND sentiment_label = 'Neutral') AS before_neu, "
            "AVG(confidence_score) FILTER (WHERE scored_at < :cutoff) AS before_avg_confidence, "
            "AVG(star_rating) FILTER (WHERE scored_at < :cutoff) AS before_avg_rating, "
            "COUNT(*) AS after_total, "
            "COUNT(*) FILTER (WHERE sentiment_label = 'Positive') AS after_pos, "
            "COUNT(*) FILTER (WHERE sentiment_label = 'Negative') AS after_neg, "
            "COUNT(*) FILTER (WHERE sentiment_label = 'Neutral') AS after_neu, "
            "AVG(confidence_score) AS after_avg_confidence, "
            "AVG(star_rating) AS after_avg_rating "
            "FROM reviews"
        ),
        get_engine(),
        params={"cutoff": cutoff},
    ).iloc[0]
    return row.to_dict()


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_sentiment_breakdown(start_date, end_date, categories, sentiments) -> pd.DataFrame:
    stmt = _filtered_sql(
        "SELECT sentiment_label, COUNT(*) AS count FROM reviews", "GROUP BY sentiment_label"
    )
    return pd.read_sql(stmt, get_engine(), params=_filter_params(start_date, end_date, categories, sentiments))


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_sentiment_trend(start_date, end_date, categories, sentiments) -> pd.DataFrame:
    stmt = _filtered_sql(
        "SELECT review_date, sentiment_label, COUNT(*) AS count FROM reviews",
        "GROUP BY review_date, sentiment_label ORDER BY review_date",
    )
    return pd.read_sql(
        stmt,
        get_engine(),
        params=_filter_params(start_date, end_date, categories, sentiments),
        parse_dates=["review_date"],
    )


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_sentiment_by_category(start_date, end_date, categories, sentiments) -> pd.DataFrame:
    stmt = _filtered_sql(
        "SELECT product_category, sentiment_label, COUNT(*) AS count FROM reviews",
        "GROUP BY product_category, sentiment_label",
    )
    return pd.read_sql(stmt, get_engine(), params=_filter_params(start_date, end_date, categories, sentiments))


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_top_negative_reviews(start_date, end_date, categories) -> pd.DataFrame:
    stmt = text(
        "SELECT review_text, confidence_score, review_date FROM reviews "
        "WHERE review_date BETWEEN :start_date AND :end_date AND product_category IN :categories "
        "AND sentiment_label = 'Negative' ORDER BY confidence_score DESC LIMIT 10"
    ).bindparams(bindparam("categories", expanding=True))
    return pd.read_sql(
        stmt,
        get_engine(),
        params={"start_date": start_date, "end_date": end_date, "categories": list(categories)},
        parse_dates=["review_date"],
    )


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_negative_keyword_counts(start_date, end_date, categories, top_n: int = 15) -> pd.DataFrame:
    stmt = text(
        "SELECT review_text FROM reviews "
        "WHERE review_date BETWEEN :start_date AND :end_date AND product_category IN :categories "
        "AND sentiment_label = 'Negative' LIMIT 3000"
    ).bindparams(bindparam("categories", expanding=True))
    texts = pd.read_sql(
        stmt,
        get_engine(),
        params={"start_date": start_date, "end_date": end_date, "categories": list(categories)},
    )["review_text"]

    counts = Counter()
    for review_text in texts:
        words = re.findall(r"[a-zA-Z']{3,}", review_text.lower())
        counts.update(w for w in words if w not in STOPWORDS)

    top = counts.most_common(top_n)
    return pd.DataFrame(top, columns=["word", "count"])


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _pct_delta(after, before, decimals=1):
    if before is None:
        return "first run"
    return f"{after - before:+.{decimals}f} pp"


def render_kpi_row(tokens):
    runs = load_pipeline_runs(limit=2)
    if runs.empty:
        st.info("No pipeline runs recorded yet.")
        return

    last_run = runs.iloc[0]
    prev_run = runs.iloc[1] if len(runs) > 1 else None
    shift = load_run_shift(last_run["run_at"])
    after_total = int(shift["after_total"] or 0)
    before_total = int(shift["before_total"] or 0)

    after_pos_pct = (shift["after_pos"] / after_total * 100) if after_total else 0.0
    after_neg_pct = (shift["after_neg"] / after_total * 100) if after_total else 0.0
    after_neu_pct = (shift["after_neu"] / after_total * 100) if after_total else 0.0
    before_pos_pct = (shift["before_pos"] / before_total * 100) if before_total else None
    before_neg_pct = (shift["before_neg"] / before_total * 100) if before_total else None
    before_neu_pct = (shift["before_neu"] / before_total * 100) if before_total else None

    after_avg_confidence = shift["after_avg_confidence"] or 0.0
    before_avg_confidence = shift["before_avg_confidence"]
    after_avg_rating = shift["after_avg_rating"] or 0.0
    before_avg_rating = shift["before_avg_rating"]

    pass_rate = (
        last_run["records_passed_qc"] / last_run["records_scored"] * 100
        if last_run["records_scored"] else 0.0
    )
    prev_pass_rate = None
    if prev_run is not None and prev_run["records_scored"]:
        prev_pass_rate = prev_run["records_passed_qc"] / prev_run["records_scored"] * 100

    row1 = st.columns(4)
    with row1[0], st.container(border=True):
        st.metric(
            "Total reviews",
            f"{after_total:,}",
            delta=f"+{int(last_run['records_loaded']):,} this run" if pd.notna(last_run["records_loaded"]) else None,
        )
    with row1[1], st.container(border=True):
        st.metric("Positive share", f"{after_pos_pct:.1f}%", delta=_pct_delta(after_pos_pct, before_pos_pct))
    with row1[2], st.container(border=True):
        st.metric(
            "Negative share", f"{after_neg_pct:.1f}%",
            delta=_pct_delta(after_neg_pct, before_neg_pct), delta_color="inverse",
        )
    with row1[3], st.container(border=True):
        st.metric(
            "Neutral share", f"{after_neu_pct:.1f}%",
            delta=_pct_delta(after_neu_pct, before_neu_pct), delta_color="off",
        )

    row2 = st.columns(4)
    with row2[0], st.container(border=True):
        conf_delta = (
            f"{after_avg_confidence - before_avg_confidence:+.3f}" if before_avg_confidence is not None else "first run"
        )
        st.metric("Avg. confidence", f"{after_avg_confidence:.3f}", delta=conf_delta, delta_color="off")
    with row2[1], st.container(border=True):
        pass_delta = f"{pass_rate - prev_pass_rate:+.1f} pp" if prev_pass_rate is not None else "first run"
        st.metric("QC pass rate", f"{pass_rate:.1f}%", delta=pass_delta)
    with row2[2], st.container(border=True):
        rating_delta = (
            f"{after_avg_rating - before_avg_rating:+.2f}" if before_avg_rating is not None else "first run"
        )
        st.metric("Avg. star rating", f"{after_avg_rating:.2f}", delta=rating_delta, delta_color="off")
    with row2[3], st.container(border=True):
        status = last_run["status"]
        st.caption("Last pipeline run")
        st.write(f"**{last_run['run_at'].strftime('%Y-%m-%d %H:%M:%S')}**")
        if status == "SUCCESS":
            st.success(f"● {status}")
        elif status == "FAILED":
            st.error(f"● {status}")
        else:
            st.info(f"● {status}")


def render_sentiment_breakdown(df: pd.DataFrame, tokens):
    st.subheader("Sentiment breakdown")
    if df.empty:
        st.info("No reviews match the current filters.")
        return
    fig = px.pie(
        df,
        names="sentiment_label",
        values="count",
        hole=0.5,
        color="sentiment_label",
        color_discrete_map=SENTIMENT_COLORS,
        category_orders={"sentiment_label": SENTIMENT_ORDER},
    )
    fig.update_traces(textinfo="label+percent")
    _themed(fig, tokens)
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("View as table"):
        st.dataframe(df, use_container_width=True, hide_index=True)


def render_sentiment_by_category(df: pd.DataFrame, tokens):
    st.subheader("Sentiment by category")
    if df.empty:
        st.info("No reviews match the current filters.")
        return
    fig = px.bar(
        df,
        x="product_category",
        y="count",
        color="sentiment_label",
        barmode="group",
        color_discrete_map=SENTIMENT_COLORS,
        category_orders={"sentiment_label": SENTIMENT_ORDER},
    )
    fig.update_layout(xaxis_title="", yaxis_title="Reviews")
    _themed(fig, tokens)
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("View as table"):
        st.dataframe(df, use_container_width=True, hide_index=True)


def render_sentiment_trend(df: pd.DataFrame, tokens):
    st.subheader("Sentiment trend over time")
    if df.empty:
        st.info("No reviews match the current filters.")
        return
    fig = px.line(
        df,
        x="review_date",
        y="count",
        color="sentiment_label",
        color_discrete_map=SENTIMENT_COLORS,
        category_orders={"sentiment_label": SENTIMENT_ORDER},
        markers=True,
    )
    fig.update_traces(line=dict(width=2))
    fig.update_layout(xaxis_title="", yaxis_title="Reviews")
    _themed(fig, tokens)
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("View as table"):
        st.dataframe(df, use_container_width=True, hide_index=True)


def render_negative_keywords(df: pd.DataFrame, tokens):
    st.subheader("Most common words in negative reviews")
    st.caption("Always scoped to Negative reviews (independent of the sentiment filter above) — respects date range and category.")
    if df.empty:
        st.info("No negative reviews match the current date range / category filters.")
        return
    fig = px.bar(
        df.sort_values("count"),
        x="count",
        y="word",
        orientation="h",
    )
    fig.update_traces(marker_color=tokens["sequential_hue"])
    fig.update_layout(xaxis_title="Occurrences", yaxis_title="")
    _themed(fig, tokens)
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("View as table"):
        st.dataframe(df, use_container_width=True, hide_index=True)


def render_top_negative_reviews(df: pd.DataFrame):
    st.subheader("Top 10 most negative reviews")
    st.caption("Always scoped to Negative reviews — respects date range and category.")
    if df.empty:
        st.info("No negative reviews match the current date range / category filters.")
        return
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "review_text": st.column_config.TextColumn("Review", width="large"),
            "confidence_score": st.column_config.NumberColumn("Confidence", format="%.3f"),
            "review_date": st.column_config.DateColumn("Date"),
        },
    )


def render_sidebar_filters():
    min_date, max_date, categories = load_filter_bounds()
    with st.sidebar:
        st.header("Filters")
        date_range = st.date_input(
            "Review date range", value=(min_date, max_date), min_value=min_date, max_value=max_date
        )
        selected_categories = st.multiselect("Product category", options=categories, default=categories)
        selected_sentiments = st.multiselect("Sentiment", options=SENTIMENT_ORDER, default=SENTIMENT_ORDER)
        st.divider()
        dark_mode = st.toggle("Dark mode", value=False)

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

    return start_date, end_date, tuple(sorted(selected_categories)), tuple(sorted(selected_sentiments)), dark_mode


def main():
    st.title("AI Sentiment Review Pipeline")
    st.caption("Use the ›› arrow at the top-left to slide in date, category, and sentiment filters.")

    if get_engine() is None:
        st.error(
            "DATABASE_URL is not configured. Set it in your .env file (local) "
            "or in your Streamlit Cloud app's Secrets."
        )
        return

    try:
        total_reviews = pd.read_sql(text("SELECT COUNT(*) AS total FROM reviews"), get_engine())["total"].iloc[0]
    except SQLAlchemyError:
        st.info(
            "No data yet. This usually means the pipeline hasn't been run yet, "
            "or the database isn't reachable — run the pipeline first."
        )
        return

    if total_reviews == 0:
        st.info("No reviews in the database yet. Run the pipeline to see results here.")
        return

    start_date, end_date, categories, sentiments, dark_mode = render_sidebar_filters()
    tokens = THEME_TOKENS["dark" if dark_mode else "light"]
    inject_theme_css(dark_mode)

    if not categories or not sentiments:
        st.warning("Select at least one category and one sentiment in the sidebar to see filtered charts.")
        return

    render_kpi_row(tokens)
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        render_sentiment_breakdown(load_sentiment_breakdown(start_date, end_date, categories, sentiments), tokens)
    with col2:
        render_sentiment_by_category(load_sentiment_by_category(start_date, end_date, categories, sentiments), tokens)

    render_sentiment_trend(load_sentiment_trend(start_date, end_date, categories, sentiments), tokens)

    col3, col4 = st.columns(2)
    with col3:
        render_negative_keywords(load_negative_keyword_counts(start_date, end_date, categories), tokens)
    with col4:
        render_top_negative_reviews(load_top_negative_reviews(start_date, end_date, categories))


if __name__ == "__main__":
    main()
