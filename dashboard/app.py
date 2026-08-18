import os

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

load_dotenv()

st.set_page_config(page_title="AI Sentiment Review Pipeline", layout="wide")

CACHE_TTL_SECONDS = 300

SENTIMENT_ORDER = ["Positive", "Negative", "Neutral"]
SENTIMENT_COLORS = {
    "Positive": "#0ca30c",
    "Negative": "#d03b3b",
    "Neutral": "#898781",
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


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_total_reviews() -> int:
    result = pd.read_sql(text("SELECT COUNT(*) AS total FROM reviews"), get_engine())
    return int(result["total"].iloc[0])


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_sentiment_breakdown() -> pd.DataFrame:
    return pd.read_sql(
        text("SELECT sentiment_label, COUNT(*) AS count FROM reviews GROUP BY sentiment_label"),
        get_engine(),
    )


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_sentiment_trend() -> pd.DataFrame:
    return pd.read_sql(
        text(
            "SELECT review_date, sentiment_label, COUNT(*) AS count "
            "FROM reviews GROUP BY review_date, sentiment_label ORDER BY review_date"
        ),
        get_engine(),
        parse_dates=["review_date"],
    )


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_sentiment_by_category() -> pd.DataFrame:
    return pd.read_sql(
        text(
            "SELECT product_category, sentiment_label, COUNT(*) AS count "
            "FROM reviews GROUP BY product_category, sentiment_label"
        ),
        get_engine(),
    )


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_top_negative_reviews() -> pd.DataFrame:
    return pd.read_sql(
        text(
            "SELECT review_text, confidence_score, review_date FROM reviews "
            "WHERE sentiment_label = 'Negative' ORDER BY confidence_score DESC LIMIT 10"
        ),
        get_engine(),
        parse_dates=["review_date"],
    )


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_last_pipeline_run() -> pd.DataFrame:
    return pd.read_sql(
        text("SELECT * FROM pipeline_runs ORDER BY run_at DESC LIMIT 1"),
        get_engine(),
        parse_dates=["run_at"],
    )


def render_sentiment_breakdown():
    st.subheader("Sentiment breakdown")
    breakdown_df = load_sentiment_breakdown()
    fig = px.pie(
        breakdown_df,
        names="sentiment_label",
        values="count",
        hole=0.5,
        color="sentiment_label",
        color_discrete_map=SENTIMENT_COLORS,
        category_orders={"sentiment_label": SENTIMENT_ORDER},
    )
    fig.update_traces(textinfo="label+percent")
    fig.update_layout(legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("View as table"):
        st.dataframe(breakdown_df, use_container_width=True, hide_index=True)


def render_sentiment_by_category():
    st.subheader("Sentiment by category")
    category_df = load_sentiment_by_category()
    if category_df.empty:
        st.info("No category data yet.")
        return
    fig = px.bar(
        category_df,
        x="product_category",
        y="count",
        color="sentiment_label",
        barmode="group",
        color_discrete_map=SENTIMENT_COLORS,
        category_orders={"sentiment_label": SENTIMENT_ORDER},
    )
    fig.update_layout(legend_title_text="", xaxis_title="", yaxis_title="Reviews")
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("View as table"):
        st.dataframe(category_df, use_container_width=True, hide_index=True)


def render_sentiment_trend():
    st.subheader("Sentiment trend over time")
    trend_df = load_sentiment_trend()
    if trend_df.empty:
        st.info("No trend data yet.")
        return
    fig = px.line(
        trend_df,
        x="review_date",
        y="count",
        color="sentiment_label",
        color_discrete_map=SENTIMENT_COLORS,
        category_orders={"sentiment_label": SENTIMENT_ORDER},
        markers=True,
    )
    fig.update_traces(line=dict(width=2))
    fig.update_layout(legend_title_text="", xaxis_title="", yaxis_title="Reviews")
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("View as table"):
        st.dataframe(trend_df, use_container_width=True, hide_index=True)


def render_top_negative_reviews():
    st.subheader("Top 10 most negative reviews")
    negative_df = load_top_negative_reviews()
    if negative_df.empty:
        st.info("No negative reviews found.")
        return
    st.dataframe(
        negative_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "review_text": st.column_config.TextColumn("Review", width="large"),
            "confidence_score": st.column_config.NumberColumn("Confidence", format="%.3f"),
            "review_date": st.column_config.DateColumn("Date"),
        },
    )


def render_last_pipeline_run():
    st.subheader("Last pipeline run")
    run_df = load_last_pipeline_run()
    if run_df.empty:
        st.info("No pipeline runs recorded yet.")
        return
    run = run_df.iloc[0]
    col1, col2, col3 = st.columns(3)
    col1.metric("Run at", run["run_at"].strftime("%Y-%m-%d %H:%M:%S"))
    records_loaded = run["records_loaded"]
    col2.metric("Records loaded", f"{int(records_loaded):,}" if pd.notna(records_loaded) else "—")
    with col3:
        status = run["status"]
        if status == "SUCCESS":
            st.success(f"Status: {status}")
        elif status == "FAILED":
            st.error(f"Status: {status}")
        else:
            st.info(f"Status: {status}")


def main():
    st.title("AI Sentiment Review Pipeline")

    if get_engine() is None:
        st.error(
            "DATABASE_URL is not configured. Set it in your .env file (local) "
            "or in your Streamlit Cloud app's Secrets."
        )
        return

    try:
        total_reviews = load_total_reviews()
    except SQLAlchemyError:
        st.info(
            "No data yet. This usually means the pipeline hasn't been run yet, "
            "or the database isn't reachable — run the pipeline first."
        )
        return

    if total_reviews == 0:
        st.info("No reviews in the database yet. Run the pipeline to see results here.")
        return

    st.metric("Total reviews processed", f"{total_reviews:,}")

    col1, col2 = st.columns(2)
    with col1:
        render_sentiment_breakdown()
    with col2:
        render_sentiment_by_category()

    render_sentiment_trend()
    render_top_negative_reviews()
    render_last_pipeline_run()


if __name__ == "__main__":
    main()
