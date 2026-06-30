import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from pathlib import Path

    return Path, mo, pl


@app.cell
def _(mo):
    mo.md("""
    # Classifier Evaluation

    This notebook is used to qualitatively review the classifier output.

    The goal is to inspect each prediction alongside the original listing
    data, review the model's reasoning, and identify any prompt or
    preprocessing improvements.
    """)
    return


@app.cell
def _(Path, pl):
    OUTPUT_PATH = Path("data/classified_listings.csv")

    results_df = pl.read_csv(OUTPUT_PATH)

    {
        "rows": results_df.height,
        "columns": results_df.width,
    }
    return (results_df,)


@app.cell
def _(mo):
    mo.md("""
    ## 1. Prediction Summary
    """)
    return


@app.cell
def _(results_df):
    (
        results_df
        .group_by("category")
        .len()
        .sort("len", descending=True)
    )
    return


@app.cell
def _(results_df):
    (
        results_df
        .group_by("confidence")
        .len()
        .sort("len", descending=True)
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. Review Table
    """)
    return


@app.cell
def _(results_df):
    REVIEW_COLUMNS = [
        "id",
        "pageTitle",
        "summary",
        "propertySubType",
        "category",
        "confidence",
        "reasoning",
    ]

    existing_review_columns = [
        column
        for column in REVIEW_COLUMNS
        if column in results_df.columns
    ]

    review_df = results_df.select(existing_review_columns)

    review_df
    return (review_df,)


@app.cell
def _(mo):
    mo.md("""
    ## 3. Filter by Category and Confidence
    """)
    return


@app.cell
def _(mo, results_df):
    category_filter = mo.ui.dropdown(
        options=["All", *sorted(results_df["category"].drop_nulls().unique().to_list())],
        value="All",
        label="Category",
    )

    confidence_filter = mo.ui.dropdown(
        options=[
            "All",
            *sorted(results_df["confidence"].drop_nulls().unique().to_list()),
        ],
        value="All",
        label="Confidence",
    )

    mo.hstack([category_filter, confidence_filter])
    return category_filter, confidence_filter


@app.cell
def _(category_filter, confidence_filter, pl, review_df):
    filtered_df = review_df

    if category_filter.value != "All":
        filtered_df = filtered_df.filter(
            pl.col("category") == category_filter.value
        )

    if confidence_filter.value != "All":
        filtered_df = filtered_df.filter(
            pl.col("confidence") == confidence_filter.value
        )

    filtered_df
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Inspect One Listing
    """)
    return


@app.cell
def _(mo, pl, results_df):
    listing_ids = (
        results_df
        .select("id")
        .to_series()
        .cast(pl.String)
        .to_list()
    )

    selected_listing_id = mo.ui.dropdown(
        options=listing_ids,
        value=listing_ids[0],
        label="Listing ID",
    )

    selected_listing_id
    return (selected_listing_id,)


@app.cell
def _(pl, results_df, selected_listing_id):
    selected_listing = (
        results_df
        .filter(pl.col("id").cast(pl.String) == selected_listing_id.value)
        .row(0, named=True)
    )

    selected_listing
    return (selected_listing,)


@app.cell
def _(mo, selected_listing):
    mo.md(f"""
    ### Prediction

    **Category:** {selected_listing.get("category")}

    **Confidence:** {selected_listing.get("confidence")}

    **Reasoning:** {selected_listing.get("reasoning")}
    """)
    return


@app.cell
def _(mo, selected_listing):
    mo.md(f"""
    ### Listing Context

    **Page title:** {selected_listing.get("pageTitle")}

    **Summary:** {selected_listing.get("summary")}

    **Property subtype:** {selected_listing.get("propertySubType")}

    **Address:** {selected_listing.get("displayAddress")}

    **Key features:**  
    {selected_listing.get("keyFeatures")}

    **Description:**  
    {selected_listing.get("description") or selected_listing.get("detailedDescription")}
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Evaluation Notes

    Use this notebook to review:

    - whether the predicted category is supported by the listing text
    - whether the confidence level matches the strength of the evidence
    - whether the reasoning cites actual listing evidence
    - whether prompt or preprocessing changes are needed
    """)
    return


if __name__ == "__main__":
    app.run()
