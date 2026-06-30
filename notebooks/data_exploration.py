import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import ast
    from pathlib import Path

    import marimo as mo
    import polars as pl

    return Path, ast, mo, pl


@app.cell
def _(mo):
    mo.md("""
    # Harkalm Listings Data Exploration

    This notebook explores `listings.csv` before building the classifier.

    The goal is to decide which columns contain useful signal for classifying each property as:

    - Nursery
    - SEN School
    - Food Store
    - None
    """)
    return


@app.cell
def _(Path, pl):
    DATA_PATH = Path("data/listings.csv")

    listings_df = pl.read_csv(DATA_PATH)

    {
        "rows": listings_df.height,
        "columns": listings_df.width,
    }
    return (listings_df,)


@app.cell
def _(mo):
    mo.md("""
    ## 1. Understand the raw dataset
    """)
    return


@app.cell
def _(listings_df, pl):
    schema_df = pl.DataFrame(
        {
            "column": listings_df.columns,
            "dtype": [str(dtype) for dtype in listings_df.dtypes],
        }
    )

    schema_df
    return


@app.cell
def _(listings_df, pl):
    column_profile = (
        listings_df.null_count()
        .transpose(
            include_header=True,
            header_name="column",
            column_names=["null_count"],
        )
        .with_columns(
            [
                (pl.col("null_count") / listings_df.height * 100)
                .round(2)
                .alias("percent_missing"),
                pl.Series(
                    "unique_values",
                    [listings_df[col].n_unique() for col in listings_df.columns],
                ),
            ]
        )
        .sort("percent_missing", descending=True)
    )

    column_profile
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. Remove obvious non-signal columns

    The original CSV has many fields that describe the listing source, agent, scrape metadata, or update lifecycle.

    Those fields do not help answer: **what kind of property is this?**
    """)
    return


@app.cell
def _(listings_df):
    DROP_COLUMNS = [
        # Coordinates
        "latitude",
        "longitude",
        # Listing lifecycle / metadata
        "firstVisibleDate",
        "listingUpdateDate",
        "updateDate",
        "listingUpdateReason",
        "postedDate",
        "monthsSincePosted",
        "listingHistory",
        # Agent information
        "agentCompanyAddress",
        "agentCompanyName",
        "agentCompanyPhone",
        "agentCompanyPostcode",
        "agentEmail",
        "agentName",
        "agentPhone",
        "agentStreet",
        "agentPostcode",
        "agentCity",
        # Search / scrape metadata
        "matched_keyword",
        "keyword",
        "changed_columns",
        # Images
        "numberOfImages",
    ]

    existing_drop_columns = [
        column for column in DROP_COLUMNS if column in listings_df.columns
    ]

    candidate_df = listings_df.drop(existing_drop_columns)

    fully_null_columns = [
        column
        for column in candidate_df.columns
        if candidate_df[column].null_count() == candidate_df.height
    ]

    candidate_df = candidate_df.drop(fully_null_columns)

    {
        "original_columns": listings_df.width,
        "dropped_metadata_columns": len(existing_drop_columns),
        "dropped_fully_null_columns": len(fully_null_columns),
        "remaining_columns": candidate_df.width,
    }
    return candidate_df, existing_drop_columns, fully_null_columns


@app.cell
def _(existing_drop_columns, fully_null_columns):
    {
        "metadata_columns_dropped": existing_drop_columns,
        "fully_null_columns_dropped": fully_null_columns,
    }
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Profile the remaining candidate columns
    """)
    return


@app.cell
def _(candidate_df, pl):
    candidate_profile = (
        candidate_df.null_count()
        .transpose(
            include_header=True,
            header_name="column",
            column_names=["null_count"],
        )
        .with_columns(
            [
                (pl.col("null_count") / candidate_df.height * 100)
                .round(2)
                .alias("percent_missing"),
                pl.Series(
                    "unique_values",
                    [
                        candidate_df[column].n_unique()
                        for column in candidate_df.columns
                    ],
                ),
            ]
        )
        .sort("percent_missing", descending=True)
    )

    candidate_profile
    return


@app.cell
def _(candidate_df, pl):
    text_column_stats = []

    for column, dtype in candidate_df.schema.items():
        if dtype == pl.String:
            text_column_stats.append(
                {
                    "column": column,
                    "non_null_count": candidate_df.select(
                        pl.col(column).is_not_null().sum()
                    ).item(),
                    "avg_chars": candidate_df.select(
                        pl.col(column).fill_null("").str.len_chars().mean()
                    ).item(),
                    "max_chars": candidate_df.select(
                        pl.col(column).fill_null("").str.len_chars().max()
                    ).item(),
                }
            )

    (pl.DataFrame(text_column_stats).sort("avg_chars", descending=True))
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Inspect candidate classification fields

    Now inspect the fields that may help a human classify the property.

    The goal is to identify the fields worth passing into preprocessing and eventually into the LLM prompt.
    """)
    return


@app.cell
def _(candidate_df):
    CANDIDATE_TEXT_COLUMNS = [
        "pageTitle",
        "summary",
        "description",
        "detailedDescription",
        "shareDescription",
        "keyFeatures",
        "propertySubType",
        "name",
        "infoReelItems",
        "analyticsTaxonomy",
    ]

    existing_candidate_text_columns = [
        column for column in CANDIDATE_TEXT_COLUMNS if column in candidate_df.columns
    ]

    existing_candidate_text_columns
    return (existing_candidate_text_columns,)


@app.cell
def _(candidate_df, existing_candidate_text_columns):
    # Change row_index to manually inspect different listings.
    row_index = 0

    (
        candidate_df.select(existing_candidate_text_columns)
        .slice(row_index, 1)
        .transpose(include_header=True)
    )
    return


@app.cell
def _(candidate_df, existing_candidate_text_columns):
    # Preview candidate text fields across a few listings.
    candidate_df.select(existing_candidate_text_columns).head(5)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. Inspect `keyFeatures`

    The assignment specifically calls out `keyFeatures` as messy.

    It appears to be a stringified list, so the production preprocessing step should parse it safely.
    """)
    return


@app.cell
def _(candidate_df, pl):
    (
        candidate_df.filter(pl.col("keyFeatures").is_not_null())
        .select("id", "keyFeatures")
        .head(10)
    )
    return


@app.cell
def _(ast, candidate_df, pl):
    key_feature_sample = (
        candidate_df.filter(pl.col("keyFeatures").is_not_null())
        .select("keyFeatures")
        .item(0, 0)
    )

    ast.literal_eval(key_feature_sample)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6. Select columns for preprocessing

    The preprocessing step should build a concise listing context for the LLM.

    The goal is **not** to pass every available field. The goal is to pass the fields that describe the property itself.
    """)
    return


@app.cell
def _(candidate_df):
    PREPROCESSING_COLUMNS = [
        # Required output identifier
        "id",
        # Strong text signal
        "pageTitle",
        "summary",
        "description",
        "detailedDescription",
        "shareDescription",
        "keyFeatures",
        # Property attributes
        "propertySubType",
        "name",
        # Location context
        "displayAddress",
        "address",
        "postalCode",
        "Region",
        # Supporting commercial context
        "sizeFt",
        "sizeAc",
        "tenure",
        "tenureFull",
        "price",
        "primaryPrice",
        "pricePerUnit",
    ]

    selected_columns = [
        column for column in PREPROCESSING_COLUMNS if column in candidate_df.columns
    ]

    preprocessing_df = candidate_df.select(selected_columns)

    preprocessing_df
    return (selected_columns,)


@app.cell
def _(selected_columns):
    selected_columns
    return


@app.cell
def _(mo):
    mo.md("""
    ## Exploration summary

    **Dropped:**

    - coordinates
    - agent contact fields
    - listing lifecycle metadata
    - scrape/search metadata
    - image counts
    - fully empty columns

    **Kept for preprocessing:**

    - listing title / summary
    - descriptions
    - `keyFeatures`
    - property subtype / name
    - location
    - size, tenure, and price as supporting context

    **Implementation note:**

    In `preprocessing.py`, each listing should be converted into a concise text block for the LLM.

    The preprocessing function should:

    - skip null fields
    - parse `keyFeatures` safely
    - avoid dumping all 67 raw columns into the model
    - preserve the original `id` for the output file
    """)
    return


if __name__ == "__main__":
    app.run()
