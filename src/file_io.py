from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from src.models import ClassificationResult

def read_listings(input_path: Path) -> list[dict[str, Any]]:
    """Read raw listings from the input CSV."""
    return pl.read_csv(input_path).to_dicts()

def read_completed_listing_ids(output_path: Path) -> set[str]:
    """Read IDs that already have successful classifications."""
    if not output_path.exists():
        return set()

    output_df = pl.read_csv(output_path)

    required_columns = {"id", "reasoning"}
    if not required_columns.issubset(output_df.columns):
        return set()

    completed_df = output_df.filter(
        pl.col("reasoning").is_not_null()
        & ~pl.col("reasoning").str.starts_with("Classification failed")
    )

    return set(
        completed_df
        .select("id")
        .to_series()
        .cast(pl.String)
        .to_list()
    )

def write_classification_results(
    input_path: Path,
    output_path: Path,
    results: list[ClassificationResult],
) -> None:
    """Write classification results alongside the original listing data."""
    original_df = pl.read_csv(input_path)

    new_results_df = pl.DataFrame([result.model_dump() for result in results])

    if output_path.exists():
        existing_output_df = pl.read_csv(output_path)

        existing_results_df = existing_output_df.select(
            ["id", "category", "confidence", "reasoning"]
        )

        results_df = (
            pl.concat([existing_results_df, new_results_df], how="vertical")
            .unique(subset=["id"], keep="last")
        )
    else:
        results_df = new_results_df

    output_df = original_df.join(
        results_df,
        on="id",
        how="left",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.write_csv(output_path)