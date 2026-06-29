from __future__ import annotations

import ast
import math
from typing import Any

from models import PreprocessedListing


PREPROCESSING_COLUMNS = [
    "pageTitle",
    "summary",
    "description",
    "detailedDescription",
    "shareDescription",
    "keyFeatures",
    "propertySubType",
    "name",
    "displayAddress",
    "address",
    "postalCode",
    "Region",
    "sizeFt",
    "sizeAc",
    "tenure",
    "tenureFull",
    "price",
    "primaryPrice",
    "pricePerUnit",
]


FIELD_LABELS = {
    "pageTitle": "Page title",
    "summary": "Summary",
    "description": "Description",
    "detailedDescription": "Detailed description",
    "shareDescription": "Share description",
    "keyFeatures": "Key features",
    "propertySubType": "Property subtype",
    "name": "Listing name",
    "displayAddress": "Display address",
    "address": "Address",
    "postalCode": "Postal code",
    "Region": "Region",
    "sizeFt": "Size in square feet",
    "sizeAc": "Size in acres",
    "tenure": "Tenure",
    "tenureFull": "Full tenure",
    "price": "Price",
    "primaryPrice": "Primary price",
    "pricePerUnit": "Price per unit",
}


def is_missing(value: Any) -> bool:
    """Return True if a raw CSV value should be treated as missing."""
    if value is None:
        return True

    if isinstance(value, float) and math.isnan(value):
        return True

    if isinstance(value, str) and not value.strip():
        return True

    return False


def clean_text(value: Any) -> str:
    """Convert a raw value into compact, readable text."""
    if is_missing(value):
        return ""

    return " ".join(str(value).strip().split())


def parse_key_features(value: Any) -> list[str]:
    """
    Parse the keyFeatures field.

    In the source CSV, keyFeatures is a stringified Python-style list rather
    than clean JSON. We parse it safely and fall back to the raw text if needed.
    """
    if is_missing(value):
        return []

    if isinstance(value, list):
        return [clean_text(item) for item in value if not is_missing(item)]

    raw_value = str(value).strip()

    try:
        parsed = ast.literal_eval(raw_value)
    except (ValueError, SyntaxError):
        return [clean_text(raw_value)]

    if not isinstance(parsed, list):
        return [clean_text(raw_value)]

    return [clean_text(item) for item in parsed if not is_missing(item)]


def format_key_features(value: Any) -> str:
    """Format parsed key features as a bullet list."""
    features = parse_key_features(value)

    if not features:
        return ""

    return "\n".join(f"- {feature}" for feature in features)


def format_field(field: str, value: Any) -> str:
    """Format a single listing field for the LLM context."""
    if field == "keyFeatures":
        formatted_features = format_key_features(value)

        if not formatted_features:
            return ""

        return f"{FIELD_LABELS[field]}:\n{formatted_features}"

    cleaned_value = clean_text(value)

    if not cleaned_value:
        return ""

    label = FIELD_LABELS.get(field, field)

    return f"{label}: {cleaned_value}"


def build_listing_context(listing: dict[str, Any]) -> str:
    """
    Build the concise text context sent to the LLM.

    We intentionally include only property-level fields identified during
    exploration, rather than dumping every raw CSV column into the prompt.
    """
    sections = [
        formatted_field
        for field in PREPROCESSING_COLUMNS
        if (formatted_field := format_field(field, listing.get(field)))
    ]

    return "\n\n".join(sections)


def preprocess_listing(listing: dict[str, Any]) -> PreprocessedListing:
    """Convert one raw listing row into the classifier input format."""
    return PreprocessedListing(
        id=str(listing.get("id")),
        context=build_listing_context(listing),
    )


def preprocess_listings(
    listings: list[dict[str, Any]],
) -> list[PreprocessedListing]:
    """Convert raw listing rows into classifier input format."""
    return [preprocess_listing(listing) for listing in listings]