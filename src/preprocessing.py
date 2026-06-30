from __future__ import annotations

import ast
import math
from typing import Any

from src.models import PreprocessedListing

# Fields selected during data exploration that provide useful semantic context
# for the LLM. Each tuple contains the source column and the label used in the prompt.
CONTEXT_FIELDS = (
    ("pageTitle", "Page title"),
    ("summary", "Summary"),
    ("description", "Description"),
    ("detailedDescription", "Detailed description"),
    ("shareDescription", "Share description"),
    ("keyFeatures", "Key features"),
    ("propertySubType", "Property subtype"),
    ("name", "Listing name"),
    ("displayAddress", "Display address"),
    ("address", "Address"),
    ("postalCode", "Postal code"),
    ("Region", "Region"),
    ("price", "Price"),
)


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

    The source CSV stores key features as a stringified Python list rather
    than valid JSON. Parse it safely and fall back to the raw text if
    parsing fails.
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
    """Format key features as a bulleted list."""
    features = parse_key_features(value)

    if not features:
        return ""

    return "\n".join(f"- {feature}" for feature in features)


def format_field(
    label: str,
    field: str,
    value: Any,
) -> str:
    """Format a single field for the LLM context."""
    if field == "keyFeatures":
        formatted = format_key_features(value)

        if not formatted:
            return ""

        return f"{label}:\n{formatted}"

    cleaned = clean_text(value)

    if not cleaned:
        return ""

    return f"{label}: {cleaned}"


def build_listing_context(
    listing: dict[str, Any],
) -> str:
    """
    Build the text context sent to the language model.

    Only property-level information is included. Listing metadata,
    agent details, and scrape metadata are intentionally omitted.
    """
    sections: list[str] = []

    for field, label in CONTEXT_FIELDS:
        formatted = format_field(
            label=label,
            field=field,
            value=listing.get(field),
        )

        if formatted:
            sections.append(formatted)

    return "\n\n".join(sections)


def preprocess_listing(
    listing: dict[str, Any],
) -> PreprocessedListing:
    """Convert one raw listing into the classifier input format."""
    return PreprocessedListing(
        id=str(listing["id"]),
        context=build_listing_context(listing),
    )


def preprocess_listings(
    listings: list[dict[str, Any]],
) -> list[PreprocessedListing]:
    """Convert raw listings into classifier inputs."""
    return [preprocess_listing(listing) for listing in listings]


def test_empty_fields_are_removed():
    listing = {
        "summary": "",
        "description": None,
        "pageTitle": "Food Store",
    }

    context = build_listing_context(listing)

    assert "Food Store" in context
    assert "Summary" not in context
    assert "Description" not in context
