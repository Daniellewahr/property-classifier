from src.preprocessing import (
    build_listing_context,
    clean_text,
    format_key_features,
    is_missing,
    parse_key_features,
)


def test_is_missing():
    assert is_missing(None)
    assert is_missing("")
    assert is_missing("   ")

    assert not is_missing("Nursery")
    assert not is_missing(123)


def test_clean_text():
    assert clean_text("  hello   world  ") == "hello world"
    assert clean_text(None) == ""


def test_parse_key_features_from_stringified_list():
    raw = "['Play area', 'Reception', 'Parking']"

    assert parse_key_features(raw) == [
        "Play area",
        "Reception",
        "Parking",
    ]


def test_parse_key_features_invalid_string():
    raw = "Play area, Reception"

    assert parse_key_features(raw) == ["Play area, Reception"]


def test_format_key_features():
    raw = "['Play area', 'Reception']"

    formatted = format_key_features(raw)

    assert "- Play area" in formatted
    assert "- Reception" in formatted


def test_build_listing_context():
    listing = {
        "pageTitle": "Former Children's Nursery",
        "summary": "Purpose-built nursery.",
        "keyFeatures": "['Outdoor play area']",
        "propertySubType": "Commercial property",
    }

    context = build_listing_context(listing)

    assert "Former Children's Nursery" in context
    assert "Purpose-built nursery." in context
    assert "- Outdoor play area" in context
