from src.classifier import PropertyClassifier
from src.models import (
    ClassificationResult,
    PreprocessedListing,
)


class FakeClassifier(PropertyClassifier):
    def classify(
        self,
        listing: PreprocessedListing,
    ) -> ClassificationResult:
        return ClassificationResult(
            id=listing.id,
            category="Nursery",
            confidence="High",
            reasoning="Test classification",
        )


def test_classify_many():
    classifier = FakeClassifier()

    listings = [
        PreprocessedListing(
            id="1",
            context="Nursery",
        ),
        PreprocessedListing(
            id="2",
            context="School",
        ),
    ]

    results = classifier.classify_many(listings)

    assert len(results) == 2
    assert results[0].category == "Nursery"
    assert results[1].category == "Nursery"
