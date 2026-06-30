from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from src.classifier import PropertyClassifier
from src.file_io import read_listings, read_completed_listing_ids, write_classification_results
from src.preprocessing import preprocess_listings

load_dotenv()


INPUT_PATH = Path("data/listings.csv")
OUTPUT_PATH = Path("data/classified_listings.csv")


def main() -> None:
    """Run the end-to-end property classification pipeline."""

    raw_listings = read_listings(INPUT_PATH)

    completed_ids = read_completed_listing_ids(OUTPUT_PATH)

    remaining_listings = [
        listing
        for listing in raw_listings
        if str(listing["id"]) not in completed_ids
    ]

    if not remaining_listings:
        print("✓ All listings have already been classified.")
        return

    print(f"Found {len(completed_ids)} previously classified listings.")
    print(f"Processing {len(remaining_listings)} remaining listings.")

    preprocessed_listings = preprocess_listings(remaining_listings)

    classifier = PropertyClassifier()

    classification_results = classifier.classify_many(preprocessed_listings)

    write_classification_results(
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
        results=classification_results,
    )

    print(f"Wrote {len(classification_results)} results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
