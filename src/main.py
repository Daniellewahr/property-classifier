def main():
    listings = load_listings()

    cleaned = preprocess_listings(listings)

    results = classify_listings(cleaned)

    save_results(results)