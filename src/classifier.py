from __future__ import annotations

import time

from google import genai

from src.config import LLM_MODEL, REQUEST_DELAY_SECONDS, TEMPERATURE
from src.models import ClassificationResult, LLMClassification, PreprocessedListing
from src.prompts import SYSTEM_PROMPT, build_classification_prompt


class ClassificationError(Exception):
    """Raised when a listing cannot be classified."""


class PropertyClassifier:
    """Classifies commercial property listings using an LLM."""

    def __init__(
        self,
        model: str = LLM_MODEL,
        client: genai.Client | None = None,
    ) -> None:
        self._model = model
        self._client = client or genai.Client()

    def classify(self, listing: PreprocessedListing) -> ClassificationResult:
        """Classify one preprocessed listing."""
        parsed_response = self._request_structured_classification(listing)

        return ClassificationResult(
            id=listing.id,
            category=parsed_response.category,
            confidence=parsed_response.confidence,
            reasoning=parsed_response.reasoning,
        )

    def classify_many(
        self,
        listings: list[PreprocessedListing],
    ) -> list[ClassificationResult]:
        """
        Classify multiple listings.

        If one listing fails, return a low-confidence fallback result rather
        than failing the entire batch.
        """
        results: list[ClassificationResult] = []
        total = len(listings)

        # The assignment dataset is small, so sequential processing keeps the
        # implementation simple. The delay helps respect Gemini free-tier
        # request limits; larger datasets could use batching or queued retries.
        for index, listing in enumerate(listings, start=1):
            print(f"[{index}/{total}] Classifying listing {listing.id}...")

            try:
                result = self.classify(listing)
                results.append(result)
                print(f"      ✓ {result.category} ({result.confidence})")
            except Exception as error:
                print(f"      ⚠️ Failed: {type(error).__name__}: {error}")
                results.append(self._build_fallback_result(listing, error))

            if index < total:
                time.sleep(REQUEST_DELAY_SECONDS)

        print(f"\n✓ Finished classifying {total} listings.")

        return results

    def _request_structured_classification(
        self,
        listing: PreprocessedListing,
    ) -> LLMClassification:
        """Request a schema-validated classification from the LLM."""
        response = self._client.models.generate_content(
            model=self._model,
            contents=[
                SYSTEM_PROMPT,
                build_classification_prompt(listing.context),
            ],
            config={
                "temperature": TEMPERATURE,
                "response_mime_type": "application/json",
                "response_schema": LLMClassification,
            },
        )

        parsed_response = response.parsed

        if parsed_response is None:
            raise ClassificationError(
                f"No parsed response returned for listing {listing.id}"
            )

        return parsed_response

    def _build_fallback_result(
        self,
        listing: PreprocessedListing,
        error: Exception,
    ) -> ClassificationResult:
        """Return a safe fallback result when classification fails."""
        return ClassificationResult(
            id=listing.id,
            category="None",
            confidence="Low",
            reasoning=f"Classification failed: {type(error).__name__}",
        )