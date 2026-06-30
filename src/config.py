import os

from dotenv import load_dotenv

load_dotenv()

LLM_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash",
)

TEMPERATURE = 0.0

REQUEST_DELAY_SECONDS = 12
