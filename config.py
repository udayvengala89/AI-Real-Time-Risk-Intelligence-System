import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

NEWS_CSV_PATH = PROJECT_ROOT / "data" / "news.csv"
LIVE_NEWS_CSV_PATH = PROJECT_ROOT / "data" / "live_news.csv"
DATA_SOURCE = os.getenv("DATA_SOURCE", "CSV").strip().upper()
if DATA_SOURCE not in {"CSV", "LIVE"}:
	DATA_SOURCE = "CSV"
NEWS_API_TOKEN = os.getenv("NEWS_API_TOKEN", "").strip()
OUTPUT_JSON_PATH = PROJECT_ROOT / "outputs" / "event_detection.json"
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:latest")
OLLAMA_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120"))
