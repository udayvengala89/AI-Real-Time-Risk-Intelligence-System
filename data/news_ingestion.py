"""Optional The News API ingestion for the live-news POC mode."""

import csv
import io
import sys
from pathlib import Path
from typing import Any

import requests

# Direct execution sets sys.path to data/; add the project root for config.py.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from config import LIVE_NEWS_CSV_PATH, NEWS_API_TOKEN, PROJECT_ROOT


NEWS_API_URL = "https://api.thenewsapi.com/v1/news/all"
SEARCH_TERMS = ('"crude oil"', '"oil prices"', '"Brent crude"', '"WTI crude"', "OPEC", '"oil supply"')
CSV_FIELDS = ["article_id", "headline", "description", "source", "published_at", "url"]
MAX_ARTICLES = 3


def fetch_live_news(
	output_path: Path = LIVE_NEWS_CSV_PATH,
	token: str | None = None,
) -> tuple[bool, str]:
	"""Fetch at most three articles and atomically replace the live-news CSV."""
	api_token = (NEWS_API_TOKEN if token is None else token).strip()
	if not api_token or api_token.upper() in {"YOUR_TOKEN", "YOUR_API_TOKEN"}:
		return False, "The News API token is missing or is still the placeholder."

	params = {
		"api_token": api_token,
		"search": " OR ".join(SEARCH_TERMS),
		"language": "en",
		"limit": MAX_ARTICLES,
		"sort": "published_at",
		"sort_order": "desc",
	}

	# Do not print request URLs or exception text because the query token is private.
	try:
		response = requests.get(NEWS_API_URL, params=params, timeout=20)
		response.raise_for_status()
		payload: Any = response.json()
	except requests.Timeout:
		return False, "The News API request timed out."
	except requests.RequestException:
		return False, "The News API request failed."
	except ValueError:
		return False, "The News API returned invalid JSON."

	if not isinstance(payload, dict) or payload.get("error") or not isinstance(payload.get("data"), list):
		return False, "The News API returned an error response."

	# Keep only the API's first three records and normalize them to Agent 1's input.
	rows = []
	for index, item in enumerate(payload["data"][:MAX_ARTICLES], start=1):
		if not isinstance(item, dict):
			continue
		headline = str(item.get("title") or "").strip()
		if not headline:
			continue
		source = item.get("source") or item.get("source_name") or ""
		if isinstance(source, dict):
			source = source.get("name") or source.get("domain") or ""
		rows.append(
			{
				"article_id": str(item.get("uuid") or item.get("id") or f"newsapi-{index}"),
				"headline": headline,
				"description": str(item.get("description") or item.get("snippet") or ""),
				"source": str(source),
				"published_at": str(item.get("published_at") or ""),
				"url": str(item.get("url") or ""),
			}
		)

	# Build the entire file in memory so a failed fetch/normalization cannot truncate it.
	buffer = io.StringIO(newline="")
	writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS, extrasaction="ignore")
	writer.writeheader()
	writer.writerows(rows)
	try:
		output_path.parent.mkdir(parents=True, exist_ok=True)
		output_path.write_text(buffer.getvalue(), encoding="utf-8", newline="")
	except OSError:
		return False, "Could not create data/live_news.csv."

	return True, f"Retrieved {len(rows)} article(s) and saved {output_path.name}."


def main() -> None:
	"""Allow optional manual verification without exposing the API token."""
	success, message = fetch_live_news()
	print(message if success else f"Live news unavailable: {message}")
	if success:
		print(f"Saved to {LIVE_NEWS_CSV_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
	main()