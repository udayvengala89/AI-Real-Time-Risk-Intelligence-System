"""Agent 1: identify potentially material events for analyst investigation."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from config import DATA_SOURCE, LIVE_NEWS_CSV_PATH, PROJECT_ROOT


# Keep these settings local to this agent so its required model and API are explicit.
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"
REQUEST_TIMEOUT_SECONDS = 60
RULE_ID = "GENERAL-EVENT-001"

# Ollama uses this schema to constrain the model response to JSON.
EVENT_SCHEMA = {
	"type": "object",
	"properties": {
		"relevant": {"type": "boolean"},
		"event_type": {"type": "string"},
		"event_summary": {"type": "string"},
		"cause": {"type": "string"},
		"price_direction": {
			"type": "string",
			"enum": ["up", "down", "mixed", "none", "unclear"],
		},
		"affected_geographies": {
			"type": "array",
			"items": {"type": "string"},
		},
		"potentially_affected_industries": {
			"type": "array",
			"items": {"type": "string"},
		},
		"confidence": {"type": "number", "minimum": 0, "maximum": 1},
		"reason": {"type": "string"},
	},
	"required": [
		"relevant",
		"event_type",
		"event_summary",
		"cause",
		"price_direction",
		"affected_geographies",
		"potentially_affected_industries",
		"confidence",
		"reason",
	],
	"additionalProperties": False,
}


def _error_result(article_id: str, reason: str) -> dict[str, Any]:
	"""Return the normal result shape when Ollama cannot provide an answer."""
	return {
		"article_id": article_id,
		"rule_id": RULE_ID,
		"relevant": False,
		"event_type": "",
		"event_summary": "",
		"cause": "",
		"price_direction": "",
		"affected_geographies": [],
		"potentially_affected_industries": [],
		"confidence": 0.0,
		"reason": f"ERROR: {reason}",
	}


def detect_event(article: dict[str, Any]) -> dict[str, Any]:
	"""Classify one article for potentially material analyst-investigation events."""
	article_id = str(article.get("id", ""))
	article_json = json.dumps(article, ensure_ascii=False, default=str)

	# Ask for semantic assessment of the event, not keyword matching or a rating action.
	prompt = f"""
You are Agent 1, a general event-detection assistant supporting a credit and
rating analyst. Apply rule {RULE_ID}: Potentially Material Business Event.

Mark an article relevant only when its meaning describes a potentially material
event that may warrant analyst investigation. Consider meaningful supply-chain
or input shortages, material commodity or cost changes, operational disruptions,
regulatory or legal actions, geopolitical developments, sanctions, labor events,
material demand or market shifts, and other significant business risks. Assess
the reported meaning rather than matching keywords. Do not flag an article
merely because it mentions a company, industry, commodity, or risk topic.

This agent only identifies events for possible analyst investigation. Do not
recommend or determine a credit rating upgrade, downgrade, or other rating action.
Use concise strings for event_type, event_summary, cause, price_direction, and
reason. Use price_direction only when the article supports a clear market or
business direction; otherwise use "unclear" or "none". If the article is not
relevant, use empty strings for those fields, empty arrays, and confidence 0.
Return ONLY valid JSON with exactly these fields: relevant, event_type,
event_summary, cause, price_direction, affected_geographies,
potentially_affected_industries, confidence, reason. Do not include article_id,
rule_id, markdown, or any text outside the JSON object.

Article:
{article_json}
""".strip()

	# Send one non-streaming request so the complete JSON response is available at once.
	try:
		print("Sending article to Ollama...", flush=True)
		response = requests.post(
			OLLAMA_URL,
			json={
				"model": OLLAMA_MODEL,
				"prompt": prompt,
				"format": EVENT_SCHEMA,
				"stream": False,
				"options": {"temperature": 0},
			},
			timeout=REQUEST_TIMEOUT_SECONDS,
		)
		print("Ollama response received.", flush=True)
		response.raise_for_status()
	except requests.Timeout:
		return _error_result(article_id, "Ollama request timed out.")
	except requests.ConnectionError:
		return _error_result(article_id, "Ollama is unavailable at the configured URL.")
	except requests.RequestException as error:
		return _error_result(article_id, f"Ollama request failed: {error}")

	# Decode both the Ollama response envelope and its JSON-only model response.
	try:
		model_output = response.json()["response"]
		result = json.loads(model_output)
	except (KeyError, TypeError, ValueError):
		print("Ollama returned invalid JSON. Raw response:", flush=True)
		print(response.text, flush=True)
		return _error_result(article_id, "Ollama returned invalid JSON.")

	# Reject incomplete or malformed objects rather than silently trusting them.
	if not isinstance(result, dict) or set(result) != set(EVENT_SCHEMA["required"]):
		print("Ollama response did not match the expected schema. Raw response:", flush=True)
		print(response.text, flush=True)
		return _error_result(article_id, "Ollama JSON did not match the event schema.")
	if not isinstance(result["relevant"], bool):
		return _error_result(article_id, "Ollama JSON contains an invalid relevant value.")
	if not isinstance(result["confidence"], (int, float)) or isinstance(result["confidence"], bool):
		return _error_result(article_id, "Ollama JSON contains an invalid confidence value.")
	if not 0 <= result["confidence"] <= 1:
		return _error_result(article_id, "Ollama confidence must be between 0 and 1.")
	if not all(
		isinstance(result[field], list)
		and all(isinstance(item, str) for item in result[field])
		for field in ("affected_geographies", "potentially_affected_industries")
	):
		return _error_result(article_id, "Ollama JSON contains invalid list values.")
	if not all(
		isinstance(result[field], str)
		for field in ("event_type", "event_summary", "cause", "price_direction", "reason")
	):
		return _error_result(article_id, "Ollama JSON contains invalid text values.")

	# Keep identifiers authoritative even if the model changes or omits them.
	result["article_id"] = article_id
	result["rule_id"] = RULE_ID
	return result


def process_news_file(file_path: str | Path) -> list[dict[str, Any]]:
	"""Read a CSV and return one event-detection result for each article row."""
	print("=" * 60, flush=True)
	print("AGENT 1 - EVENT DETECTION", flush=True)
	print("=" * 60, flush=True)
	print("\nLoading news file...", flush=True)

	try:
		df = pd.read_csv(file_path)
	except pd.errors.EmptyDataError:
		df = pd.DataFrame()

	# Clean missing CSV cells and skip rows that contain no article data.
	articles = []
	for row in df.to_dict(orient="records"):
		article = {
			str(key): None if pd.isna(value) else str(value)
			for key, value in row.items()
		}
		if article.get("id") in (None, "") and article.get("article_id") not in (None, ""):
			article["id"] = article["article_id"]
		if not any(value not in (None, "") for value in article.values()):
			continue
		articles.append(article)

	print(f"Found {len(articles)} articles.", flush=True)
	results = []
	relevant_count = 0
	error_count = 0

	for article_number, article in enumerate(articles, start=1):
		headline = article.get("headline") or "(no headline)"
		print(f"\nProcessing article {article_number}/{len(articles)}...", flush=True)
		print("\nCalling Ollama...", flush=True)

		result = detect_event(article)
		results.append(result)
		print("\nResult:", flush=True)
		industries = ", ".join(result["potentially_affected_industries"]) or "None"
		print(f"Article ID: {result['article_id']}", flush=True)
		print(f"Headline: {headline}", flush=True)
		print(f"Relevant: {result['relevant']}", flush=True)
		print(f"Event Type: {result['event_type']}", flush=True)
		print(f"Price Direction: {result['price_direction']}", flush=True)
		print(f"Affected Industries: {industries}", flush=True)
		print(f"Confidence: {result['confidence']:.2f}", flush=True)
		print(f"Reason: {result['reason']}", flush=True)

		if result["reason"].startswith("ERROR:"):
			error_count += 1
			error_reason = result["reason"][len("ERROR: "):]
			if error_reason == "Ollama request timed out.":
				print("Ollama timeout - moving to next article", flush=True)
			else:
				print(f"Error: {error_reason}", flush=True)
		else:
			if result["relevant"]:
				relevant_count += 1

	print("\n" + "=" * 60, flush=True)
	print("AGENT 1 SUMMARY", flush=True)
	print("=" * 60, flush=True)
	print(f"\nTotal articles: {len(articles)}", flush=True)
	print(f"Relevant: {relevant_count}", flush=True)
	print(f"Not Relevant: {len(articles) - relevant_count - error_count}", flush=True)
	print(f"Errors: {error_count}", flush=True)
	return results


# Preserve the existing app.py import while exposing the requested function name.
def process_news_csv(file_path: str | Path) -> list[dict[str, Any]]:
	return process_news_file(file_path)


def main(data_source: str | None = None) -> None:
	selected_source = (data_source or DATA_SOURCE).strip().upper()
	if selected_source not in {"CSV", "LIVE"}:
		selected_source = "CSV"
	news_file = LIVE_NEWS_CSV_PATH if selected_source == "LIVE" else PROJECT_ROOT / "data" / "news.csv"
	output_file = PROJECT_ROOT / "outputs" / "agent1_results.json"
	results = process_news_file(news_file)
	output_file.parent.mkdir(parents=True, exist_ok=True)
	output_file.write_text(
		json.dumps(results, ensure_ascii=False, indent=2),
		encoding="utf-8",
	)
	print(f"\nStructured results saved to: {output_file}", flush=True)


if __name__ == "__main__":
	main()
